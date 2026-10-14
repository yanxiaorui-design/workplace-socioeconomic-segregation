"""City-specific IDW and commuter-weighted SES quartiles (Supplementary Note 3)."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import CRS


def idw(xy_known, values, xy_target, projected_crs, k=12, power=2):
    """Interpolate positive prices using nearest neighbors in a metre-based CRS."""
    crs = CRS.from_user_input(projected_crs)
    if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in crs.axis_info[:2]):
        raise ValueError('IDW requires projected coordinates in metres')
    xy = np.asarray(xy_known, float)
    target = np.asarray(xy_target, float)
    v = np.asarray(values, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or target.ndim != 2 or target.shape[1] != 2 or len(xy) != len(v) or len(xy) == 0:
        raise ValueError('Invalid coordinate/value shapes or empty housing observations')
    if not all(np.isfinite(a).all() for a in [xy, target, v]) or (v <= 0).any() or k < 1 or power <= 0:
        raise ValueError('Finite coordinates and positive prices, k and power required')
    distance, index = cKDTree(xy).query(target, k=list(range(1, min(k, len(xy)) + 1)))
    # A small distance floor matches the stated IDW rule at coincident points.
    distance = np.maximum(distance, 1e-10)
    weights = distance ** (-power)
    return (weights * v[index]).sum(axis=1) / weights.sum(axis=1)


def assign_quartiles(prices, counts):
    """Use commuter-weighted thresholds; equal prices stay in the same SES group."""
    p, w = np.asarray(prices, float), np.asarray(counts, float)
    valid = np.isfinite(p) & np.isfinite(w) & (w > 0)
    if not valid.any():
        raise ValueError('No valid prices with positive commuter counts')
    order = np.argsort(p[valid], kind='stable')
    sorted_p, sorted_w = p[valid][order], w[valid][order]
    # Left search implements the lowest price reaching each cumulative share.
    thresholds = sorted_p[np.searchsorted(np.cumsum(sorted_w), sorted_w.sum() * np.array([.25, .5, .75]), side='left')]
    labels = np.full(len(p), None, dtype=object)
    labels[valid] = np.array(['Q1', 'Q2', 'Q3', 'Q4'])[np.searchsorted(thresholds, p[valid], side='left')]
    return labels, thresholds


def interpolate_homes(communities, homes, projected_crs, k=12, power=2):
    """Inputs must already be clipped to one city and projected in the same CRS."""
    c = communities.copy()
    for col in ['x', 'y', 'price']:
        c[col] = pd.to_numeric(c[col], errors='coerce')
    c = c.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'price'])
    c = c[c.price > 0].drop_duplicates(['community_name', 'x', 'y'])
    h = homes.copy()
    if h.home_polyid.duplicated().any():
        raise ValueError('Each home grid must occur once')
    h['house_pric'] = idw(c[['x', 'y']], c.price, h[['x', 'y']], projected_crs, k, power)
    h['ses_group'], thresholds = assign_quartiles(h.house_pric, h['count'])
    return h, thresholds
