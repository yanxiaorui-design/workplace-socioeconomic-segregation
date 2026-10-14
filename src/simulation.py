"""Uniform individual assignment from complete network-time feasible choice sets."""
import numpy as np
import pandas as pd
from .segregation import clean_flows


def build_routes(flows, grids, router, max_requests=1000):
    """All other 250 m grids in the same city; no radial candidate truncation."""
    g = validate_grids(grids)
    homes = sorted(clean_flows(flows).home_polyid.unique())
    if not set(homes) <= set(g.index):
        raise ValueError('Home grid coordinates missing')
    pairs = len(homes) * (len(g) - 1)
    if pairs > max_requests:
        raise ValueError(f'{pairs} directed pairs exceed max_requests={max_requests}; review quota and explicitly raise the limit')
    rows = []
    for home in homes:
        origin = tuple(g.loc[home, ['lon_gcj02', 'lat_gcj02']].astype(float))
        for dest in sorted(g.index):
            if dest == home:
                continue
            result = router.route(origin, tuple(g.loc[dest, ['lon_gcj02', 'lat_gcj02']].astype(float)))
            rows.append({'home_polyid': home, 'work_polyid': dest, **result})
    return pd.DataFrame(rows)


def validate_grids(grids):
    """Require a unique 250 m grid identifier and a nonmissing 1 km parent."""
    g = grids.copy()
    for c in ['polyid', 'work_1km']:
        if g[c].isna().any():
            raise ValueError(f'Missing grid mapping: {c}')
        g[c] = g[c].astype(str)
    if g.polyid.duplicated().any():
        raise ValueError('Grid IDs must be unique within each city')
    if 'city_code' in g and g.city_code.nunique() != 1:
        raise ValueError('Supply one city grid table per run')
    return g.set_index('polyid')


def simulate(flows, grids, routes, seconds=900, seed=20261012):
    """Draw individual destinations uniformly from complete network-time choice sets."""
    if seconds <= 0:
        raise ValueError('Walking threshold must be positive')
    d, g, r = clean_flows(flows), validate_grids(grids), routes.copy()
    for c in ['home_polyid', 'work_polyid']:
        r[c] = r[c].astype(str)
    if r.duplicated(['home_polyid', 'work_polyid']).any():
        raise ValueError('Duplicate directed routes')
    if not r.status.isin(['ok', 'no_route']).all():
        raise ValueError('Unresolved route errors cannot be treated as unreachable')
    good = r.status.eq('ok')
    for col in ['duration_s', 'distance_m']:
        values = pd.to_numeric(r.loc[good, col], errors='raise')
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid routing values')
        r.loc[good, col] = values
    rng = np.random.default_rng(seed)
    records = []
    for home, subset in d.groupby('home_polyid', sort=True):
        if home not in g.index:
            raise ValueError('Home is missing from grid table')
        local = r[r.home_polyid.eq(home)]
        # Missing pairs cannot be assumed unreachable: doing so changes the choice set.
        if set(local.work_polyid) != set(g.index) - {home}:
            raise ValueError(f'Incomplete all-other-grid route matrix for home {home}')
        candidates = sorted(local.loc[local.status.eq('ok') & (local.duration_s <= seconds), 'work_polyid'])
        if not candidates:
            raise ValueError(f'No reachable alternative grid for home {home}; no nearest-grid fallback')
        for row in subset.sort_values(['ses_group', 'age_type', 'gender_type', 'work_1km']).to_dict('records'):
            # Exactly the distribution of independent uniform individual draws, without expansion.
            counts = rng.multinomial(int(row['count']), np.ones(len(candidates)) / len(candidates))
            for dest, n in zip(candidates, counts):
                if n:
                    record = {k: row[k] for k in ['home_polyid','ses_group','age_type','gender_type']}
                    record.update({'work_polyid': dest, 'work_1km': str(g.loc[dest, 'work_1km']), 'count': int(n)})
                    if 'homecity' in row:
                        record['homecity'] = row['homecity']
                    records.append(record)
    out = pd.DataFrame(records)
    keys = [c for c in out if c != 'count']
    out = out.groupby(keys, as_index=False, dropna=False)['count'].sum()
    if out['count'].sum() != d['count'].sum():
        raise AssertionError('Simulation did not conserve workers')
    return out
