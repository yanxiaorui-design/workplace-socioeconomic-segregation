"""Place segregation and count-weighted worker exposure; one city per call."""
import numpy as np
import pandas as pd

GROUPS = ['Q1', 'Q2', 'Q3', 'Q4']


def clean_flows(frame):
    """Validate one-city aggregate flows and retain the working-age analysis sample."""
    required = ['home_polyid', 'work_1km', 'ses_group', 'age_type', 'gender_type', 'count']
    if set(required) - set(frame):
        raise ValueError(f'Missing flow columns: {set(required) - set(frame)}')
    d = frame.copy()
    for col in ['home_polyid', 'work_1km']:
        if d[col].isna().any():
            raise ValueError(f'Missing {col}')
        d[col] = d[col].astype(str)
    d['count'] = pd.to_numeric(d['count'], errors='raise')
    if not np.isfinite(d['count']).all() or (d['count'] < 0).any() or (d['count'] % 1 != 0).any():
        raise ValueError('Counts must be finite nonnegative integers, not expansion weights')
    d['count'] = d['count'].astype('int64')
    d['age_type'] = pd.to_numeric(d['age_type'], errors='coerce')
    d['gender_type'] = pd.to_numeric(d['gender_type'], errors='coerce')
    d = d[d.age_type.isin(range(3, 11)) & d.gender_type.isin([1, 2]) & d.ses_group.isin(GROUPS) & (d['count'] > 0)].copy()
    if d.empty:
        raise ValueError('No eligible working-age flows')
    if 'homecity' in d and d.homecity.nunique() != 1:
        raise ValueError('Run each city separately to avoid grid-ID collisions')
    d['age_group'] = np.select([d.age_type <= 5, d.age_type <= 8], ['25-39', '40-54'], default='55-64')
    return d


def workplace(frame, minimum_workers=30):
    """Return grid SES-composition scores after the inclusive worker threshold."""
    d = clean_flows(frame)
    counts = d.pivot_table(index='work_1km', columns='ses_group', values='count', aggfunc='sum', fill_value=0).reindex(columns=GROUPS, fill_value=0)
    # Reindexing above preserves absent SES groups as zero shares in the formula.
    n = counts.sum(axis=1)
    score = (2 / 3) * counts.div(n, axis=0).sub(.25).abs().sum(axis=1)
    out = pd.DataFrame({'workplace_segregation': score, 'total_count': n})
    return out[out.total_count >= minimum_workers].reset_index()


def summarize(frame, minimum_workers=30):
    """Return grid scores, group exposures and distinct place/worker summaries."""
    d = clean_flows(frame)
    grids = workplace(d, minimum_workers)
    if grids.empty:
        raise ValueError('No work grids meet the minimum worker count')
    # Only retained work grids enter either the exposure numerator or denominator.
    merged = d.merge(grids[['work_1km', 'workplace_segregation']], on='work_1km', how='inner', validate='many_to_one')
    rows = []
    for kind in [None, 'ses_group', 'gender_type', 'age_group']:
        groups = [('all', merged)] if kind is None else merged.groupby(kind, sort=True)
        for label, sub in groups:
            rows.append({'group_type': kind or 'all', 'group': str(label), 'workers': int(sub['count'].sum()), 'worker_experienced_segregation': np.average(sub.workplace_segregation, weights=sub['count'])})
    city = {'workplace_median': float(grids.workplace_segregation.median()), 'workplace_mean': float(grids.workplace_segregation.mean()), 'worker_experienced_mean': float(np.average(grids.workplace_segregation, weights=grids.total_count)), 'retained_grids': len(grids), 'eligible_workers': int(d['count'].sum()), 'retained_workers': int(merged['count'].sum())}
    return grids, pd.DataFrame(rows), city
