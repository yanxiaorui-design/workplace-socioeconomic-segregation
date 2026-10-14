"""City fixed effects, HC1 robust inference, and exact within-R2 LMG."""
from itertools import combinations
from math import factorial
import numpy as np
import pandas as pd
import statsmodels.api as sm

LOG_COLUMNS = ['commuting_distance', 'Dis2Center', 'nearest_dis', 'roadLength', 'total_count', 'POI_count']
RAW_COLUMNS = ['POI_entropy', 'Elite Business & Finance', 'Industrial & Logistics', 'Public & Knowledge']
PREDICTORS = ['log_' + x for x in LOG_COLUMNS] + RAW_COLUMNS


def prepare(frame, minimum_pois=10, missing='drop'):
    """Filter the pooled sample, transform covariates and retain scaling parameters."""
    cols = ['workplace_segregation'] + LOG_COLUMNS + RAW_COLUMNS
    d = frame.copy()
    if d[['city_code', 'region']].isna().any().any():
        raise ValueError('Missing city/region identifiers')
    d[cols] = d[cols].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    if missing == 'drop':
        d = d.dropna(subset=cols)
    elif missing == 'legacy_zero':
        # Explicit historical compatibility; never impute the dependent variable.
        d = d.dropna(subset=['workplace_segregation'])
        d[LOG_COLUMNS + RAW_COLUMNS] = d[LOG_COLUMNS + RAW_COLUMNS].fillna(0)
    else:
        raise ValueError('missing must be drop or legacy_zero')
    d = d[d.POI_count >= minimum_pois].copy()
    if d.empty or (d[LOG_COLUMNS + RAW_COLUMNS] < 0).any().any() or not d.workplace_segregation.between(0, 1).all():
        raise ValueError('Empty sample or invalid covariates/outcome')
    for c in LOG_COLUMNS:
        d['log_' + c] = np.log1p(d[c])
    means, scales = d[PREDICTORS].mean(), d[PREDICTORS].std(ddof=1)
    if (scales <= 0).any() or scales.isna().any():
        raise ValueError('Constant or insufficient covariate observations')
    d[PREDICTORS] = (d[PREDICTORS] - means) / scales
    return d, pd.DataFrame({'mean': means, 'sd': scales})


def lmg(y, x, names):
    """Allocate within-city R-squared by averaging marginal subset contributions."""
    p = x.shape[1]
    sst = np.dot(y, y)
    if sst <= 0:
        raise ValueError('No within-city variation in the outcome')
    cache = {(): 0.0}
    for size in range(1, p + 1):
        for sub in combinations(range(p), size):
            a = x[:, sub]
            residual = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
            cache[sub] = 1 - np.dot(residual, residual) / sst
    contributions = []
    for j in range(p):
        value = 0.0
        for size in range(p):
            for sub in combinations([i for i in range(p) if i != j], size):
                weight = factorial(size) * factorial(p - size - 1) / factorial(p)
                value += weight * (cache[tuple(sorted(sub + (j,)))] - cache[sub])
        contributions.append(value)
    total = sum(contributions)
    return pd.DataFrame({'variable': names, 'lmg_R2_contribution': contributions, 'lmg_percent': np.array(contributions) * 100 / total if total > 0 else np.nan})


def fit_region(d):
    """Fit one regional city-FE model with HC1 covariance and t-based intervals."""
    # Explicit city dummies produce the FE estimator and account for FE degrees of freedom.
    x = pd.concat([d[PREDICTORS], pd.get_dummies(d.city_code, prefix='city', drop_first=True, dtype=float)], axis=1)
    x = sm.add_constant(x).astype(float)
    if len(d) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError('Insufficient observations or collinear regional FE design')
    result = sm.OLS(d.workplace_segregation, x).fit(cov_type='HC1', use_t=True)
    cols = ['workplace_segregation'] + PREDICTORS
    within = d[cols] - d.groupby('city_code')[cols].transform('mean')
    importance = lmg(within.iloc[:, 0].to_numpy(), within[PREDICTORS].to_numpy(), PREDICTORS)
    ci = result.conf_int()
    coef = pd.DataFrame({'coefficient': result.params, 'robust_se': result.bse, 'p_value': result.pvalues, 'ci_low': ci[0], 'ci_high': ci[1]})
    return coef, importance, {'nobs': int(result.nobs), 'cities': int(d.city_code.nunique()), 'r2_overall_with_city_effects': result.rsquared, 'adjusted_r2_with_city_effects': result.rsquared_adj, 'r2_within': float(importance.lmg_R2_contribution.sum()), 'covariance': 'HC1', 'inference': 't, residual degrees of freedom includes city effects'}
