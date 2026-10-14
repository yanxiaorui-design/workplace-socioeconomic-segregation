"""Command line entry point; all paths are explicit and raw data stay local."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
from src.segregation import summarize
from src.housing import interpolate_homes
from src.regression import prepare, fit_region
from src.routing import AmapRouter
from src.simulation import build_routes, simulate

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    """Read identifiers as strings to preserve leading zeros and join keys."""
    return pd.read_csv(path, dtype={c: 'str' for c in ['home_polyid','work_polyid','work_1km','polyid','homecity','city_code']})


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def metrics(flows, output, config):
    """Write grid, population-group and city summaries for one scenario."""
    output.mkdir(parents=True, exist_ok=True)
    grids, exposure, summary = summarize(flows, config['minimum_workers'])
    grids.to_csv(output/'workplace.csv', index=False)
    exposure.to_csv(output/'worker_experienced.csv', index=False)
    write_json(output/'city_summary.json', summary)
    return summary


def regress(data, out, config):
    """Standardize once across the sample, then fit each regional model."""
    out.mkdir(parents=True, exist_ok=True)
    d, scaling = prepare(data, config['minimum_pois'], config['missing_regression'])
    scaling.to_csv(out/'scaling.csv', index_label='variable')
    summaries = {}
    for region, sub in d.groupby('region', sort=True):
        if region not in ['East','Central','West','Northeast']:
            raise ValueError('region must be East, Central, West or Northeast')
        coef, importance, info = fit_region(sub)
        coef.to_csv(out/f'{region}_coefficients.csv', index_label='variable')
        importance.to_csv(out/f'{region}_lmg.csv', index=False)
        summaries[region] = info
    write_json(out/'fit_summary.json', summaries)


def simulations(flows, grids, routes, output, config):
    """Compare seeded counterfactual replicates with the same baseline inputs."""
    baseline = metrics(flows, output/'baseline', config)
    rows = []
    for rep in range(config['replicates']):
        seed = config['seed'] + rep
        simulated = simulate(flows, grids, routes, config['walking_seconds'], seed)
        folder = output/f'replicate_{rep:03d}'
        summary = metrics(simulated, folder, config)
        simulated.to_csv(folder/'simulated_flows.csv', index=False)
        changes = {f'delta_{k}': summary[k] - baseline[k] for k in ['workplace_median', 'worker_experienced_mean']}
        rows.append({'replicate': rep, 'seed': seed, **summary, **changes})
    table = pd.DataFrame(rows)
    table.to_csv(output/'replicate_summary.csv', index=False)
    columns = ['workplace_median', 'worker_experienced_mean', 'delta_workplace_median', 'delta_worker_experienced_mean']
    table[columns].agg(['mean','std','min','max']).to_csv(output/'monte_carlo_summary.csv', index_label='statistic')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['demo','segregation','housing','regression','routes','simulate'])
    p.add_argument('--config', type=Path, default=ROOT/'config.json')
    p.add_argument('--output', type=Path, default=ROOT/'results/demo')
    p.add_argument('--flows', type=Path)
    p.add_argument('--grids', type=Path)
    p.add_argument('--routes', type=Path)
    p.add_argument('--communities', type=Path)
    p.add_argument('--homes', type=Path)
    p.add_argument('--covariates', type=Path)
    p.add_argument('--projected-crs')
    p.add_argument('--cache', type=Path, default=ROOT/'cache/amap.sqlite')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--max-requests', type=int, default=1000)
    a = p.parse_args()
    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    if config['replicates'] < 1:
        p.error('replicates must be positive')
    a.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if a.stage == 'demo':
        demo = ROOT/'data/demo'
        flows, grids, routes = [read_csv(demo/name) for name in ['flows.csv','grids.csv','routes_synthetic.csv']]
        h, q = interpolate_homes(read_csv(demo/'communities.csv'), read_csv(demo/'homes.csv'), 'EPSG:32650', config['idw_neighbors'], config['idw_power'])
        h.to_csv(a.output/'housing.csv', index=False)
        write_json(a.output/'quartile_thresholds.json', q.tolist())
        # The demo runs the housing-to-SES-to-segregation chain, not unrelated examples.
        flows = flows.drop(columns='ses_group').merge(h[['home_polyid','ses_group']], on='home_polyid', validate='many_to_one')
        metrics(flows, a.output/'observed', config)
        regress(read_csv(demo/'covariates.csv'), a.output/'regression', config)
        simulations(flows, grids, routes, a.output/'simulation', config)
    elif a.stage == 'segregation':
        if a.flows is None: p.error('--flows required')
        metrics(read_csv(a.flows), a.output, config)
    elif a.stage == 'housing':
        if not all([a.communities,a.homes,a.projected_crs]): p.error('--communities, --homes and --projected-crs required')
        h, q = interpolate_homes(read_csv(a.communities),read_csv(a.homes),a.projected_crs,config['idw_neighbors'],config['idw_power'])
        h.to_csv(a.output/'housing.csv', index=False)
        write_json(a.output/'quartile_thresholds.json',q.tolist())
    elif a.stage == 'regression':
        if a.covariates is None: p.error('--covariates required')
        regress(read_csv(a.covariates), a.output, config)
    elif a.stage == 'routes':
        if not all([a.flows,a.grids]): p.error('--flows and --grids required')
        a.cache.parent.mkdir(parents=True,exist_ok=True)
        router = AmapRouter(a.cache, offline=a.offline)
        try:
            routes = build_routes(read_csv(a.flows),read_csv(a.grids),router,a.max_requests)
            routes.to_csv(a.output/'routes.csv',index=False)
        finally:
            router.close()
    elif a.stage == 'simulate':
        if not all([a.flows,a.grids,a.routes]): p.error('--flows, --grids and --routes required')
        simulations(read_csv(a.flows),read_csv(a.grids),read_csv(a.routes),a.output,config)
    inputs = [p for p in [a.config, a.flows, a.grids, a.routes, a.communities, a.homes, a.covariates] if p is not None]
    if a.stage == 'demo': inputs += sorted((ROOT/'data/demo').glob('*.csv'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    write_json(a.output/'run_metadata.json', {'stage':a.stage,'input_sha256':hashes,'configuration':config,'elapsed_seconds':time.perf_counter()-start,'synthetic_demo':a.stage=='demo'})
    print(f'Completed {a.stage}: {a.output}')


if __name__ == '__main__':
    main()
