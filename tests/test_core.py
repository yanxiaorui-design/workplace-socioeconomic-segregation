import numpy as np
import pandas as pd
import pytest
from src.segregation import workplace, summarize
from src.housing import idw, assign_quartiles
from src.routing import AmapRouter, parse_response
from src.simulation import simulate
from src.regression import prepare, fit_region, PREDICTORS
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def flow(groups, counts, works=None):
    n=len(groups)
    return pd.DataFrame({'home_polyid':['h']*n,'work_1km':works or ['w']*n,'ses_group':groups,'count':counts,'age_type':[3]*n,'gender_type':[1]*n})


def test_segregation_extremes_and_boundary():
    assert workplace(flow(['Q1','Q2','Q3','Q4'],[10]*4)).workplace_segregation.iloc[0] == 0
    assert workplace(flow(['Q1'],[30])).workplace_segregation.iloc[0] == 1
    assert workplace(flow(['Q1'],[29])).empty
    with pytest.raises(ValueError): workplace(flow(['Q1'],[30.2]))


def test_place_median_differs_from_worker_exposure():
    d=flow(['Q1','Q1','Q2','Q3','Q4'],[400,25,25,25,25],['a','b','b','b','b'])
    _,_,s=summarize(d)
    assert s['workplace_median']==.5
    assert s['worker_experienced_mean']==.8


def test_idw_and_weighted_ties():
    assert idw([[0,0]],[100],[[10,20]],'EPSG:32650')[0]==100
    assert np.allclose(idw([[0,0],[2,0]],[100,300],[[1,0]],'EPSG:32650'),200)
    labels,q=assign_quartiles([10,10,20,30],[1,1,1,1])
    assert list(labels)==['Q1','Q1','Q3','Q4']
    assert list(q)==[10,10,20]
    with pytest.raises(ValueError): idw([[0,0]],[100],[[1,1]],'EPSG:4326')


def test_network_time_controls_simulation():
    f=flow(['Q1'],[100])
    g=pd.DataFrame({'polyid':['h','a','b'],'work_1km':['H','A','B']})
    r=pd.DataFrame({'home_polyid':['h','h'],'work_polyid':['a','b'],'status':['ok','ok'],'duration_s':[900,901],'distance_m':[1500,500]})
    out=simulate(f,g,r)
    assert set(out.work_polyid)=={'a'}
    assert out['count'].sum()==100
    with pytest.raises(ValueError): simulate(f,g,r.iloc[:1])
    r.loc[0,'duration_s']=902
    with pytest.raises(ValueError): simulate(f,g,r)


def test_api_errors_and_no_route():
    with pytest.raises(RuntimeError): parse_response({'status':'0','infocode':'10001'})
    assert parse_response({'status':'1','route':{'paths':[]}})['status']=='no_route'
    result=parse_response({'status':'1','route':{'paths':[{'duration':'850','distance':'1300'},{'duration':'900','distance':'1000'}]}})
    assert result['duration_s']==850 and result['distance_m']==1300


def test_api_cache_offline_and_direction(tmp_path, monkeypatch):
    router=AmapRouter(tmp_path/'cache.sqlite',key='TEST_KEY',interval=0)
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'status':'1','route':{'paths':[{'duration':'800','distance':'900'}]}}
    calls=[]
    def get(*args,**kwargs):
        calls.append(kwargs)
        return Response()
    monkeypatch.setattr(router.session,'get',get)
    router.route((116,39),(116.01,39))
    router.offline=True
    assert router.route((116,39),(116.01,39))['duration_s']==800
    assert len(calls)==1
    with pytest.raises(RuntimeError): router.route((116.01,39),(116,39))
    router.close()
    assert b'TEST_KEY' not in (tmp_path/'cache.sqlite').read_bytes()


def test_fe_matches_within_and_lmg_sums():
    d,_=prepare(pd.read_csv(ROOT/'data/demo/covariates.csv'))
    d=d[d.region=='East']
    coef,lmg,stats=fit_region(d)
    cols=['workplace_segregation']+PREDICTORS
    w=d[cols]-d.groupby('city_code')[cols].transform('mean')
    beta=np.linalg.lstsq(w[PREDICTORS],w.workplace_segregation,rcond=None)[0]
    assert np.allclose(coef.loc[PREDICTORS,'coefficient'],beta)
    assert np.isclose(lmg.lmg_percent.sum(),100)
    assert np.isclose(lmg.lmg_R2_contribution.sum(),stats['r2_within'])


def test_demo_simulation_conserves_demographics_and_repeats():
    f=pd.read_csv(ROOT/'data/demo/flows.csv')
    g=pd.read_csv(ROOT/'data/demo/grids.csv')
    r=pd.read_csv(ROOT/'data/demo/routes_synthetic.csv')
    a=simulate(f,g,r,seed=42)
    b=simulate(f,g,r,seed=42)
    pd.testing.assert_frame_equal(a,b)
    keys=['home_polyid','ses_group','age_type','gender_type']
    pd.testing.assert_series_equal(f.groupby(keys)['count'].sum(),a.groupby(keys)['count'].sum())
