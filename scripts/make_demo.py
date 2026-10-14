"""Generate fully artificial demo tables. No source mobile records are read."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]/'data/demo'
root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(42)
n=16
ids=[f'g{i:02}' for i in range(n)]
x=np.array([i%4*250 for i in range(n)],float)
y=np.array([i//4*250 for i in range(n)],float)
grids=pd.DataFrame({'polyid':ids,'work_1km':[f'w{i//4}' for i in range(n)],'lon_gcj02':116.4+x/85000,'lat_gcj02':39.9+y/111000,'city_code':'synthetic'})
grids.to_csv(root/'grids.csv',index=False)
homes=pd.DataFrame({'home_polyid':ids,'x':500000+x,'y':4400000+y,'count':400})
homes.to_csv(root/'homes.csv',index=False)
pd.DataFrame({'community_name':[f'synthetic_{i}' for i in range(n)],'x':500000+x,'y':4400000+y,'price':10000+np.arange(n)*1000}).to_csv(root/'communities.csv',index=False)
flows=[]
for i,h in enumerate(ids):
 for age,gender in [(3,1),(6,2),(9,1),(5,2)]:
  flows.append({'homecity':'synthetic','home_polyid':h,'work_polyid':ids[(i*3+age)%n],'work_1km':grids.work_1km.iloc[(i*3+age)%n],'age_type':age,'gender_type':gender,'ses_group':f'Q{i//4+1}','count':100})
pd.DataFrame(flows).to_csv(root/'flows.csv',index=False)
# Hand-constructed toy network costs; deliberately asymmetric and with detours.
# These are NOT Amap responses and are NOT empirical routes.
routes=[]
for i,h in enumerate(ids):
 for j,d in enumerate(ids):
  if i==j: continue
  distance=abs(x[i]-x[j])+abs(y[i]-y[j])+100*((i+j)%3)
  duration=distance/1.2+((i*7+j)%4)*15
  routes.append({'home_polyid':h,'work_polyid':d,'status':'ok','duration_s':duration,'distance_m':distance,'source':'synthetic_toy_network'})
pd.DataFrame(routes).to_csv(root/'routes_synthetic.csv',index=False)
rows=[]
for region in ['East','Central','West','Northeast']:
 for city in range(3):
  for grid in range(35):
   v=rng.lognormal(2,.5,10)
   rows.append({'region':region,'city_code':f'{region}_{city}','workplace_segregation':np.clip(.35+.03*city+.06*np.log1p(v[0])+rng.normal(0,.04),0,1),'commuting_distance':v[0]*1000,'Dis2Center':v[1]*1000,'nearest_dis':v[2]*100,'roadLength':v[3]*100,'total_count':v[4]*30,'POI_count':20+v[5],'POI_entropy':v[6]/10,'Elite Business & Finance':v[7],'Industrial & Logistics':v[8],'Public & Knowledge':v[9]})
pd.DataFrame(rows).to_csv(root/'covariates.csv',index=False)
print('Generated synthetic demo only')
