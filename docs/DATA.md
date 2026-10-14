# Data dictionary and preparation

All CSVs use UTF-8. IDs must be nonmissing strings and unique where indicated. Process each city separately unless the stage explicitly accepts a multi-city covariate table.

| Table | Required columns | Interpretation |
|---|---|---|
| flows.csv | home_polyid, work_1km, ses_group, age_type, gender_type, count | Aggregated commuter records. ses_group is Q1–Q4; count is nonnegative integer sample count. Optional homecity and work_polyid. |
| homes.csv | home_polyid, x, y, count | One residential 250 m centroid per row, metre coordinates, total sampled commuters residing there. |
| communities.csv | community_name, x, y, price | Community average selling price per m²; one city, same projected CRS as homes. Invalid/nonpositive prices removed; duplicate name+coordinates removed. |
| grids.csv | polyid, work_1km, lon_gcj02, lat_gcj02 | All 250 m city-grid centroids and their unique 1 km parent IDs; optional city_code. Coordinates for routing are GCJ-02 degrees. |
| routes.csv | home_polyid, work_polyid, status, duration_s, distance_m | Complete directed matrix for every occupied eligible home and every other city grid. status is ok or no_route. API output also records endpoint, coordinates, timestamp and CRS. |
| covariates.csv | city_code, region, workplace_segregation, commuting_distance, Dis2Center, nearest_dis, roadLength, total_count, POI_count, POI_entropy, Elite Business & Finance, Industrial & Logistics, Public & Knowledge | One retained work grid per row. Distances and road length in metres; worker and POI counts in consistent units. region is East/Central/West/Northeast. |

## Input conventions

Age codes 3–5 correspond to 25–39, 6–8 to 40–54, and 9–10 to 55–64. Gender codes 1 and 2 represent male and female, respectively. The software expects preprocessed flow tables, prepared covariates and a unique fine-grid-to-parent-grid mapping.

Housing inputs must share a projected coordinate reference system with metre units. Amap routing inputs require GCJ-02 longitude/latitude. Coordinate conversion and geographic preparation must be completed before using these tables.

## Synthetic demonstration

All six demo tables were generated with `scripts/make_demo.py`. No row is copied, perturbed, sampled or anonymized from the confidential research dataset. Grid coordinates and prices are artificial. `routes_synthetic.csv` contains toy costs, not live routing responses. The regression input is a separate synthetic table. Demo outputs are not empirical study results.
