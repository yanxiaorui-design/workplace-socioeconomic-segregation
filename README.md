# Workplace socioeconomic segregation

Code for calculating workplace and worker-experienced socioeconomic segregation, interpolating residential housing prices, fitting city fixed-effects regression models, and simulating a 15-minute walking scenario. Version 1.0.4.

The repository includes a fully synthetic demonstration dataset. Confidential mobility records and empirical study results are not distributed. The demo verifies software execution rather than reproduction of the manuscript results.

## Layout

```text
workplace-socioeconomic-segregation/
├── main.py                     # Run the demo or an individual analysis stage
├── config.json                 # Runtime settings and random seeds
├── requirements.txt            # Exact tested dependencies
├── LICENSE                     # MIT license
├── CITATION.cff                 # Software citation and version
├── src/
│   ├── housing.py              # Housing-price interpolation and SES assignment
│   ├── segregation.py          # Workplace scores and worker-experienced segregation
│   ├── regression.py           # Regression analysis and variable importance
│   ├── routing.py              # Amap walking API requests and route caching
│   └── simulation.py           # Walking-scenario simulation
├── scripts/
│   ├── make_demo.py            # Regenerate the synthetic input tables
│   └── verify_demo.py          # Compare demo outputs with reference results
├── data/
│   ├── demo/                  # Six synthetic input tables
│   └── expected_results.json  # Reference results for 17 output files
├── tests/
│   ├── test_core.py            # Mathematical and statistical checks
│   └── test_pipeline.py        # End-to-end demo test
└── docs/
    └── DATA.md                 # Required input fields and units
```

## System requirements

Tested on Windows with Python 3.12. Dependencies include NumPy, pandas, SciPy, statsmodels, requests, pyproj and pytest; exact versions are listed in `requirements.txt`. Linux and macOS have not been tested. No specialized hardware is required; a standard laptop with approximately 8 GB RAM is sufficient for the demo. Research-data requirements depend on city size and the number of routing pairs.

## Installation

From the folder containing `main.py`:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Estimated installation time: 2–10 minutes with binary wheels and a normal internet connection. If PowerShell activation is unavailable, use `.venv\Scripts\python.exe` instead of `python`.

## Demo

```bash
python main.py demo
python scripts/verify_demo.py --output results/demo
```

The demo exercises all analysis modules using synthetic inputs. No API key or internet access is needed after installation.

Outputs are written to `results/demo/`. The observed synthetic example contains 6,400 workers and four retained work grids:

```text
workplace_median = 0.25
workplace_mean = 0.25
worker_experienced_mean = 0.25
Verified 17 synthetic reference outputs
```

The warmed pipeline runs in a few seconds; allow approximately 1–2 minutes for cold imports and tests. Run all nine automated checks with:

```bash
python -m pytest -q
```

## Run with research data

Place authorized, prepared inputs under `data/private/`. Run each city separately for housing, segregation and simulation; regression accepts a combined multi-city covariate table. See [data definitions](docs/DATA.md) for required fields and units.

```bash
# Interpolate housing prices and assign SES groups
python main.py housing --communities data/private/communities.csv --homes data/private/homes.csv --projected-crs EPSG:32650 --output results/city/housing

# Calculate observed workplace and worker-experienced segregation
python main.py segregation --flows data/private/flows.csv --output results/city/observed

# Estimate regional city fixed-effects models and relative importance
python main.py regression --covariates data/private/covariates.csv --output results/regression
```

Use the appropriate projected CRS in metres for the city; EPSG:32650 is an example. Before calculating segregation, join the housing output fields `house_pric` and `ses_group` to commuter flows by `home_polyid`. Raw mobile-signal processing, home/work inference and geographic covariate preparation occur upstream.

For the walking counterfactual, set `AMAP_API_KEY` in your local environment and provide all city 250 m grid centroids in GCJ-02 coordinates with their 1 km parent IDs:

```powershell
$env:AMAP_API_KEY = 'YOUR_KEY'
```

```bash
python main.py routes --flows data/private/flows.csv --grids data/private/grids.csv --cache cache/city.sqlite --output results/city/routing --max-requests 1000
python main.py simulate --flows data/private/flows.csv --grids data/private/grids.csv --routes results/city/routing/routes.csv --output results/city/simulation
```

The private cache supports resuming route collection; `--offline` requires a complete cache. Review pair counts and API quotas before increasing `--max-requests`. Failed or incomplete routing inputs stop execution. Live API responses reflect the service at the query time.

## Settings and output files

Use `config.json` to inspect runtime settings, or pass `--config path/to/settings.json`. Run `python main.py --help` to see command-line options.

Outputs include housing estimates, segregation tables, regression tables, simulated flows and summary files. Each run also records configuration and input hashes in `run_metadata.json`. Generated results and private caches are excluded from version control. Use a separate `--output` directory for each analysis.

## Data availability and license

Mobile-phone data are not publicly available to protect privacy. Research access may be requested from the corresponding author identified in the article, subject to provider approval and applicable restrictions. Housing prices originate from [Anjuke](https://www.anjuke.com/), POIs from [Amap](https://lbs.amap.com/), and road geometry from [OpenStreetMap](https://www.openstreetmap.org/) (© OpenStreetMap contributors; [ODbL](https://www.openstreetmap.org/copyright)). Walking routes are obtained separately through the [Amap API](https://lbs.amap.com/api/webservice/guide/api/direction).

This package contains the analysis modules listed above; it does not include all upstream processing or figure-generation scripts. Tests use synthetic data and mocked API responses; empirical and live-API validation requires authorized research inputs. Code and synthetic data use the [MIT license](LICENSE); third-party data retain their own terms. Citation metadata are provided in [CITATION.cff](CITATION.cff).
