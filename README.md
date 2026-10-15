# Analysing Air Quality and Its Link to Urban Factors

Data Science Group Assignment 1, Option B. We take one year of hourly roadside air quality readings,
clean them, explore how pollution changes by hour, weekday, season and weather, train regression models
that predict carbon monoxide (CO) from urban and environmental factors, and turn the results into
decisions a city traffic office could act on.

**Group members:** _(names and IDs)_  
**Repository:** _(GitHub URL once pushed)_

## Repository contents

| Path | What it is |
|---|---|
| `air_quality_option_b.ipynb` | The full workflow, with outputs saved, from loading the data to the policy conclusion |
| `models/co_best_model.joblib` | Best-scoring trained pipeline (preprocessing + model), saved with joblib |
| `models/co_urban_factors_model.joblib` | Random Forest trained on time and weather only, used for the scenario planning grid |
| `models/model_card.json` | What each model expects as input, the training and test periods, and its test scores |
| `figures/` | Every chart from the notebook as a numbered PNG (used in the report) |
| `clean_hourly.csv` | The cleaned hourly table produced by Section 3 of the notebook |
| `requirements.txt` | Pinned package versions |

## How to run

**Google Colab (recommended).** Upload `air_quality_option_b.ipynb` (or open it from GitHub via
`File → Open notebook → GitHub`), then `Runtime → Run all`. The first cell installs `ucimlrepo`;
everything else is pre-installed in Colab. The dataset is fetched directly from the UCI repository, so
nothing needs to be uploaded. The run takes a few minutes; figures and models appear in the Colab
file panel under `figures/` and `models/`.

**Locally.**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook air_quality_option_b.ipynb
```

Then `Kernel → Restart & Run All`. Random seeds are fixed (`RANDOM_STATE = 42`), so re-running
reproduces the numbers in the saved outputs.

**Using the saved model without the notebook.**

```python
import json, joblib, pandas as pd
model = joblib.load("models/co_best_model.joblib")
card = json.load(open("models/model_card.json"))
row = pd.DataFrame([{...}])          # one row with the columns listed in card["best_model"]["features"]
model.predict(row)                   # CO in mg/m³
```

## Dataset source documentation

| | |
|---|---|
| **Name** | Air Quality |
| **Source** | UCI Machine Learning Repository, dataset ID 360, https://archive.ics.uci.edu/dataset/360/air+quality |
| **DOI** | 10.24432/C59K5F |
| **Creator** | Saverio De Vito (ENEA, Italy); donated 22 March 2016 |
| **Licence** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Accessed** | 15 October 2026, via the `ucimlrepo` Python package (`fetch_ucirepo(id=360)`) |
| **Contents** | 9,357 hourly records, 10 March 2004 to 4 April 2005, from a multi-sensor device at road level in a polluted area of an Italian city. 15 columns: date, time, reference-analyser concentrations of CO, NMHC, benzene, NOx and NO2, five metal-oxide sensor responses, temperature, relative humidity and absolute humidity. |
| **Missing values** | Coded as -200 in the source file |
| **Pre-processing before analysis** | None. The data is used exactly as served by `ucimlrepo`. All cleaning (datetime parsing, -200 → NaN, dropping the 90%-missing NMHC column, gap-limited interpolation of weather and sensor columns) is done inside the notebook, Sections 1 to 3, and documented there. |

Citation (APA 7):

> Vito, S. (2016). *Air Quality* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F

Original paper describing the data collection:

> De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008). On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario. *Sensors and Actuators B: Chemical, 129*(2), 750–757. https://doi.org/10.1016/j.snb.2007.09.060

## Method in one paragraph

Missing values are analysed by column, by month and by gap length before anything is filled; the
NMHC column is dropped, the target is never imputed, and weather and sensor columns are interpolated
only inside gaps of six hours or less. The exploratory analysis pairs each time-based chart with a
statistical test (Mann-Whitney for weekday vs weekend, Kruskal-Wallis across hours and months,
Spearman for weather) and counts breaches of the EU and WHO limits. Two feature sets are compared:
urban factors only (hour, weekday, weekend, holiday, rush hour, season, temperature, humidity) and the
same plus the cheap sensor readings. Linear Regression, Ridge, Random Forest and Gradient Boosting are
trained on a chronological 80/20 split with time-ordered cross-validation and evaluated with RMSE, R²,
adjusted R² and MAE. Permutation importance, partial dependence and a model-generated hour × season
scenario grid translate the results into decisions.

## Results summary

Held-out test period: 28 January to 4 April 2005 (last 20% of the timeline, chronological split).

| Model | Features | Test RMSE (mg/m³) | Test R² | Test MAE |
|---|---|---|---|---|
| Baseline: hour-of-day mean | hour only | 1.093 | 0.293 | 0.838 |
| Linear Regression | A: urban factors | 1.040 | 0.359 | 0.817 |
| Random Forest | A: urban factors | 1.031 | 0.371 | 0.775 |
| Gradient Boosting | A: urban factors | 1.028 | 0.374 | 0.767 |
| Linear Regression | B: + cheap sensors | 0.597 | 0.789 | 0.470 |
| Ridge Regression | B: + cheap sensors | 0.577 | 0.803 | 0.453 |
| Gradient Boosting | B: + cheap sensors | 0.496 | 0.854 | 0.345 |
| **Random Forest (tuned)** | **B: + cheap sensors** | **0.490** | **0.858** | **0.335** |

Key findings:

* Hour of day is the dominant factor. Mean CO is 3.7 mg/m³ at 19:00 and 0.7 mg/m³ at 05:00. Weekend daily means are 30% lower than weekday means (1.66 vs 2.36 mg/m³, Mann-Whitney p < 0.001).
* Time and weather alone explain about a third of hour-to-hour variation; adding five low-cost sensor readings raises R² to 0.86 and halves the error.
* A low-cost CO sensor calibrated in spring under-reads by about 0.6 mg/m³ by December, so sensor networks need seasonal recalibration.
* The site never breaks the EU 8-hour CO limit but exceeds the WHO 24-hour CO guideline on 15 days, the EU hourly NO2 limit on 87 days and the EU annual benzene limit.
* Recommendation: weekday rush-hour traffic measures (07:00–10:00, 17:00–21:00) all year, extended to the full working day in winter, plus a recalibrated low-cost sensor network on the busiest corridors.

Limitations: one site, one year, no particulate matter, rainfall, wind or traffic counts; associations, not causal effects. See notebook Section 10.
