# Australian Weather Data Analysis Using Python

A data cleaning, feature engineering and visualisation project on roughly 142,000 daily weather observations from 49 Australian weather stations, built in Python with Pandas, Matplotlib, Seaborn and Scikit-learn and tested with `unittest`.

---

## Overview

This project works through the [Rain in Australia dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) (`weatherAUS.csv`) from start to finish: exploring the raw data, investigating and treating missing values, merging datasets, engineering new features, sorting with lambda functions, visualising weather patterns and validating each step with automated tests.

### Objectives

- Explore the structure and characteristics of the weather dataset.
- Identify and address missing values, duplicate records and categorical inconsistencies.
- Investigate class imbalance and relationships between variables.
- Demonstrate dataset merging with Pandas.
- Create new features from existing weather measurements.
- Implement sorting with Python lambda functions.
- Visualise temperature, rainfall, humidity and other weather patterns.
- Validate data-processing steps with Python's `unittest` framework.

---

## Dataset

| | |
|---|---|
| **Source** | `weatherAUS.csv` (Kaggle, Bureau of Meteorology observations) |
| **Raw size** | 142,193 rows × 24 columns |
| **Locations** | 49 weather stations across Australia |
| **Variables** | Temperature, rainfall, evaporation, sunshine, wind direction and speed, humidity, pressure, cloud cover, `RainToday`, `RainTomorrow` |
| **Cleaned size** | 140,787 rows × 20 columns, 0 missing values, 0 duplicates |

---

## Data cleaning

Missing values were investigated at the column, station and row level before any treatment was chosen.

| Step | Decision | Reason |
|------|----------|--------|
| **Dropped 4 columns** | `Sunshine`, `Evaporation`, `Cloud9am`, `Cloud3pm` | 38–48% missing, and 100% missing at 12–19 stations, so station-level imputation was impossible. Removing them cut incomplete rows from **60.3% to 20.6%**. |
| **Removed 1,406 rows** | Rows with missing `Rainfall` | `Rainfall` and `RainToday` were missing on exactly the same rows, so neither could be rebuilt from the other. This is about 0.99% of the data. |
| **Numeric imputation** | Median by station, then overall median as a fallback | Keeps local climate differences. The fallback covers stations with no readings at all (e.g. pressure at MountGinini, Newcastle, Penrith, SalmonGums). |
| **Categorical imputation** | Mode by station, then overall mode | Used for the three wind-direction columns, where a median would make no sense. |
| **Date conversion** | `Date` converted to datetime (strict `YYYY-MM-DD`) | Allows time-based analysis and feature extraction. |
| **Validation checks** | Wind directions checked against the 16 compass points; `RainToday` checked against rainfall | No invalid labels found. `RainToday = Yes` matches rainfall above 1.0 mm exactly. |

`RISK_MM` was excluded from imputation and correlation analysis because it contains next-day rainfall, which would leak information about `RainTomorrow`.

---

## Feature engineering

Starting from the merged dataset, the following columns were added (19 → 28 columns):

- **Date parts:** `Year`, `Month`, `Day`, `DayOfWeek`
- **Season:** Australian seasons (Summer = Dec–Feb, Autumn = Mar–May, Winter = Jun–Aug, Spring = Sep–Nov)
- **Daily changes:** `TempRange`, `HumidityChange`, `PressureChange`, `WindSpeedChange`

Three rows had a negative `TempRange` caused by imputation. Each was corrected by adjusting only the value that was originally missing. The engineered features were then scaled with both **Min-Max normalisation** and **Z-score standardisation** for comparison.

---

## Merging

The cleaned data was split into a **temperature** dataset and an **atmosphere** dataset and joined back together on `Date` + `Location` with `validate='one_to_one'`. The merge was checked by:

- comparing row counts and checking for duplicate keys;
- testing `inner`, `left`, `right` and `outer` joins on a deliberately trimmed copy, with `indicator=True` to show unmatched rows;
- running `pd.testing.assert_frame_equal` to confirm every merged value matches the original.

---

## Key findings

**Class imbalance.** About 78% of days are `RainTomorrow = No` and 22% are `Yes`. Any future prediction model would need suitable metrics or class balancing.

**Strongly correlated variables (|r| > 0.90):**

| Pair | r |
|------|---|
| `MaxTemp` – `Temp3pm` | 0.98 |
| `Pressure9am` – `Pressure3pm` | 0.96 |
| `MinTemp` – `Temp9am` | 0.90 |

**City comparison (Perth, Melbourne, Sydney, Brisbane, Darwin):**

- **Darwin** is hot all year (median max ≈ 33°C) with little seasonal change, and has the highest average daily rainfall (≈ 5.1 mm).
- **Melbourne** has the lowest median maximum temperature (≈ 19.5°C) and many hot-day outliers.
- **Perth, Melbourne, Sydney and Brisbane** show clear seasonal cycles, warmest in January–February and coolest in July.
- Afternoon **temperature and humidity** have a moderate negative relationship (r ≈ −0.56).

| Location | Warmest month | Coolest month |
|----------|---------------|---------------|
| Perth | January (32.4°C) | July (18.4°C) |
| Melbourne | January (27.6°C) | July (14.9°C) |
| Sydney | January (27.5°C) | July (18.0°C) |
| Brisbane | February (30.6°C) | July (21.6°C) |
| Darwin | October (34.0°C) | June (31.2°C) |

### Visualisations

- Missing values by column (horizontal bar chart)
- Correlation heatmap of weather variables
- Maximum temperature by city (box plot)
- Average maximum temperature by month and city (heatmap)
- Monthly average maximum temperature in Perth (line chart)
- Average daily rainfall by city (bar chart)
- Afternoon temperature vs humidity (scatter plot, 5,000-row sample)

---

## Testing

Three `unittest` test classes run inside the notebook, and all **13 tests pass**:

| Test class | What it checks |
|------------|----------------|
| `TestTemperatureRange` (3 tests) | The temperature-range function handles normal, equal and negative temperatures |
| `TestDatasetIntegrity` (5 tests) | Cleaned data is non-empty, with no missing values, duplicate rows or duplicate `Date`/`Location` keys, and all required columns present |
| `TestDataMerging` (5 tests) | Merged data has the correct row count, unique keys, no missing values, all required columns, and values identical to the source |

---

## Repository structure

```
├── Weather Australia.ipynb    # Full analysis notebook
├── weatherAUS.csv             # Raw dataset (download from Kaggle)
├── weatherAUS_cleaned.csv     # Generated: cleaned dataset
├── weatherAUS_merged.csv      # Generated: merged dataset
└── README.md
```

---


## Limitations

- Imputed values are statistical estimates, not recorded observations. About 8% of rows come from stations with no pressure readings, so their pressure values are an overall median.
- Dropping `Sunshine`, `Evaporation` and cloud cover loses potentially useful information.
- Removing rows with missing rainfall could add bias if that missingness is linked to particular stations or weather conditions.
- Some stations have gaps in their records, which can affect comparisons between cities.
- All relationships found are associations, not cause and effect.
