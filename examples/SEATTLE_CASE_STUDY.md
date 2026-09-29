# Seattle daily-temperature case study

This is a reproducible demonstration on real historical observations, not a claim of production forecast performance.

## Source and provenance

- Dataset: `seattle-weather.csv` from [Vega Datasets](https://github.com/vega/vega-datasets), pinned to commit `ede9366badecc625cd6bcea5c4aaa055870c6ca6`.
- Vega's [data package metadata](https://github.com/vega/vega-datasets/blob/main/datapackage.json) describes this as NOAA-derived daily Seattle weather data for instructional use, with an original-source link to NOAA and a U.S. government dataset license label. Verify any intended redistribution independently. This repository downloads the source on demand; it does not bundle the raw file.
- The pinned raw CSV is 48,219 bytes; expected SHA-256: `0845078a290b48e3149ab8639966824110a251db4e06fc144c06ebb534af23be`. The script rejects other bytes.
- Only `date` and `temp_max` are used. No weather category, precipitation, wind or other columns are used.

## Forecasting scenario

At 00:00 UTC on each day, predict that day's maximum temperature from the previous observed day's maximum. The source CSV does **not** contain publication timestamps. The script assumes an observation becomes available at 00:00 UTC on the following day. That lag is an explicit hypothetical policy and is not verified NOAA release timing. A 24-hour freshness limit rejects older facts. Labels are considered complete at the next day's 00:00 UTC, and the walk-forward split purges labels overlapping a test window.

The five expanding folds each hold out 90 days after an initial 365 distinct training dates. The model is a train-only-imputed, standardized ridge regression using one lagged temperature feature. It is compared with the mean of training targets, which is deliberately weak; a stronger seasonal or persistence baseline remains future work. Seven-day circular blocks are used for a descriptive paired bootstrap interval. No hyperparameter tuning is performed on held-out folds in this script.

## Run

From the repository root:

```sh
python examples/run_seattle_weather.py
```

To run offline, download the pinned CSV separately and pass `--source /path/to/seattle-weather.csv`. Use `--output-dir /new/directory` to preserve generated inputs, split plan, joined features and report for inspection. The directory must not already exist.

On 2026-09-29 with Python 3.12 on Windows, the verified pinned file produced 1,460 daily samples and 450 held-out predictions. The ridge RMSE was 2.692°C versus 7.765°C for the train-mean baseline, with paired squared-loss delta interval approximately [-64.96, -42.02] °C². These numbers describe this one chosen scenario and assumed availability policy only. They do not establish significance after model selection, calibrated uncertainty, or operational forecasting quality.
