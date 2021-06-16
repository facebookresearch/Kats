<div align="center">
<img src="kats_logo.svg" width="30%"/>
</div>

<div align="center">
  <a href="https://github.com/facebookresearch/Kats/actions">
  <img alt="Github Actions" src="https://github.com/facebookresearch/Kats/actions/workflows/build_and_test.yml/badge.svg"/>
  </a>
  <a href="https://github.com/facebookresearch/Kats/blob/master/CONTRIBUTING.md">
  <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"/>
  </a>
</div>

## Description

Kats is a toolkit to analyze time series data, a lightweight, easy-to-use, and generalizable framework to perform time series analysis. Time series analysis is an essential component of Data Science and Engineering work at industry, from understanding the key statistics and characteristics, detecting regressions and anomalies, to forecasting future trends. Kats aims to provide the one-stop shop for time series analysis, including detection, forecasting, feature extraction/embedding, multivariate analysis, etc. Kats is released by Facebook's Infrastructure Strategy team. It is available for download on [PyPI](https://pypi.python.org/pypi/kats/).

## Important links

- Homepage: https://facebookresearch.github.io/Kats/
- Source code repository: https://github.com/facebookresearch/kats
- Contributing: https://github.com/facebookresearch/Kats/blob/master/CONTRIBUTING.md
- Tutorials: https://github.com/facebookresearch/Kats/tree/master/tutorials
- Kats Python package: TODO
- Release blogpost: TODO

## Installation in Python [TODO]

Kats is on PyPI, so you can use `pip` to install it.

```bash
pip install -U kats
```

To install from a copy of this source code, in the Kats directory run

```bash
pip install -e .
```

## Examples

Here are a few sample snippets from a subset of Kats offerings:

### Forecasting

Using `Prophet` model to forecast the `air_passengers` data set.

```python
from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams

# take `air_passengers` data as an example
air_passengers_df = pd.read_csv("../kats/data/air_passengers.csv")
air_passengers_ts = TimeSeriesData(air_passengers_df)

# create a model param instance
params = ProphetParams(seasonality_mode='multiplicative') # additive mode gives worse results

# create a prophet model instance
m = ProphetModel(air_passengers_ts, params)

# fit model simply by calling m.fit()
m.fit()

# make prediction for next 30 month
fcst = m.predict(steps=30, freq="MS")
```

### Detection

Using `CUSUM` detection algorithm on simulated data set.

```python
# import packages
from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDetector

# simulate time series with increase
np.random.seed(10)
df_increase = pd.DataFrame(
    {
        'time': pd.date_range('2019-01-01', '2019-03-01'),
        'increase':np.concatenate([np.random.normal(1,0.2,30), np.random.normal(2,0.2,30)]),
    }
)

# convert to TimeSeriesData object
timeseries = TimeSeriesData(df_increase)

# run detector and find change points
change_points = CUSUMDetector(timeseries).detector()
```

### TSFeatures

We can extract meaningful features from the given time series data

```python
# Initiate feature extraction class
from kats.tsfeatures.tsfeatures import TsFeatures

features = TsFeatures().transform(air_passengers_ts)
```

## Changelog

### Version 0.1 (TODO)

- Initial release

## License

Kats is licensed under the [MIT license](TODO).
