<div align="center">
<img src="kats_logo.svg" width="40%"/>
</div>

<div align="center">
  <a href="https://github.com/facebookresearch/Kats/actions">
  <img alt="Github Actions" src="https://github.com/facebookresearch/Kats/actions/workflows/build_and_test.yml/badge.svg"/>
  </a>
  <a href="https://pypi.python.org/pypi/kats">
  <img alt="PyPI Version" src="https://img.shields.io/pypi/v/kats.svg"/>
  </a>
  <a href="https://github.com/facebookresearch/Kats/blob/master/CONTRIBUTING.md">
  <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"/>
  </a>
</div>

## Description

Kats is a toolkit to analyze time series data, a lightweight, easy-to-use, and generalizable framework to perform time series analysis. Time series analysis is an essential component of Data Science and Engineering work at industry, from understanding the key statistics and characteristics, detecting regressions and anomalies, to forecasting future trends. Kats aims to provide the one-stop shop for time series analysis, including detection, forecasting, feature extraction/embedding, multivariate analysis, etc.

Kats is released by Facebook's *Infrastructure Data Science* team. It is available for download on [PyPI](https://pypi.python.org/pypi/kats/).

## Important links

- Homepage: https://facebookresearch.github.io/Kats/
- Kats Python package: https://pypi.org/project/kats/
- Facebook Engineering Blog Post: https://engineering.fb.com/2021/06/21/open-source/kats/
- Source code repository: https://github.com/facebookresearch/kats
- Contributing: https://github.com/facebookresearch/Kats/blob/master/CONTRIBUTING.md
- Tutorials: https://github.com/facebookresearch/Kats/tree/master/tutorials

## Installation in Python

Kats is on PyPI, so you can use `pip` to install it.

```bash
pip install --upgrade pip
pip install kats
```

If you need only a small subset of Kats, you can install a minimal version of Kats with
```bash
MINIMAL_KATS=1 pip install kats
```
which omits many dependencies (everything in `test_requirements.txt`).
However, this will disable many functionalities and cause `import kats` to log
warnings. See `setup.py` for full details and options.

## Examples

Here are a few sample snippets from a subset of Kats offerings:

### Forecasting

Using `Prophet` model to forecast the `air_passengers` data set.

```python
import pandas as pd

from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams

# take `air_passengers` data as an example
air_passengers_df = pd.read_csv(
    "../kats/data/air_passengers.csv",
    header=0,
    names=["time", "passengers"],
)

# convert to TimeSeriesData object
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
import numpy as np
import pandas as pd

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
import pandas as pd
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures

# take `air_passengers` data as an example
air_passengers_df = pd.read_csv(
    "../kats/data/air_passengers.csv",
    header=0,
    names=["time", "passengers"],
)

# convert to TimeSeriesData object
air_passengers_ts = TimeSeriesData(air_passengers_df)

# calculate the TsFeatures
features = TsFeatures().transform(air_passengers_ts)
```

## Citing Kats

If you use Kats in your work or research, please use the following BibTeX entry.

```
@software{Jiang_KATS_2022,
author = {Jiang, Xiaodong and Srivastava, Sudeep and Chatterjee, Sourav and Yu, Yang and Handler, Jeffrey and Zhang, Peiyi and Bopardikar, Rohan and Li, Dawei and Lin, Yanjun and Thakore, Uttam and Brundage, Michael and Holt, Ginger and Komurlu, Caner and Nagalla, Rakshita and Wang, Zhichao and Sun, Hechao and Gao, Peng and Cheung, Wei and Gao, Jun and Wang, Qi and Guerard, Marius and Kazemi, Morteza and Chen, Yulin and Zhou, Chong and Lee, Sean and Laptev, Nikolay and Levendovszky, Tihamér and Taylor, Jake and Qian, Huijun and Zhang, Jian and Shoydokova, Aida and Singh, Trisha and Zhu, Chengjun and Baz, Zeynep and Bergmeir, Christoph and Yu, Di and Koylan, Ahmet and Jiang, Kun and Temiyasathit, Ploy and Yurtbay, Emre},
license = {MIT License},
month = {3},
title = {{Kats}},
url = {https://github.com/facebookresearch/Kats},
version = {0.2.0},
year = {2022}
}
```

## Changelog

### Version 0.2.0
* Forecasting
    * Added global model, a neural network forecasting model
    * Added [global model tutorial](https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_205_globalmodel.ipynb)
    * Consolidated backtesting APIs and some minor bug fixes
* Detection
    * Added model optimizer for anomaly/ changepoint detection
    * Added evaluators for anomaly/changepoint detection
    * Improved simulators, to build synthetic data and inject anomalies
    * Added new detectors: ProphetTrendDetector, Dynamic Time Warping based detectors
    * Support for meta-learning, to recommend anomaly detection algorithms and parameters for your dataset
    * Standardized API for some of our legacy detectors: OutlierDetector, MKDetector
    * Support for Seasonality Removal in StatSigDetector
* TsFeatures
    * Added time-based features
* Others
    * Bug fixes, code coverage improvement, etc.

### Version 0.1.0

* Initial release

## Contributors
Kats is currentely maintaned by community with the main contributions and leading from [Nickolai Kniazev](https://www.linkedin.com/in/nickknyazev/) and Peter Shaffery

Kats is a project with several skillful researchers and engineers contributing to it.
Kats was started and built by [Xiaodong Jiang](https://www.linkedin.com/in/xdjiang/) with major contributions coming
from many talented individuals in various forms and means. A non-exhaustive but growing list needs to mention: [Sudeep Srivastava](https://www.linkedin.com/in/sudeep-srivastava-2129484/), [Sourav Chatterjee](https://www.linkedin.com/in/souravc83/), [Jeff Handler](https://www.linkedin.com/in/jeffhandl/), [Rohan Bopardikar](https://www.linkedin.com/in/rohan-bopardikar-30a99638), [Dawei Li](https://www.linkedin.com/in/lidawei/), [Yanjun Lin](https://www.linkedin.com/in/yanjun-lin/), [Yang Yu](https://www.linkedin.com/in/yangyu2720/), [Michael Brundage](https://www.linkedin.com/in/michaelb), [Caner Komurlu](https://www.linkedin.com/in/ckomurlu/), [Rakshita Nagalla](https://www.linkedin.com/in/rakshita-nagalla/), [Zhichao Wang](https://www.linkedin.com/in/zhichaowang/), [Hechao Sun](https://www.linkedin.com/in/hechao-sun-83b9ba4b/), [Peng Gao](https://www.linkedin.com/in/peng-gao-9137a24b/), [Wei Cheung](https://www.linkedin.com/in/weizhicheung/), [Jun Gao](https://www.linkedin.com/in/jun-gao-71352b64/), [Qi Wang](https://www.linkedin.com/in/qi-wang-9231a783/), [Morteza Kazemi](https://www.linkedin.com/in/morteza-kazemi-pmp-csm/), [Tihamér Levendovszky](https://www.linkedin.com/in/tiham%C3%A9r-levendovszky-29639b5/), [Jian Zhang](https://www.linkedin.com/in/jian-zhang-73718917/), [Ahmet Koylan](https://www.linkedin.com/in/ahmetburhan/), [Kun Jiang](https://www.linkedin.com/in/kunqiang-jiang-ph-d-0988aa1b/), [Aida Shoydokova](https://www.linkedin.com/in/ashoydok/), [Ploy Temiyasathit](https://www.linkedin.com/in/nutcha-temiyasathit/), Sean Lee, [Nikolay Pavlovich Laptev](http://www.nikolaylaptev.com/), [Peiyi Zhang](https://www.linkedin.com/in/pyzhang/), [Emre Yurtbay](https://www.linkedin.com/in/emre-yurtbay-27516313a/), [Daniel Dequech](https://www.linkedin.com/in/daniel-dequech/), [Rui Yan](https://www.linkedin.com/in/rui-yan/), [William Luo](https://www.linkedin.com/in/wqcluo/), [Marius Guerard](https://www.linkedin.com/in/mariusguerard/), [Pietari Pulkkinen](https://www.linkedin.com/in/pietaripulkkinen/), [Uttam Thakore](https://www.linkedin.com/in/uttam-thakore/), [Trisha Singh](https://www.linkedin.com/in/trishasingh2696/), [Huijun Qian](https://www.linkedin.com/in/huijun-qian-0845a958/), [Chengjun Zhu](https://www.linkedin.com/in/chengjunzhu/), [Di Yu](https://www.linkedin.com/in/diyu05/), [Zeynep Erkin Baz](https://www.linkedin.com/in/zeynep-erkin-baz-51198a21), and [Christoph Bergmeir](https://au.linkedin.com/in/christoph-bergmeir-00711a14/).

## License

Kats is licensed under the [MIT license](LICENSE).
