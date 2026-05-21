# Evaluating hybrid deep learning architectures for day-ahead electricity price forecasting in the german spot market

This repository is the official implementation of *Evaluating hybrid deep learning architectures for day-ahead electricity price forecasting in the german spot market*.


## Abstract

Forecasting day-ahead electricity prices is a longstanding focus of econometric research and energy market participants. Recent studies indicate that contemporary approaches, including hybrid and deep learning techniques, are increasingly used. However, the complexity and resource requirements of advanced techniques make custom hybrid learning architectures challenging. For example, inconsistent adherence to temporal constraints is a recurring limitation. Additionally, inappropriate model interpretations and comparisons can be misleading, often due to the black-box nature of neural networks. To address these limitations, a comprehensive selection of feature engineering and machine learning techniques was used to forecast hourly day-ahead electricity prices on the German spot market. This selection included architectures incorporating Variational Mode Decomposition, convolutional and recurrent neural networks, multi-head attention, and transformers, all under strict temporal training and forecasting constraints. The supervised models and signal decomposition techniques were tuned using Bayesian optimization and frequency separation analyses, respectively. To compare real-time performance, these frameworks were tested on a year of hourly data. The results revealed that novel hybrid deep learning architectures demonstrate superior performance during regime changes and volatility spikes inherent to energy markets compared with traditional statistical methods.


## Requirements

To install requirements:

```
pip install -r requirements.txt
```

It is recommended that a separate virtual environment is used for the purpose of running scripts from this repository. It is recommended to use [uv](https://docs.astral.sh/uv/) for virtual environment management.

For hyperparameter tuning, model training, and model predictions, scripts are intended to be run on an HPC computer cluster with GPU availability. Thus, an up-to-date remote directory must also be kept on an HPC computer cluster. Depending on the specific setup, some of the `bash` scripts may need slight modifications to file paths.


## Data Processing

Text


## Training

Text


## Evaluation

Text


## Results

Text