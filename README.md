# DeepTimeSeriesFromScratch
Deep neural network models implemented from scratch in PyTorch for time series forecasting.

## Table of Contents
- [Introduction](#introduction)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
- [API](#api)
- [License](#license)

## Introduction
This repository contains implementations of various deep learning models for time series forecasting, all built from scratch using PyTorch. The models included are:
- Vanilla Recurrent Neural Network (RNN)
- Gated Recurrent Unit (GRU)
- Long Short-Term Memory (LSTM)
- Transformer

These implementations are designed to help understand the inner workings of these models and to provide a solid foundation for building more complex time series forecasting solutions.

## Models Implemented
- **Vanilla RNN**: A basic recurrent neural network that captures temporal dependencies in sequential data.
- **LSTM**: Advanced RNN variant designed to remember/forget past temporal dependencies.
- **GRU**: Simplified version of LSTM that offers similar peformance with fewer gates. 
- **Transformer**: A state-of-the-art model using attention mechanisms. Decoder based transformer is implemented for generative applications.
- **MoE Transformer**: Mixture of Experts version of Transformer. Experts are weighless and identical. 

## Dataset
### Air Passengers Dataset
A classic time series dataset that records the monthly totals of international airline passengers from 1949 to 1960. The dataset is included in the repository (*airline-passengers.csv*).

The dataset contains the following columns:
- `Month`: The month of the observation.
- `Passengers`: The number of passengers carried by the airline.

### Panama Electricty Load Forecasting
A collection of data related to electricity consumption in Panama, specifically aimed at forecasting future electricity loads. This dataset typically includes historical records of electricity consumption, possibly broken down by region or time intervals (like hourly, daily, or monthly). It may also incorporate additional variables such as weather parameters (temperature, humidity, precipitation) and special days (public holidays, weekends) that could influence electricity demand.

This is a multivariate dataset with many columns. For more info: [Kaggle dataset page](https://www.kaggle.com/datasets/pateljay731/panama-electricity-load-forecasting/)

### Numenta Anomoly Benchmark (NAB)
*Original repository:* [NAB GitHub repository](https://github.com/numenta/NAB)

The NAB is a comprehensive benchmark suite for evaluating anomaly detection algorithms in time series data. It provides a set of datasets that cover various domains and includes ground-truth annotations of anomalies. The goal of NAB is to facilitate the comparison of different anomaly detection methods and to advance the field by providing standardized evaluation metrics and datasets.

The NAB dataset collection includes:
- **Synthetic Datasets**: These datasets are artificially generated and designed to test various aspects of anomaly detection algorithms.
- **Real-World Datasets**: These datasets come from real-world applications and include anomalies that have been manually annotated.

The evaluation metrics used in NAB include:
- **True Positive Rate (TPR)**: The proportion of actual anomalies that are correctly identified.
- **False Positive Rate (FPR)**: The proportion of normal data points incorrectly classified as anomalies.
- **Score**: A combined metric that accounts for both precision and recall.

#### Cloning the NAB Repository
To clone the NAB repository, use the following command:

```bash
$ git clone https://github.com/numenta/NAB.git
```

## API
The models can be deployed with as web service using RESTful APIs. FastAPI is used to implement API gateway and endpoints. Server code is implemented in **generative_forecast_server.py** file. You can utilize our RESTful API by sending a POST request to the /predict endpoint. Simply prepare your input data in a JSON format and pass it as the request body. An example client code is implemented in **generative_forecast_client.py** file. 

#### Currently supported endpoints:
| Endpoint URL | HTTP Method | Description |
| --- | --- | --- |
| `/predict` | `POST` | Loads a pretrained model and makes predictions for a given number of steps |

#### The request JSON format is given below:
| Property | Type | Description |
| --- | --- | --- |
| `request_asctime` | `str` | Asctime for the request |
| `df_request` | `str` | Dataframe of the request data where index is timestamps (datetime object) |
| `num_steps` | `int` | Number of steps in the forecast |

## License
This project is licensed under the MIT License - please see the [LICENSE](LICENSE) file for details.
