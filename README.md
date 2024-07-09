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

## Dataset
### Air Passengers Dataset
A classic time series dataset that records the monthly totals of international airline passengers from 1949 to 1960. The dataset is included in the repository (*airline-passengers.csv*).

The dataset contains the following columns:
- `Month`: The month of the observation.
- `Passengers`: The number of passengers carried by the airline.

### Panama Electricty Load Forecasting
A collection of data related to electricity consumption in Panama, specifically aimed at forecasting future electricity loads. This dataset typically includes historical records of electricity consumption, possibly broken down by region or time intervals (like hourly, daily, or monthly). It may also incorporate additional variables such as weather parameters (temperature, humidity, precipitation) and special days (public holidays, weekends) that could influence electricity demand.

This is a multivariate dataset with many columns. For more info: [Kaggle dataset page](https://www.kaggle.com/datasets/pateljay731/panama-electricity-load-forecasting/)

### API
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
