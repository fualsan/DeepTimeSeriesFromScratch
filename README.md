# DeepTimeSeriesFromScratch
Deep neural network models implemented from scratch in PyTorch for time series forecasting.

## Table of Contents
- [Introduction](#introduction)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
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
This project uses the Air Passengers dataset, a classic time series dataset that records the monthly totals of international airline passengers from 1949 to 1960. The dataset is included in the repository (*airline-passengers.csv*).

The dataset contains the following columns:
- `Month`: The month of the observation.
- `Passengers`: The number of passengers carried by the airline.

## License
This project is licensed under the MIT License - please see the [LICENSE](LICENSE) file for details.
