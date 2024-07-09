import torch
#import torch.nn as nn
#import torch.nn.functional as F

from models.transformer import GPTTimeSeries

import numpy as np
import pandas as pd
from pprint import pprint

from sklearn.preprocessing import StandardScaler

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'USING DEVICE: {device}')


# uvicorn runs this
app = FastAPI()

# Load Saved Checkpoint
checkpoint = torch.load('./saved_models/GPTTimeSeries_Autoregressive.pt')
print('Checkpoint is loaded with keys:')
pprint(list(checkpoint.keys()))
print()

# Load Saved (Pre-trained) Model
hyperparameters = checkpoint['hyperparameters']
print('Model hyperparameters is loaded with:')

for k, v in hyperparameters.items():
    print(f'{k:<25} {v}')
print()


model = GPTTimeSeries(
    input_features_size=hyperparameters['input_features_size'],
    date_input_features_size=hyperparameters['date_input_features_size'],
    date_features_dim=hyperparameters['date_features_dim'],
    features_dim=hyperparameters['hidden_features_size'],
    output_features_size=hyperparameters['output_features_size'],
    num_heads=hyperparameters['num_heads'],
    ff_dim=hyperparameters['ff_dim'],
    num_decoder_layers=hyperparameters['num_decoder_layers'],
    emb_dropout_prob=hyperparameters['emb_dropout_prob'],
    attn_dropout_prob=hyperparameters['attn_dropout_prob'],
    ff_dropout_prob=hyperparameters['ff_dropout_prob'],
    attn_use_bias=hyperparameters['attn_use_bias'],
    ff_use_bias=hyperparameters['ff_use_bias'],
    output_features_bias=hyperparameters['output_features_bias'],
)


model.to(device)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()
print('Model is loaded!')


@torch.no_grad()
def generative_forecast(model, data, timestamps, num_steps, lag_window_size, use_amp, device):
    model.eval()
    
    predictions = []
    time_indexes = []
    
    # covnert to tensor
    # data.shape: (lags, features)
    lags = torch.tensor(data[-lag_window_size:, :], dtype=torch.float32, device=device)
    
    # artificially add batch dimension
    # (we are not using the dataloader here!)
    # data.shape: (1, lags, features)
    lags = lags.unsqueeze(0)

    # Datetime indexes 
    #timestamps = df_full.index 
    # Delta time: calculate the time difference between two samples 
    delta_time = timestamps[1] - timestamps[0]
    # Get last timestamp
    current_timestamp = timestamps[-1]

    def generate_date_tensor(_timestamp, _lags, _device):
        _timestamp = _timestamp[-lag_window_size:]
        return torch.tensor([_timestamp.month, _timestamp.day, _timestamp.hour], dtype=torch.float32, device=_device).permute(1, 0)
    
    # single step
    for idx in range(num_steps):

        # get the last lag steps
        lags = lags[:, -lag_window_size:, :]
        #print(lags)

        # date
        date = generate_date_tensor(timestamps, lag_window_size, device).unsqueeze(0)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            forecast_pred = model(lags, date)
        
        # (batch, forecast, output_features_size)-> (1, window_size-1, output_features_size)
        # TAKE THE LAST PREDICTION STEP AS FORECAST!
        predictions.append(forecast_pred[0][-1].cpu().numpy())

        # update current timestamp
        current_timestamp = current_timestamp + delta_time
        time_indexes.append(current_timestamp)
        
        # append last forecast to the end
        # TAKE THE LAST PREDICTION STEP AS FORECAST!
        lags = torch.cat((lags, forecast_pred[:, -1:, :].detach()), dim=1)
        
        # next timestamp
        timestamps = timestamps + delta_time

    return predictions, time_indexes


# TEMPLATE FOR REQUEST
class GenerativeForecastRequest(BaseModel):
    request_asctime: str
    df_request: str
    num_steps: int


@app.post('/predict')
async def predict(request: GenerativeForecastRequest):

    print('Request asctime:', request.request_asctime)

    df_request = pd.read_json(request.df_request)

    print(df_request.head())

    # PREPROCESSING DATA
    # Standart Scaler
    feature_scaler = StandardScaler()
    df_request[df_request.columns] = feature_scaler.fit_transform(df_request[df_request.columns])
    
    
    pred_generative, time_indexes_generative = generative_forecast(
        model=model, 
        data=df_request.values,
        timestamps=df_request.index,
        num_steps=request.num_steps, 
        lag_window_size=hyperparameters['window_size'], 
        use_amp=hyperparameters['use_amp'], 
        device=device
    )

    pred_generative_array = np.array(pred_generative)

    generative_results_dict = {}
    
    # loop over features
    for feature_id, feature_key in enumerate(df_request.columns):
        generative_results_dict[feature_key] = pred_generative_array[:, feature_id]
            
    df_generative = pd.DataFrame(data=generative_results_dict, index=time_indexes_generative)

    # REVERSE THE PREPROCESSING FOR ORIGINAL RANGE
    df_generative[df_generative.columns] = feature_scaler.inverse_transform(df_generative[df_generative.columns])    
    
    json_response = {
        'response_asctime': time.asctime(),
        'df_response': df_generative.to_json()
    }

    return json_response


if __name__ == '__main__':
	# optionally run from terminal: uvicorn t2i_server:app --host 0.0.0.0 --port 8000 --reload
	# accept every connection (not only local connections)
    uvicorn.run(app, host='0.0.0.0', port=8000)