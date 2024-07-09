import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests
import time
from pprint import pprint
import json
from io import StringIO


df_full = pd.read_csv('data/panama_electricity_load_forecasting/train.csv')
df_full['datetime'] = pd.to_datetime(df_full['datetime'], dayfirst=True)
df_full.set_index('datetime', inplace=True)

print('DATASET HEAD:')
print(df_full.head())
print()

print('DATASET TAIL:')
print(df_full.tail())
print()

# should be min of original model
REQUEST_WINDOW_SIZE = 200 * 2 # * 2 is added for convenience

data_request = {
    'request_asctime': time.asctime(),
    'df_request': df_full[-REQUEST_WINDOW_SIZE:].to_json(), # Send last REQUEST_WINDOW_SIZE timestamps
    'num_steps': 100, # number of forecast steps predicted by model
}


json_request = json.dumps(data_request).encode('utf-8')

# CHANGE 'localhost' TO TARGET URL!
response = requests.post('http://localhost:8000/predict', data=json_request, headers={'Content-Type': 'application/json'})

json_response = json.loads(response.text)

#print(json_response.keys())
#print('RECEIVED DATA:')
#print(json_response)

print(f"RESPONSE RECEIVED AT: {json_response['response_asctime']}")

df_response = pd.read_json(StringIO(json_response['df_response']))
print(df_response.head())

selected_feature = 'nat_demand'
df_response[selected_feature].plot()
plt.show()