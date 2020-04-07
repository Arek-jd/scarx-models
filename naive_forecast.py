import numpy as np

def naive_forecast(prices, num_days, first_dayofweek):
    forecasts=np.zeros(prices.shape)
    current_dayofweek=first_dayofweek
    for day in range(7, num_days):
        if current_dayofweek in {0, 5, 6}:
            lag = 7 
        else:
            lag = 1 
        forecasts[day,:]=prices[day-lag,:]
        current_dayofweek = (current_dayofweek + 1) % 7
    return forecasts
