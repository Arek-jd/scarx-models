import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from date_functions import *
import time
from get_WMAE import *
import pywt
from scipy.optimize import basinhopping

def remove_mean(window_size,first_day_index,lprices,lloads):
    window_size_h = window_size*24
    first_hour_index = first_day_index*24
    qs = lprices[first_hour_index:first_hour_index+window_size_h]
    qs_mean=np.mean(qs)
    qs = np.reshape(qs,(-1,24)) #qs : (window_size,24)

    qs=qs-qs_mean

    zt = lloads[first_hour_index:first_hour_index+window_size_h+24]
    zt = np.reshape(zt,(-1,24)) #zt : (window_size+1,24)
    
    qsmin = np.min(qs, axis = 1, keepdims = True)
    return qs, qsmin, zt, qs_mean

def decomposition_HP(lprices, lloads, first_day_index, window_size,lambd):
    window_size_h = window_size*24
    first_hour_index=first_day_index*24
    qs = lprices[first_hour_index:first_hour_index+window_size_h]
    zt = lloads[first_hour_index:first_hour_index+window_size_h+24]

    #removing LTSC
    qs, Ts = sm.tsa.filters.hpfilter(qs, lambd)
    zt, _ = sm.tsa.filters.hpfilter(zt, lambd)
    
    qs = np.reshape(qs,(-1,24)) #qs: (window_size,24)
    zt = np.reshape(zt,(-1,24)) #zt: (window_size+1,24)
    
    qsmin = np.min(qs, axis = 1, keepdims = True)
    
    #Ts[-24:] used as T_hat for day+1
    return qs, qsmin, zt, Ts[-24:]

def decomposition_wavelet(lprices, lloads, first_day_index, window_size, level):
    window_size_h = window_size*24
    first_hour_index=first_day_index*24
    qs = lprices[first_hour_index:first_hour_index+window_size_h]
    zt = lloads[first_hour_index:first_hour_index+window_size_h+24]
    #removing LTSC
    mode='symmetric'
    wavelet='db24'
    coeffs = pywt.wavedec(qs, wavelet, level=14, mode=mode)
    Ts = pywt.waverec(coeffs[:15-level] + [None]*level, wavelet, mode)
    Ts = Ts[:len(qs)]
    qs = qs - Ts

    coeffs = pywt.wavedec(zt, wavelet, level=14, mode=mode)
    zt_LTSC = pywt.waverec(coeffs[:15-level] + [None]*level, wavelet, mode)
    zt_LTSC = zt_LTSC[:len(zt)]
    zt = zt -zt_LTSC
    
    qs = np.reshape(qs,(-1,24)) #qs: (window_size,24)
    zt = np.reshape(zt,(-1,24)) #zt: (window_size+1,24)
    
    qsmin = np.min(qs, axis = 1, keepdims = True)
    
    #Ts[-24:] used as T_hat for day+1
    return qs, qsmin, zt, Ts[-24:]

def get_calibartion_dataset(qs, qsmin, zt, window_size,h, D1, D2, D3):
    #qs: (window_size, 24)
    #qsmin: (window_size,1)
    #zt: (window_size+1,24)
    #D1, D2, D3: (window_size+1)
    X=np.ones((window_size-7,8))
    X[:,0]=qs[6:-1,h]
    X[:,1]=qs[5:-2,h]
    X[:,2]=qs[0:-7,h]
    X[:,3]=qsmin[6:-1,0]
    X[:,4]=zt[7:-1,h]
    X[:,5]=D1[7:-1]
    X[:,6]=D2[7:-1]
    X[:,7]=D3[7:-1]

    Y=qs[7:,h]
    
    Xr=np.array([qs[-1,h],qs[-2,h],qs[-7,h],qsmin[-1,0],zt[-1,h],D1[-1],D2[-1],D3[-1]])
    #estimation set: X, Y
    #prediction set: Xr
    return X, Y, Xr 

def get_dummies(dates):
    #changes dates (given as str 'YYYYmmdd') to dummies Mon: D1, Sat: D2, Sun: D3
    dates_asint = dates.astype(int)
    dates_asint = np.reshape(dates_asint,(-1,24)) 
    dayofweek=np.zeros((dates_asint.shape[0],1))
    for i in range(dates_asint.shape[0]):
        dayofweek[i,0] = get_dayofweek(dates_asint[i,0])
    D1 = (dayofweek == 0).astype(int) # Mon
    D2 = (dayofweek == 5).astype(int) # Sat
    D3 = (dayofweek == 6).astype(int) # Sun
    return D1, D2, D3

def get_dummies_inwindow(D1,D2,D3, window_size, first_day_index):
    D1 = D1[first_day_index:first_day_index+window_size+1,0] 
    D2 = D2[first_day_index:first_day_index+window_size+1,0]
    D3 = D3[first_day_index:first_day_index+window_size+1,0]
    #D1, D2, D3: (window_size +1)
    return D1, D2, D3

def get_estimated_parameters_LSM(X,Y):
    XT=X.T
    return np.linalg.inv(np.dot(XT,X)).dot(XT).dot(Y)

def make_prediction(X,params,T):
    return np.exp(np.dot(X,params)+T)

def loss_function(params, X, Y):
    Yhat = np.dot(X,params)
    return np.mean(np.abs(Y - Yhat))

def run_model(dataset, window_size, first_eday, last_eday, param):
    #param: lambd for HP, level for wavelet
    raw_data = np.genfromtxt(f'DATA/{dataset}.txt')
    qs_real = np.reshape(raw_data[:,2],(-1,24))
    num_days=qs_real.shape[0]

    qs_predictions = np.zeros(qs_real.shape)
    D1_all,D2_all,D3_all=get_dummies(raw_data[:,0])
    lprices = np.log(raw_data[:,2])
    lloads = np.log(raw_data[:,3])

    for day in range(window_size+1,num_days+1):
        first_day_index=day-(window_size+1)
        D1, D2, D3 = get_dummies_inwindow(D1_all,D2_all,D3_all, window_size, first_day_index)
        #qs, qsmin, zt, Ts_hat=decomposition_wavelet(lprices, lloads, first_day_index, window_size,param)
        #qs, qsmin, zt, Ts_hat=decomposition_HP(lprices, lloads, first_day_index, window_size,param)
        qs, qsmin, zt, Ts_hat=remove_mean(window_size,first_day_index,lprices,lloads)
        for hour in range(24):
            X, Y, Xr = get_calibartion_dataset(qs, qsmin, zt, window_size,hour,D1,D2,D3)
            ### LSE regression ###
            params=get_estimated_parameters_LSM(X,Y) 
            ### LAD regression ###
            #res = sm.QuantReg(Y,X).fit()
            #params=res.params
            ### or
            #params=np.random.randn(1,8)
            #nms=basinhopping(loss_function,params, minimizer_kwargs={'args':(X, Y)},niter=1)
            #params=nms.x
            

            c_prediction=make_prediction(Xr,params,Ts_hat) #if wavelet or HP filters are used, change Ts_hat -> Ts_hat[hour]
            qs_predictions[first_day_index+window_size,hour]=c_prediction
            print(f'(day,hour):\t({day},{hour}):\t{c_prediction}')

    date = raw_data[0,0].astype(int)
    first_day = get_datetime(date)
    WMAE, ave_num = get_WMAE(qs_real, qs_predictions, first_day, first_eday, last_eday)
    return qs_real, qs_predictions, WMAE

dataset = 'NPdata_2013-2016'#'GEFCOM_hourly'#
window_size = 360
first_eday = datetime(2013, 12, 27)#datetime(2011, 12, 27)#
last_eday = datetime(2015, 12, 24)#datetime(2013, 12, 16)#

qs_real, qs_predictions, _ = run_model(dataset, window_size, first_eday, last_eday, None)

plt.plot(qs_real[window_size:,:].flatten())
plt.plot(qs_predictions[window_size:,:].flatten())
plt.show()


