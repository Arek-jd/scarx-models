import numpy as np

def get_WMAE(real, forecasts, first_day, first_eday, last_eday):
    print('\nFirst day for WMAE:\t'+ first_eday.strftime('%d.%m.%Y') + '\t(day of the week: ' + str(first_eday.weekday()) + ')' +
      '\nLast day for WMAE:\t'+ last_eday.strftime('%d.%m.%Y') + '\t(day of the week: ' + str(last_eday.weekday()) + ')\n')
    #returns average WMAE in %, number of weeks
    first_index = (first_eday - first_day).days
    last_index = (last_eday - first_day).days + 1

    real = real[first_index:last_index,:]
    forecasts = forecasts[first_index:last_index,:]

    WMAE = np.mean(np.abs(real - forecasts), axis = 1, keepdims = True)
    WMAE = np.mean(np.reshape(WMAE, (-1,7)), axis = 1, keepdims = True)

    Pdash = np.mean(real, axis = 1, keepdims = True)
    Pdash = np.mean(np.reshape(Pdash, (-1,7)), axis = 1, keepdims = True) 
    
    WMAE = np.mean(WMAE/Pdash)*100
    ave_num = Pdash.shape[0]
    print(f'WMAE averaged over {ave_num} weeks:\t{WMAE:.3f}%')
    return WMAE, ave_num
