from datetime import datetime

def get_datetime(date_int):
    date_str=str(date_int)
    return datetime.strptime(date_str,'%Y%m%d')

def get_datetime_str(date_int):
    #returns date as a string
    return get_datetime(date_int).strftime('%d.%m.%Y')

def get_dayofweek(date_int):
    #returns: 0 - Mon, 1 - Tue, ..., 6 - Sun
    return get_datetime(date_int).weekday()