## Requirements

import time
import requests
from functools import reduce

import pandas as pd


def collect_weather_data(first_datetime:str, last_datetime:str, verbose:bool=False) -> pd.DataFrame:

    '''
    start   :   Start date in ISO 8601 format (e.g., '2024-01-01T00:00:00Z').
    end     :   End date in ISO 8601 format (e.g., '2024-12-31T23:00:00Z').
    verbose :   Whether to print progress messages.
    '''

    start = time.time()
    coordinates = [
        (52.5200, 13.4050),  # Berlin
        (53.5488, 9.9872),   # Hamburg
        (48.1351, 11.5820),  # Munich
        (50.9375, 6.9603)    # Cologne
    ]
    weather_variables = [
        'precipitation',
        'cloud_cover',
        'sunshine',
        'temperature',
        'relative_humidity'
    ]
    url = 'https://api.brightsky.dev/weather'
    parameters = {
        'date':first_datetime,
        'last_date':last_datetime
    }
    concat_list = []
    for lat, lon in coordinates:
        parameters['lat'] = lat # type: ignore
        parameters['lon'] = lon # type: ignore
        data = requests.get(url, parameters).json()
        temp = pd.DataFrame(data['weather'])[['timestamp'] + weather_variables]
        concat_list.append(temp)
    weather = pd.concat(concat_list)
    weather['datetime'] = pd.to_datetime(weather['timestamp'], format='ISO8601', utc=True) + pd.Timedelta(hours=-1)
    weather = weather\
        .drop(columns='timestamp')\
        .groupby('datetime', as_index=False)\
        .agg('mean')
    if verbose:
        print(f'Collected weather data in {time.time()-start:.2f} seconds.\n')

    return weather


def collect_price_data(first_datetime:str, last_datetime:str, verbose:bool=False) -> pd.DataFrame:

    '''
    start   :   Start date in ISO 8601 format (e.g., '2024-01-01T00:00:00Z').
    end     :   End date in ISO 8601 format (e.g., '2024-12-31T23:00:00Z').
    verbose :   Whether to print progress messages.
    '''

    start = time.time()
    url = 'https://api.energy-charts.info/price'
    parameters = {
        'start':first_datetime,
        'end':last_datetime
    }
    data = requests.get(url, params=parameters).json()
    data.pop('license_info'); data.pop('unit'); data.pop('deprecated')
    price = pd.DataFrame(data)
    price['datetime'] = pd.to_datetime(price['unix_seconds'], unit='s', utc=True)
    price = price.drop(columns='unix_seconds')
    if verbose:
        print(f'Collected day-ahead price data in {time.time()-start:.2f} seconds.\n')
    
    return price


def collect_production_data(first_datetime:str, last_datetime:str, verbose:bool=False) -> pd.DataFrame:
    
    '''
    start   :   Start date in ISO 8601 format (e.g., '2024-01-01T00:00:00Z').
    end     :   End date in ISO 8601 format (e.g., '2024-12-31T23:00:00Z').
    verbose :   Whether to print progress messages.
    '''

    start = time.time()
    url = 'https://api.energy-charts.info/public_power'
    parameters = {
        'start':first_datetime,
        'end':last_datetime
    }
    data = requests.get(url, params=parameters).json()
    public_power = pd.DataFrame()
    public_power['datetime'] = pd.to_datetime(data['unix_seconds'], unit='s', utc=True) + pd.Timedelta(hours=-1)
    for production_type_data in data['production_types']:
        col = production_type_data['name'].lower().replace(' ', '_').replace('-', '_')
        public_power[col] = production_type_data['data']
    public_power = public_power[
        public_power['datetime'].dt.minute == 0
    ].drop(columns=['hydro_pumped_storage_consumption', 'cross_border_electricity_trading'])
    if verbose:
        print(f'Collected production data in {time.time()-start:.2f} seconds.\n')
    
    return public_power


def collect_cbet_data(first_datetime:str, last_datetime:str, verbose:bool=False) -> pd.DataFrame:

    '''
    start   :   Start date in ISO 8601 format (e.g., '2024-01-01T00:00:00Z').
    end     :   End date in ISO 8601 format (e.g., '2024-12-31T23:00:00Z').
    verbose :   Whether to print progress messages.
    '''

    start = time.time()
    url = 'https://api.energy-charts.info/cbet'
    parameters = {
        'start':first_datetime,
        'end':last_datetime
    }
    data = requests.get(url, parameters).json()
    cbet = pd.DataFrame()
    cbet['datetime'] = pd.to_datetime(data['unix_seconds'], unit='s', utc=True) + pd.Timedelta(hours=-1)
    for country_data in data['countries']:
        col = country_data['name'].lower().replace(' ', '_').replace('-', '_') + '_cbet'
        cbet[col] = country_data['data']
    cbet = cbet[
        cbet['datetime'].dt.minute == 0
    ]
    if verbose:
        print(f'Collected cross-border electricity trading data in {time.time()-start:.2f} seconds.\n')
    
    return cbet


def collect_data(first_datetime:str, last_datetime:str, verbose:bool=False) -> pd.DataFrame:

    '''
    start   :   Start date in ISO 8601 format (e.g., '2024-01-01T00:00:00Z').
    end     :   End date in ISO 8601 format (e.g., '2024-12-31T23:00:00Z').
    verbose :   Whether to print progress messages.
    '''

    alt_first_datetime = (pd.to_datetime(first_datetime, format='ISO8601', utc=True) + pd.Timedelta(hours=1)).strftime(r'%Y-%m-%dT%H:%M:%SZ')
    alt_last_datetime = (pd.to_datetime(last_datetime, format='ISO8601', utc=True) + pd.Timedelta(hours=1)).strftime(r'%Y-%m-%dT%H:%M:%SZ')

    weather = collect_weather_data(alt_first_datetime, alt_last_datetime, verbose)
    price = collect_price_data(first_datetime, last_datetime, verbose)
    production = collect_production_data(alt_first_datetime, alt_last_datetime, verbose)
    cbet = collect_cbet_data(alt_first_datetime, alt_last_datetime, verbose)

    start = time.time()
    df = reduce(
        lambda l, r : l.merge(r, on='datetime', how='outer'),
        [weather, price, production, cbet]
    ).sort_values(by='datetime', ascending=True)
    if verbose:
        print(f'Concatenated data in {time.time()-start:.2f} seconds.\n')

    return df