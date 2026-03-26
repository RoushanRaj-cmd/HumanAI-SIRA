import pandas as pd
import numpy as np

def fetch_jhu_csse_covid19(country="US"):
    """
    Fetches raw JHU CSSE COVID-19 time-series data for a specific country.
    """
    url_confirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    url_deaths = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    url_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
    
    try:
        df_c = pd.read_csv(url_confirmed)
        df_d = pd.read_csv(url_deaths)
        df_r = pd.read_csv(url_recovered)
        
        c = df_c[df_c['Country/Region'] == country].iloc[:, 4:].sum(axis=0).values
        d = df_d[df_d['Country/Region'] == country].iloc[:, 4:].sum(axis=0).values
        r = df_r[df_r['Country/Region'] == country].iloc[:, 4:].sum(axis=0).values
        
        i = c - r - d
        
        return {
            'confirmed': c,
            'deaths': d,
            'recovered': r,
            'active': i
        }
    except Exception as e:
        print(f"Failed to fetch real-world data: {e}")
        return None

def normalize_real_data(data, population):
    """
    Converts raw historical sequences to fractions for the PINN model.
    """
    s = population - data['confirmed']
    i = data['active']
    r = data['recovered'] + data['deaths']
    
    t = np.arange(len(s))
    
    S_frac = s / population
    I_frac = i / population
    R_frac = r / population
    
    return t, np.vstack([S_frac, I_frac, R_frac]).T
