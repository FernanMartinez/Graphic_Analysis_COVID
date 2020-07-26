import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DEATH_BY_COV = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
CASES_COV = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'


def data_preparation(url):
    """
    Function to prepare the data to be analyzed.
    """
    data_df = pd.read_csv(url)
    colums_exclud = data_df.columns[[0, 2, 3]]
    data_df.drop(colums_exclud,
                 axis=1,
                 inplace=True)
    data_df = data_df.set_index("Country/Region")
    data_df = data_df.groupby(level=0).sum()
    return data_df


def data_country(cases_table, death_table, country):
    """
    Function to create a matrix with data of confirmed cases of COVID-19 and
    death by COVID-19. The first column is data-times, the second column is
    confirmed cases of COVID-19, and the third column is deaths by COVID-19.
    PARAMETERS
    ----------
    data_table: DataFrame with cases
    death_table: DataFrame with deaths
    country: Country to analize
    """
    data_cases = cases_table.loc[country]
    data_death = death_table.loc[country]
    data_array = np.c_[data_cases.index,
                       data_cases.values,
                       data_death.values]
    return data_array


# ------------------------------------------------
cases_df = data_preparation(url=CASES_COV)
death_df = data_preparation(url=DEATH_BY_COV)
# ------------------------------------------------

# def country_plot(country, subplot=True, save=False)
country = "Brazil"
data_df = data_country(cases_table=cases_df,
                       death_table=death_df,
                       country=country)
cases = data_df[:, [0, 1]]
print(cases)
