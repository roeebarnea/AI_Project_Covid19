import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def add_more_death(WWC):
    WWC['more_death'] = 0
    countries = WWC['location'].unique()
    for c in countries:
        CON = WWC.loc[WWC['location'] == c]
        dates = CON['date']
        is_first = True
        for d in dates:
            if is_first:
                WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['more_death']] =0
                is_first = False
                continue
            death_today = WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['new_deaths']]
            death_yesterday = WWC.loc[(WWC['date'] == (d - pd.DateOffset(days=1))) & (WWC['location'] == c), ['new_deaths']]

            if (death_today.size == 0):
                continue
            if (math.isnan(death_today.iat[0,0])):
                continue
            if (death_yesterday.size == 0):
                continue
            if (math.isnan(death_yesterday.iat[0,0])):
                continue

            if death_today.iat[0,0] > death_yesterday.iat[0,0] :
                WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['more_death']] = 1
            else:
                WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['more_death']] = -1

def add_more_new_cases(WWC):
    WWC['more_new_cases'] = 0
    countries = WWC['location'].unique()
    for c in countries:
        CON = WWC.loc[WWC['location'] == c]
        dates = CON['date']
        is_first = True
        for d in dates:
            if is_first:
                WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['more_new_cases']] =0
                is_first = False
                continue
            new_cases_today = WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['new_cases']]
            new_cases_yesterday = WWC.loc[(WWC['date'] == (d - pd.DateOffset(days=1))) & (WWC['location'] == c), ['new_cases']]

            if (new_cases_today.size == 0):
                continue
            if (math.isnan(new_cases_today.iat[0,0])):
                continue
            if (new_cases_yesterday.size == 0):
                continue
            if (math.isnan(new_cases_yesterday.iat[0,0])):
                continue

            if new_cases_today.iat[0,0] > new_cases_yesterday.iat[0,0] :
                WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['more_new_cases']] = 1
            else:
                WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['more_new_cases']] = -1

def add_30_columns(WWC):
    for i in range(1, 31):
        s = 'new_death_' + str(i) + '_day_ago'
        WWC[s] = 0
    for i in range(1, 31):
        s = 'new_cases_' + str(i) + '_day_ago'
        WWC[s] = 0

def add_new_death_30_ago(WWC):
    WWC['more_death'] = 0
    countries = WWC['location'].unique()
    for c in countries:
        CON = WWC.loc[WWC['location'] == c]
        dates = CON['date']
        is_first = True
        for d in dates:
            for i in range(1,31):
                s = 'new_death_' + str(i) + '_day_ago'
                death_ago = WWC.loc[
                    (WWC['date'] == (d - pd.DateOffset(days=i))) & (WWC['location'] == c), ['new_deaths']]

                if (death_ago.size == 0) :
                    continue

                if (math.isnan(death_ago.iat[0, 0])):
                    continue

                WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), [s]] = death_ago.iat[0, 0]

def add_new_cases_30_ago(WWC):
    WWC['more_death'] = 0
    countries = WWC['location'].unique()
    for c in countries:
        CON = WWC.loc[WWC['location'] == c]
        dates = CON['date']
        is_first = True
        for d in dates:
            for i in range(1, 31):
                s = 'new_cases_' + str(i) + '_day_ago'
                case_ago = WWC.loc[
                    (WWC['date'] == (d - pd.DateOffset(days=i))) & (WWC['location'] == c), ['new_cases']]

                if (case_ago.size == 0) :
                    continue

                if (math.isnan(case_ago.iat[0, 0])):
                    continue

                WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), [s]] = case_ago.iat[0, 0]

def add_stats_columns(WWC):
    WWC['Cases_STATS_min'] = 0
    WWC['Cases_STATS_max'] = 0
    WWC['Cases_STATS_avg'] = 0
    WWC['Cases_STATS_std'] = 0
    WWC['Cases_STATS_var'] = 0
    WWC['Cases_STATS_percentile_25'] = 0
    WWC['Cases_STATS_percentile_50'] = 0
    WWC['Cases_STATS_percentile_75'] = 0
    WWC['Cases_STATS_entropy'] = 0

    WWC['Death_STATS_min'] = 0
    WWC['Death_STATS_max'] = 0
    WWC['Death_STATS_avg'] = 0
    WWC['Death_STATS_std'] = 0
    WWC['Death_STATS_var'] = 0
    WWC['Death_STATS_percentile_25'] = 0
    WWC['Death_STATS_percentile_50'] = 0
    WWC['Death_STATS_percentile_75'] = 0
    WWC['Death_STATS_entropy'] = 0

def add_statistic_30(WWC):
    WWC['more_death'] = 0
    countries = WWC['location'].unique()
    for c in countries:
        CON = WWC.loc[WWC['location'] == c]
        dates = CON['date']
        for d in dates:
            arr_c = np.array([])
            arr_d = np.array([])
            for i in range(1, 31):
                st_cases = 'new_cases_' + str(i) + '_day_ago'
                st_death = 'new_death_' + str(i) + '_day_ago'

                var_cases = WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), [st_cases]].iat[0, 0]
                var_death = WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), [st_death]].iat[0, 0]

                arr_c = np.append(arr_c, [var_cases])
                arr_d = np.append(arr_d, [var_death])

            series_cases = pd.Series(arr_c)
            series_death = pd.Series(arr_d)

            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Cases_STATS_min']] = series_cases.min()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Cases_STATS_max']] = series_cases.max()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Cases_STATS_avg']] = series_cases.mean()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Cases_STATS_std']] = series_cases.std()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Cases_STATS_var']] = series_cases.var()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Cases_STATS_percentile_25']] = \
                series_cases.quantile(0.25)
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Cases_STATS_percentile_50']] = \
                series_cases.quantile(0.5)
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Cases_STATS_percentile_75']] = \
                series_cases.quantile(0.75)
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Cases_STATS_entropy']] = \
                scipy.stats.entropy(series_cases)

            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Death_STATS_min']] = series_death.min()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Death_STATS_max']] = series_death.max()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Death_STATS_avg']] = series_death.mean()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Death_STATS_std']] = series_death.std()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Death_STATS_var']] = series_death.var()
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Death_STATS_percentile_25']] = \
                series_death.quantile(0.25)
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Death_STATS_percentile_50']] = \
                series_death.quantile(0.5)
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Death_STATS_percentile_75']] = \
                series_death.quantile(0.75)
            WWC.loc[(WWC['date'] == d) & (WWC['location'] == c), ['Death_STATS_entropy']] = \
                scipy.stats.entropy(series_death)

def add_GDP(WWC):
    WWC['GDP'] = 0
    countries = WWC['location'].unique()
    GDP = pd.read_csv('GDP_number.csv')
    for c in countries:
        c_gdp = GDP.loc[(GDP['countries'] == c), ['GDP']]

        if (c_gdp.size == 0):
            continue


        WWC.loc[(WWC['location'] == c), ['GDP']] = c_gdp.iat[0, 0];

def remove_all_nulls(WWC):
    WWC.drop(['iso_code', 'continent', 'total_cases', 'total_deaths', 'new_deaths_smoothed',
              'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million', 'icu_patients',
              'icu_patients_per_million', 'hosp_patients', 'hosp_patients_per_million', 'hosp_patients',
              'hosp_patients_per_million', 'weekly_icu_admissions', 'weekly_icu_admissions_per_million',
              'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million', 'new_tests', 'total_tests',
              'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_units', 'total_vaccinations',
              'total_vaccinations_per_hundred', 'aged_65_older', 'aged_70_older',
              'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers',
              'handwashing_facilities'], axis=1, inplace=True)

    return (WWC.dropna())

def change_negative_to_zero2(WWC):
    for col in WWC.columns:
        if (col in ['location', 'date']):
            continue
        WWC[col][WWC[col] < 0] = 0


def create_WWC_DATA():
    WWC = pd.read_csv('WorldWideCountries-09-01-2021.csv')
    WWC['date'] = pd.to_datetime(WWC['date'])
    remove_all_nulls(WWC)
    change_negative_to_zero2(WWC)
    WWC.to_csv('WWC_clean_09_01_2021.csv')

    add_more_death(WWC)
    add_more_new_cases(WWC)
    add_30_columns(WWC)
    add_new_death_30_ago(WWC)
    add_new_cases_30_ago(WWC)
    add_stats_columns(WWC)
    add_statistic_30(WWC)
    add_GDP(WWC)

    WWC.to_csv('WWC_all_data_clean_09_01_2021.csv')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # WWC = pd.read_csv('WorldWideCountries-26-12-2020.csv')


    # ISR = WWC.loc[WWC['location'] == 'Israel']
    # ISR.to_csv('Israel.csv', index=True)

    # ISR = pd.read_csv('Israel.csv')


    # ISR['date'] = pd.to_datetime(ISR['date'])
    # add_more_death(ISR)
    # add_more_new_cases(ISR)
    #
    # add_30_columns(ISR)
    # add_new_death_30_ago(ISR)
    # add_new_cases_30_ago(ISR)
    # ISR.to_csv('Israel_30.csv')


    # # ISR = pd.read_csv('Israel_30.csv')
    # add_stats_columns(ISR)
    # add_statistic_30(ISR)
    # ISR.to_csv('Israel_30_stats.csv')

    # ISR = pd.read_csv('Israel_30_stats.csv')
    # add_GDP(ISR)
    #
    # ISR.to_csv('test2.csv')

    create_WWC_DATA()


    print('bla')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
