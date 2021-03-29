import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
from numbers import Number

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

def change_negative_to_zero(WWC):
    num = WWC._get_numeric_data()
    num[num < 0] = 0

def change_negative_to_zero2(WWC):
    for col in WWC.columns:
        if (col in ['location', 'date']):
            continue
        WWC[col][WWC[col] < 0] = 0

def fill_col_with_zero_insted_null(WWC, col):
    WWC[[col]] = WWC[[col]].fillna(value=0)

def cleaning(WWC):
    WWC.drop(['new_vaccinations', 'new_vaccinations_per_million'], axis=1, inplace=True)
    fill_col_with_zero_insted_null(WWC, 'new_cases')
    fill_col_with_zero_insted_null(WWC, 'new_cases_smoothed')
    fill_col_with_zero_insted_null(WWC, 'new_deaths')
    fill_col_with_zero_insted_null(WWC, 'total_cases_per_million')
    fill_col_with_zero_insted_null(WWC, 'new_cases_per_million')
    fill_col_with_zero_insted_null(WWC, 'total_deaths_per_million')
    fill_col_with_zero_insted_null(WWC, 'new_deaths_per_million')
    fill_col_with_zero_insted_null(WWC, 'reproduction_rate')

def complete_cols_with_average(WWC):
    WWC['total_tests_per_thousand'].fillna((WWC['total_tests_per_thousand'].mean()), inplace=True)
    WWC['new_tests_per_thousand'].fillna((WWC['new_tests_per_thousand'].mean()), inplace=True)
    WWC['positive_rate'].fillna((WWC['positive_rate'].mean()), inplace=True)
    WWC['tests_per_case'].fillna((WWC['tests_per_case'].mean()), inplace=True)
    WWC['stringency_index'].fillna((WWC['stringency_index'].mean()), inplace=True)
    WWC['population_density'].fillna((WWC['population_density'].mean()), inplace=True)
    WWC['median_age'].fillna((WWC['median_age'].mean()), inplace=True)
    WWC['gdp_per_capita'].fillna((WWC['gdp_per_capita'].mean()), inplace=True)
    WWC['hospital_beds_per_thousand'].fillna((WWC['hospital_beds_per_thousand'].mean()), inplace=True)
    WWC['human_development_index'].fillna((WWC['human_development_index'].mean()), inplace=True)


if __name__ == '__main__':
    WWC = pd.read_csv('test_WWC_all_data_clean_09_01_2021.csv')
    WWC.drop(['extreme_poverty'], axis=1, inplace=True)

    complete_cols_with_average(WWC)
    WWC.dropna(inplace=True)
    WWC.to_csv('WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv')



    #WWC.dropna(inplace=True)
    #WWC.to_csv('WWC_all_data_clean_09_01_2021_no_nulls.csv')

