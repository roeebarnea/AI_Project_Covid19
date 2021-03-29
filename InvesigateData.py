import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats

def percentageOfNulls(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True)
    return missing_value_df


if __name__ == '__main__':
    WWC = pd.read_csv('WorldWideCountries-09-01-2021.csv')
    missing_value_df = percentageOfNulls(WWC)
    missing_value_df.to_csv('WorldWideCountries-09-01-2021_Nulls_Percentage.csv')