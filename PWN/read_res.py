import pandas as pd

name = "res/experiments/01_23_2024_02_32_08__PWNEM-ReadM4_Hourly.pkl"
unpickled_df = pd.read_pickle(name)
print(unpickled_df[3]['SMAPE'])

y=0
for x in unpickled_df[3]['SMAPE']:
    y += unpickled_df[3]['SMAPE'][x]

#x = y/4
y = y/414
print(y)