import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import data

mushrooms_raw = pd.read_csv('/Data-Science-Final/data/raw/secondary_data.csv',sep=';')
print(mushrooms_raw)

print(mushrooms_raw.isna().sum())

## Only cells without null values?

variables = ['class', 'cap-diameter', 'cap-shape', 'cap-color', 'does-bruise-or-bleed', 'gill-color', 'stem-height', 'stem-width', 'stem-color', 'has-ring', 'habitat', 'season']
mushrooms_nona = mushrooms_raw[variables]
print(mushrooms_nona.isna().sum())
mushrooms.to_csv('Data-Science-Final/data/raw/mushrooms.csv', index=False)
