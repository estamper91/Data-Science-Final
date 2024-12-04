import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mushrooms_nona = pd.read_csv('/Data-Science-Final/data/raw/mushrooms.csv')

## Replacing values 


col_dict = {'n':'brown', 'b':'buff', 'g':'gray', 'r':'green', 'p':'pink', 'k':'black', 'o':'orange', 'u':'purple', 'e':'red', 'w':'white', 'y':'yellow', 'l':'blue', 'f':'none'}

class_bool = mushrooms_nona['class'].replace({'p':0, 'e':1})
bruise_bleed_bool = mushrooms_nona['does-bruise-or-bleed'].replace({'f':0, 't':1})
ring_bool = mushrooms_nona['has-ring'].replace({'f':0, 't':1})
season_num = mushrooms_nona['season'].replace({'s':0, 'u':1, 'a':2, 'w':3})
habitat_full = mushrooms_nona['habitat'].replace({'g':'grasses', 'l':'leaves', 'm':'meadows', 'p':'paths', 'h':'heaths', 'u':'urban', 'w':'waste', 'd':'woods'})
cap_s = mushrooms_nona['cap-shape'].replace({'b':'bell', 'c':'conical', 'x':'convex', 'f':'flat', 's':'sunken', 'p':'spherical', 'o':'other'})
cap_c = mushrooms_nona['cap-color'].replace(col_dict)
gill_c = mushrooms_nona['gill-color'].replace(col_dict)
stem_c = mushrooms_nona['stem-color'].replace(col_dict)


new = {'class':class_bool, 'bruise_bleed':bruise_bleed_bool, 'has_ring':ring_bool, 'season':season_num, 'habitat':habitat_full,
       'cap_diameter':mushrooms_nona['cap-diameter'], 'cap_shape':cap_s, 'cap_color':cap_c, 'gill_color':gill_c,
       'stem_height':mushrooms_nona['stem-height'], 'stem_width':mushrooms_raw['stem-width'], 'stem_color':stem_c}
mushrooms = pd.DataFrame(data = new)
print(mushrooms.head())
print(mushrooms.columns)

#saving the cleaned data
mushrooms.to_csv('Data-Science-Final/data/preprocessed/mushrooms_cleaned.csv', index=False)

#checking the data
mC = pd.read_csv('mushrooms_cleaned.csv')
print(mC.head())
#testing adjusted keys
print(mC['stem_color'].unique())

print(mC['class'].value_counts())
