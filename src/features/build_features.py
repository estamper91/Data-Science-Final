import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mushrooms = pd.read_csv('data/preprocessed/mushrooms_cleaned.csv')
print(mushrooms.head())


# is a variable useful???
variables_cat = ['bruise_bleed', 'has_ring', 'season', 'habitat', 'cap_shape', 'cap_color', 'gill_color', 'stem_color']
print(mushrooms['class'].value_counts())

for item in variables_cat:
    df = mushrooms[['class', item]]
    srt_item = df.groupby(item)['class'].value_counts(normalize = True).unstack()
    plot = srt_item.plot(kind = 'bar', stacked = True,cmap='Paired')
    plt.title(item)
    plt.legend(labels=['Edible','Poisonous'])
    plt.savefig(item+" Bar Chart.jpg")
    plt.show()

#I also ran it with normalize = True in the parentheses for 
# .value_counts().

#It was observed that specimens that were found in the 'path' 
# habitat were always poisonous, and specimens found in the 
# 'urban' or 'waste' habitats were always edible. 
# Similarly, specimens with the stem color 'buff' were always 
# found to be edible, whereas if the stem color fell into 
# 'other' they were always poisonous. 

#In the gill color variable, if the gills were identified to 
# be red or brown in color, there were more likely to be 
# poisonous, in contrast, if the gill color was identified as 
# buff, then it was significantly more likely to be classed as 
# edible. 

#If the mushroom's cap color was identified as green, red, 
# pink, orange, or purple, the mushroom has a much higher 
# likelihood of being classified as poisonous. And similar 
# to the gill color, if the color was identified as buff, 
# then it was significantly more likely to be classed as edible. 

#In cap shape, if the mushroom was identified with a bell cap
# shape or put into the 'other' category, then it had a higher 
# proportion of poisonous mushrooms. 

variables_num = mushrooms[['cap_diameter', 'stem_height', 'stem_width']]

for var in variables_num:
    data = mushrooms[['class', var]]
    data.groupby('class')[var].hist(bins=20,alpha=0.5, legend=True)
    plt.title(var)
    plt.legend(labels=['Edible','Poisonous'])
    plt.savefig(var+" Histogram.jpg")
    plt.show()

#The most notable observation is that when any of the three
# numerical variables are high, they are more likely to be 
# edible than if they are small. 
