import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mushrooms = pd.read_csv('data/preprocessed/mushrooms_cleaned.csv')

sns.scatterplot(data=mushrooms,x='stem_height',y='stem_width',hue='class',alpha=0.25)
plt.savefig('sHeightvssWidth.jpg')
plt.show()
sns.scatterplot(data=mushrooms,x='stem_height',y='cap_diameter',hue='class',alpha=0.25)
plt.savefig('sHeightvscDiameter.jpg')
plt.show()
sns.scatterplot(data=mushrooms,x='cap_diameter',y='stem_width',hue='class',alpha=0.25)
plt.savefig('cDiametervssWidth.jpg')
plt.show()
