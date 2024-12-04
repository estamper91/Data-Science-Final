# Data-Science-Final
## Overview
**This classification assignments seeks to test three different models on mushrooms to identify them as edible or poisonous.**   

This project contains several folders with original and cleaned data, code to represent and allow for recreation of the results, and several visualizations for data analysis. A notebook is included which contains code and results. 

## Dataset 
**Source:** https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset  

**Number of rows:** 61,068  

**Features:** cap-diameter, cap-shape, cap-surface (descriptive, eg. “silky”, “fibrous”, etc.), cap-color, does-bruise-bleed (binary), gill-attachment, gill-spacing, gill-color, stem-height, stem-width, stem-root (descriptive, eg. “bulbous”), stem-surface (see “cap-surface”), stem-color, veil-type (partial or universal), veil-color, has-rings (binary), ring-type (descriptive, eg. “large”, “flaring”, etc.), spore-print-color, habitat, season, class (edible or poisonous)   

**Initial Observations:** The dataset is well maintained and contains many string-type variables. However, the formatting is consistent and requires minimal cleaning. The main portion of data-wrangling involved might be keying the strings into numeric values. The dataset also contains several binary variables. For our purposes, we will be examining whether a mushroom is poisonous or not. 
