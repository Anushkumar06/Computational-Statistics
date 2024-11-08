import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot(data, attribute, target_attribute=None):
 
  print(data[attribute].describe())
  plt.figure(figsize=(8, 6))
  sns.boxplot(x=data[attribute])
  plt.title(f"Box Plot of {attribute}")
  plt.show()
  if target_attribute:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[attribute], y=data[target_attribute])
    plt.title(f"Dependency Curve of {attribute} vs {target_attribute}")
    plt.show()
data = pd.read_csv('your_data.csv')
attribute_to_analyze = 'attribute_name'
target_attribute = 'target_attribute_name'  
analyze_and_plot(data, attribute_to_analyze, target_attribute)