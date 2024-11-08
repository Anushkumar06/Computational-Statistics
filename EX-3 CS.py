import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_and_plot(data):
 
  total_records = len(data)
  print("Total number of records:", total_records)
  for column in data.columns:
    print(f"\nAttribute: {column}")
    print("Mean:", data[column].mean())
    print("Standard Deviation:", data[column].std())
  for column in data.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(data[column], bins=30, density=True, alpha=0.6, label='Histogram')
    mu, std = stats.norm.fit(data[column])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.show()
data = pd.read_csv('your_data.csv')
analyze_and_plot(data)