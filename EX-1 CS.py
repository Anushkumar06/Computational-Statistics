import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
iris = sns.load_dataset("iris")
attribute = 'sepal_length'
print(iris.head())
sns.histplot(iris[attribute], kde=True)
plt.title(f"Histogram and KDE of {attribute}")
plt.show()
mean = iris[attribute].mean()
std = iris[attribute].std()
x = range(int(iris[attribute].min()), int(iris[attribute].max()) + 1)
y = [1/(std * (2*3.14159)*0.5) * 2.71828(-0.5 * ((i-mean)/std)*2) for i in x]
plt.plot(x, y)
plt.title(f"Probability Distribution of {attribute}")
plt.show()
print("Skewness:", skew(iris[attribute]))
print("Kurtosis:", kurtosis(iris[attribute]))
