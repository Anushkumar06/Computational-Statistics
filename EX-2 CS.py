import pandas as pd
iris_data = pd.read_csv("Iris.csv")  
print("First 10 records:\n", iris_data.head(10))
print("\nNumber of rows:", iris_data.shape[0])
print("Number of columns:", iris_data.shape[1])
print("\nColumn names:", iris_data.columns)
print("\nMean of all attributes:\n", iris_data.mean())