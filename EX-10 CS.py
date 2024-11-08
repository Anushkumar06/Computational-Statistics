import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = {'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'feature3': [3, 6, 9, 12, 15]}
df = pd.DataFrame(data)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)
kmeans = KMeans(n_clusters=2)  # Choose the number of clusters
kmeans.fit(data_scaled)
labels = kmeans.predict(data_scaled)
print(labels)
