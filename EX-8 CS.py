import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
data = {'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'feature3': [3, 6, 9, 12, 15]}
df = pd.DataFrame(data)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)
pca = PCA(n_components=2)  # Choose the number of components
pca.fit(data_scaled)
data_pca = pca.transform(data_scaled)
print(data_pca)