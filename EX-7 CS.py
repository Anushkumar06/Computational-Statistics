import numpy as np
from scipy.linalg import eig
def lda(X, y):
  
  classes = np.unique(y)
  mean_overall = np.mean(X, axis=0)
  mean_vectors = []
  for c in classes:
    mean_vectors.append(np.mean(X[y==c], axis=0))
  mean_vectors = np.array(mean_vectors)
  S_W = np.zeros((X.shape[1], X.shape[1]))
  S_B = np.zeros((X.shape[1], X.shape[1]))
  for i, mean_vec in enumerate(mean_vectors):
    class_i = X[y==classes[i]]
    S_W += (class_i - mean_vec).T @ (class_i - mean_vec)
    n = class_i.shape[0]
    mean_diff = mean_vec - mean_overall
    S_B += n * (mean_diff).reshape(-1, 1) @ (mean_diff).reshape(1, -1)
  eigenvalues, eigenvectors = eig(S_B, S_W)
  idx = eigenvalues.argsort()[::-1]
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:, idx]
  X_lda = X @ eigenvectors
  return X_lda