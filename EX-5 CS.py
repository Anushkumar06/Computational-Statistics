import numpy as np
from scipy.optimize import minimize
def exponential_pdf(x, lam):
  return lam * np.exp(-lam * x)
def log_likelihood(lam, data):
  return np.sum(np.log(exponential_pdf(data, lam)))
def mle_exponential(data):
  initial_guess = 1.0
  result = minimize(lambda lam: -log_likelihood(lam, data), initial_guess)
  return result.x[0]
data = np.random.exponential(scale=2, size=100)
estimated_lambda = mle_exponential(data)
print(estimated_lambda)
