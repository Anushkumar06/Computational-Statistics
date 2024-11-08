import matplotlib.pyplot as plt
import numpy as np
data = np.random.randn(1000)
plt.hist(data, bins=30, density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram')
plt.show()