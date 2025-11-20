
import numpy as np

class NormalDistribution:
  def __init__(self, mean = 0, std_dev = 1):
    self.mean = mean
    self.std_dev = std_dev

  def N(self, x):
    coeff = 1 / (self.std_dev * np.sqrt(2 * np.pi))
    exponent = - (x - self.mean) ** 2 / (2 * self.std_dev ** 2)
    return coeff * np.exp(exponent)

if __name__ == "__main__":
  normal_dist = NormalDistribution()

  import matplotlib.pyplot as plt
  x = np.linspace(-5, 5, 1000)
  y = normal_dist.N(x)
  plt.plot(x, y)
  plt.title('Standard Normal Distribution')
  plt.xlabel('x')
  plt.ylabel('N(x|0,1)')
  plt.grid()
  plt.legend(['N(x|0,1)'])
  plt.show()
