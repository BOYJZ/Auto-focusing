import numpy as np
import random
import matplotlib.pyplot as plt
def noisy_gaussian(x, y, z, SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z):
  A=random.randint(3000, 493000)
  B=7000
  C=7000
  f = A*np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  noisy_f =  f + ( A+ B) * np.random.normal(0, 1/SNR, f.shape) + C
  return noisy_f
x=[]
y=[]
for i in range(1000):
    x.append(i)
    y.append(noisy_gaussian(0,0,0,1000000,0,0,0,1,1,1))
plt.plot(x,y)
plt.xlabel('time')
plt.ylabel('intensity')