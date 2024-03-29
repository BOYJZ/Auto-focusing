import bayesian_optimization as bo #pip install bayesian-optimization, now local
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import SequentialDomainReductionTransformer
# Define the search space
pbounds = {'x': (0, 1),'y': (0, 1),'z': (0, 1)}

# Create the optimizer object

time=1
iteration=200
y=[]
values=[]
asnr=[10]
for snr in asnr:
    success=0
    for j in range(time):
        def noisy_gaussian(x, y, z,SNR=snr, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1):
          f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
          noisy_f =  f + np.random.normal(0, 1/SNR, f.shape)  
          return noisy_f
        bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.3)
        optimizer = bo.BayesianOptimization(f=noisy_gaussian, pbounds=pbounds,bounds_transformer=bounds_transformer)
        success_time=0
        #cal first iteration
        first=iteration
        optimizer.maximize(init_points=0,n_iter=iteration)#random search & number of iteration
        for iter, res in enumerate(optimizer.res):
            pos=res['params']
            distance=np.sqrt(pos['x']**2+pos['y']**2+pos['z']**2)
            if distance< 0.225:
                first=iter
                break
        if iteration==first:
                success_ratio=0
        else:
        #cal the stability
            success_time=0
        #print(first)
        for iter, res in enumerate(optimizer.res):
            if iter<=first:
                continue
            pos=res['params']
            distance=np.sqrt(pos['x']**2+pos['y']**2+pos['z']**2)
            #print('iteration',iter)
            #print(distance)
            values.append(res['target'])
            if distance< 0.225:
                success_time+=1
                print(iter,'at this time, BO get inside the location bar')
            success_ratio=success_time/(iteration-first)
        if success_ratio>0.5:
            success+=1
    y.append(success/time)

print('Bayes Optimisation',y)
plt.plot(asnr, y, marker='o',color='b')
plt.title('pro of sucess v.s. snr')
plt.xlabel('snr')
plt.ylabel('pro of sucess')
plt.show()
plt.plot(values)
plt.show()
