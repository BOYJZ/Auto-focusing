import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def noisy_gaussian(x, y, z, SNR=10, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  noisy_f =  f + np.random.normal(0, 1/SNR, f.shape)  
  return noisy_f


def fit_ensemble(x, y, z, n_estimators=100, max_depth=50,snr=1000):
    # Fit an ensemble of decision trees
    estimators = []
    for i in range(n_estimators):
        model = DecisionTreeRegressor(max_depth=max_depth)
        # Generate a random subset of the data for this model
        idx = np.random.choice(len(x), size=len(x), replace=True)
        x_sub = x[idx]
        y_sub = y[idx]
        z_sub = z[idx]
        f_sub=0
        num=1
        for i in range(num):
            f_sub += noisy_gaussian(x_sub, y_sub, z_sub,snr)
        f_sub/=num
        # Fit the model on the noisy data
        model.fit(np.column_stack((x_sub, y_sub, z_sub)), f_sub)
        estimators.append(model)
    return estimators

def predict_ensemble(x, y, z, estimators):
    # Predict the noisy function using each model
    predictions = np.zeros(len(x))
    for model in estimators:
        predictions += model.predict(np.column_stack((x, y, z)))
    # Average the predictions to get the final prediction
    prediction = predictions / len(estimators)
    # Find the location of the maximum of the predicted function
    idx_max = np.argmax(prediction)
    x_max = x[idx_max]
    y_max = y[idx_max]
    z_max = z[idx_max]
    return x_max, y_max, z_max

asnr=[0.0000001,1,2,3,4,5,6,7,10,15,20,30]
u=[]
for SNR in asnr:
    success=0
    for i in range(100):
    
        # Generate some noisy data
        x = np.linspace(-np.sqrt(3), np.sqrt(3), 150)
        y = np.linspace(-np.sqrt(3), np.sqrt(3), 150)
        z = np.linspace(-np.sqrt(3), np.sqrt(3), 150)


        # Split the data into training and testing sets
        idx_train = np.random.choice(len(x), size=int(0.8 * len(x)), replace=False)
        idx_test = np.setdiff1d(range(len(x)), idx_train)
        x_train, y_train, z_train= x[idx_train], y[idx_train], z[idx_train]
        x_test, y_test, z_test= x[idx_test], y[idx_test], z[idx_test]


        # Fit the ensemble on the training data
        estimators = fit_ensemble(x_train, y_train, z_train, n_estimators=10, max_depth=5,snr=SNR)

        # Make predictions on the testing data
        x_pred, y_pred, z_pred = predict_ensemble(x_test, y_test, z_test, estimators)

        # Print the predicted location of the maximum
        #print(f"Predicted location of maximum: ({x_pred:.2f}, {y_pred:.2f}, {z_pred:.2f})")
        if np.sqrt(x_pred**2+y_pred**2+z_pred**2)<0.225:
            #print('success!')
            success+=1
        #else:
            #print('failure!')
    print(success/100)
    u.append(success/100)
    
plt.plot(asnr, u, marker='o',color='b')
plt.title('pro of sucess v.s. snr')
plt.xlabel('snr')
plt.ylabel('pro of sucess')
plt.show()


















