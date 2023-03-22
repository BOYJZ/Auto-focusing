import numpy as np
import random
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def noisy_gaussian(x, y, z, SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z):
  A=random.randint(3000, 493000)
  B=7000
  C=7000
  f = A*np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  noisy_f =  f + (A + B) * np.random.normal(0, 1/SNR, f.shape) + C
  return noisy_f

'''
num_samples: number of samples
num_points: number of points in each samples
[lower_snr,higher_snr]: range of SNR of each samples
'''
def generate_dataset(num_samples,num_points,lower_snr,higher_snr):
    X = []
    for i in range(num_samples):
        X.append(0)
    y = np.zeros((num_samples, 3))
    snr=[]
    for i in range(num_samples):
        SNR = np.random.uniform(lower_snr, higher_snr)
        #maximum location
        mu_x = np.random.uniform(-1, 1)
        mu_y = np.random.uniform(-1, 1)
        mu_z = np.random.uniform(-1, 1)
        #sigma_x = np.random.uniform(0.5, 2)
        #sigma_y = np.random.uniform(0.5, 2)
        #sigma_z = np.random.uniform(0.5, 2)
        sigma_x,sigma_y,sigma_z=1,1,1

        points_per_dim = int(np.ceil(num_points**(1/3.)))

        dx = np.linspace(-1, 1, points_per_dim)
        dy = np.linspace(-1, 1, points_per_dim)
        dz = np.linspace(-1, 1, points_per_dim)

        xx, yy, zz = np.meshgrid(dx, dy, dz)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[:num_points]

        values=[]
        for k in range(num_points):
            values.append(noisy_gaussian(points[k][0], points[k][1], points[k][2], SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z))
        X[i] = np.array(values).T
        y[i] = np.array([mu_x, mu_y, mu_z])
        snr.append(SNR)
    return np.array(X), np.array(y), np.array(snr)

########################################################################################
num_samples=10
num_points=1000
lower_snr=0.1
higher_snr=100
SNR=10
########################################################################################

# Generate the dataset
X_train, y_train,snr_train = generate_dataset(num_samples,num_points,lower_snr,higher_snr)

# Reshape the input data to match the expected input shape of the model
X_train_input = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Define the input shape
input_shape = (num_points, 1)

# Define the model architecture
model = models.Sequential()

# Add the convolutional layers
model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=1024, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))

# Flatten the output of the convolutional layers
model.add(layers.Flatten())

# Add the dense layers with dropout
model.add(layers.Dense(512, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='linear'))

# Compile the model
model.compile(optimizer='Nadam', loss='mean_squared_error', metrics=['mean_absolute_error'])

#fit the model
model.fit(X_train_input, y_train, epochs=10)

#save the model
#model.save('my_model.h5')

#how to load model?
#from keras.models import load_model
#model = load_model('my_model.h5')


#x=[1,1,1,1,1,1]
#padded_x = np.pad(x, (0, num_points-len(x)), 'constant', constant_values=0)



lower_snr,higher_snr=1,100
X_test, y_test,snr = generate_dataset(100,num_points,lower_snr,higher_snr)
#prediction
y_pred = model.predict(X_test)
success=0
x_snr=[]
y_pro=[]
total_number=[]
num_parts=5
for i in range(num_parts):#split the snr into 5 parts, convert continuous snr to discrete snr
    x_snr.append(lower_snr+(higher_snr-lower_snr)*i/(num_parts-1))
    y_pro.append(0)
    total_number.append(0)
for i in range(len(y_pred)):
    distance = np.sqrt(  (y_pred[i][0]-y_test[i][0])**2 + (y_pred[i][1]-y_test[i][1])**2 + (y_pred[i][2]-y_test[i][2])**2  )
    tem_snr = min(x_snr, key=lambda x: abs(x-snr[i]))
    #print(int( (tem_snr-lower_snr) / ((higher_snr-lower_snr)/(num_parts-1)) ))
    total_number[int( (tem_snr-lower_snr) / ((higher_snr-lower_snr)/(num_parts-1)) )]+=1
    print('distance',distance)
    #print('test',y_test[i])
    #print('prediction',y_pred[i])
    #print('#####################################################################')
    if distance < 0.225:
        success+=1
        y_pro[int( (tem_snr-lower_snr) / ((higher_snr-lower_snr)/(num_parts-1)) )]+=1
for i in range(num_parts):
    y_pro[i]=y_pro[i]/total_number[i]        
print(success/len(y_pred))
print('discrete snr is',x_snr)
print('#success in each snr is',y_pro)
print('#total examples in each snr is',total_number)
plt.plot(x_snr,y_pro)
plt.xlabel('discrete snr')
plt.ylabel('pro of success')
plt.title('Success Probability vs SNR')















