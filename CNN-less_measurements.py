import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def noisy_gaussian(x, y, z, SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,A,B,C):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  noisy_f =  A * f + A * np.random.normal(0, (1/SNR), f.shape) + C
  return noisy_f

'''
num_samples: number of samples
num_points: number of points in each samples
    [lower_snr,higher_snr]: range of SNR of each samples
center: detect center
input_snr, input_location: for the 4 last experiments, use same snr and location as first generation
'''
def generate_dataset(num_samples,num_points,lower_snr,higher_snr):
    X = np.zeros((num_samples, num_points))
    y = np.zeros((num_samples, 3))
    array_A=[]
    array_B=[]
    snr=[]
    for i in range(num_samples):
        A=np.random.randint(3000, 493000)
        B=np.random.randint(300,7000)
        C=B
        
        #SNR = np.random.uniform(lower_snr, higher_snr)
        SNR = abs(np.random.normal(0, std_dev))

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
        values = noisy_gaussian(points[:,0], points[:,1], points[:,2], SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,A,B,C)
        max_value=max(values)
        normalized_values = [x / max_value for x in values]
        X[i] = np.array(normalized_values).T
        y[i] = np.array([mu_x, mu_y, mu_z])
        
        #save
        array_A.append(A)
        array_B.append(B)
        snr.append(SNR)
    return np.array(X), np.array(y), np.array(snr), array_A, array_B

def generate_dataset_pro(num_samples,num_points,lower_snr,higher_snr):
    X = np.zeros((num_samples, num_points))
    y = np.zeros((num_samples, 3))
    array_A=[]
    array_B=[]
    snr=[]
    for i in range(num_samples):
        A=np.random.randint(3000, 493000)
        B=np.random.randint(300,7000)
        C=B
        
        SNR = np.random.uniform(lower_snr, higher_snr)
        #SNR = abs(np.random.normal(0, std_dev))

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
        values = noisy_gaussian(points[:,0], points[:,1], points[:,2], SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,A,B,C)
        max_value=max(values)
        normalized_values = [x / max_value for x in values]
        X[i] = np.array(normalized_values).T
        y[i] = np.array([mu_x, mu_y, mu_z])
        
        #save
        array_A.append(A)
        array_B.append(B)
        snr.append(SNR)
    return np.array(X), np.array(y), np.array(snr), array_A, array_B

def generate_dataset_with_set_snr(num_samples,num_points,snr):
    X = np.zeros((num_samples, num_points))
    y = np.zeros((num_samples, 3))

    for i in range(num_samples):
        A=np.random.randint(3000, 493000)
        B=np.random.randint(300,7000)
        C=B
        
        SNR = snr

        #maximum location
        mu_x = np.random.uniform(-1, 1)
        mu_y = np.random.uniform(-1, 1)
        mu_z = np.random.uniform(-1, 1)
        
        sigma_x,sigma_y,sigma_z=1,1,1

        points_per_dim = int(np.ceil(num_points**(1/3.)))

        dx = np.linspace(-1, 1, points_per_dim)
        dy = np.linspace(-1, 1, points_per_dim)
        dz = np.linspace(-1, 1, points_per_dim)

        xx, yy, zz = np.meshgrid(dx, dy, dz)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[:num_points]

        values=[]
        values = noisy_gaussian(points[:,0], points[:,1], points[:,2], SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,A,B,C)
        max_value=max(values)
        min_value=min(values)
        normalized_values = [(x - min_value) / (max_value - min_value) for x in values]
        X[i] = np.array(normalized_values).T
        y[i] = np.array([mu_x, mu_y, mu_z])
        
    return np.array(X), np.array(y)


########################################################################################
num_samples=3000000
num_points=16
lower_snr=1
higher_snr=100
std_dev = 35
measurements=200
num_test=10000
train_epochs=10
########################################################################################



#how to load model?
from keras.models import load_model
model = load_model('my_model.h5')

def mask(x, num_points_to_add):
    masked=[]
    for i in range(len(x)):
        masked.append(0)
    selected_indices=[]
    for j in range(num_points_to_add):
        selected_indices.append(int(j/num_points_to_add*len(x)))

    for j in range(len(selected_indices)):
        masked[selected_indices[j]] = x[ selected_indices[j]]

    return masked

def draw_pro_snr():   
    lower_snr=0.1
    higher_snr=1
    X_test, y_test,snr,  array_A, array_B = generate_dataset_pro(num_test,num_points,lower_snr,higher_snr)
    y_pred = model.predict(X_test)
    success=0
    x_snr=[]
    y_pro=[]
    total_number=[]
    num_parts=50
    for i in range(num_parts):#split the snr into 5 parts, convert continuous snr to discrete snr
        x_snr.append(lower_snr+(higher_snr-lower_snr)*i/(num_parts-1))
        y_pro.append(0)
        total_number.append(0)
    for i in range(len(y_pred)):
        distance = np.sqrt(  (y_pred[i][0]-y_test[i][0])**2 + (y_pred[i][1]-y_test[i][1])**2 + (y_pred[i][2]-y_test[i][2])**2  )
        tem_snr = min(x_snr, key=lambda x: abs(x-snr[i]))
        #print(int( (tem_snr-lower_snr) / ((higher_snr-lower_snr)/(num_parts-1)) ))
        total_number[int( (tem_snr-lower_snr) / ((higher_snr-lower_snr)/(num_parts-1)) )]+=1
        #print('distance',distance)
        #print('test',y_test[i])
        #print('prediction',y_pred[i])
        #print('#####################################################################')
        if distance < 0.225:
            success+=1
            y_pro[int( (tem_snr-lower_snr) / ((higher_snr-lower_snr)/(num_parts-1)) )]+=1
    for i in range(num_parts):
        if total_number[i]==0:
            continue
        y_pro[i]=y_pro[i]/total_number[i]        
    print(success/len(y_pred))
    print('discrete snr is',x_snr)
    print('#success in each snr is',y_pro)
    print('#total examples in each snr is',total_number)
    print('#############################################################################')
    plt.plot(x_snr,y_pro)
    plt.xlabel('discrete snr')
    plt.ylabel('pro of success')
    plt.title('Success Probability vs SNR')
    plt.show()
    return


def draw_meas_snr():
    x_snr=[1.0, 3.020408163265306, 5.040816326530612, 7.061224489795919, 9.081632653061224, 11.10204081632653, 13.122448979591837, 15.142857142857142, 17.163265306122447, 19.183673469387756, 21.20408163265306, 23.224489795918366, 25.244897959183675, 27.26530612244898, 29.285714285714285, 31.306122448979593, 33.326530612244895, 35.3469387755102, 37.36734693877551, 39.38775510204081, 41.40816326530612, 43.42857142857143, 45.44897959183673, 47.46938775510204, 49.48979591836735, 51.51020408163265, 53.53061224489796, 55.55102040816327, 57.57142857142857, 59.59183673469388, 61.61224489795919, 63.63265306122449, 65.65306122448979, 67.6734693877551, 69.6938775510204, 71.71428571428571, 73.73469387755102, 75.75510204081633, 77.77551020408163, 79.79591836734694, 81.81632653061224, 83.83673469387755, 85.85714285714286, 87.87755102040816, 89.89795918367346, 91.91836734693878, 93.93877551020408, 95.95918367346938, 97.9795918367347, 100.0]
    
    #x_snr=[100]
    y=[]
    X_mask=[]
    signal=0
    num_test=100
    for snr in x_snr:
        X_test,y_test=generate_dataset_with_set_snr(num_test,num_points,snr)
        success=0
        X_mask=[]
        step=10
        total=int(200/step)
        for i in range(len(X_test)):
            for j in range(1,num_points,step):
                masked=mask(X_test[i],j)
                signal+=1
                X_mask.append(masked)
        y_pred=model.predict(X_mask)
        for k in range(len(X_test)):
            tem=num_points
            for i in range(total):
                
                index=k*total+i
                distance = np.sqrt(  (y_pred[index][0]-y_test[k][0])**2 + (y_pred[index][1]-y_test[k][1])**2 + (y_pred[index][2]-y_test[k][2])**2  )
                if distance < 0.225:
                    tem=i/total*num_points
                    break
            success+=tem
        success/=len(X_test)
        y.append(success)
    print(x_snr,y)
    plt.plot(x_snr,y)
    plt.ylabel('#measurements')
    plt.xlabel('snr')
    plt.show()

draw_pro_snr()
#draw_meas_snr()