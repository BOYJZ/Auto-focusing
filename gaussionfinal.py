import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import random
from scipy.special import logsumexp

def gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  return f

def noisy_gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  noisy_f =  f + np.random.normal(0, 1/SNR, f.shape)  
  return noisy_f

1

def gradient_ascent(noise,x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=1, learning_rate=0.01, max_iter=100, step_size = 0.04,aim_value=0.98,stop_steps=0.001):
  if noise==0:
      f = gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z)
  else:
      f = noisy_gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR)
  trajectory = [(x, y, z)]
  history_f=[]
  stop_signal=10
  tem_iter=0
  while tem_iter<max_iter :
    tem_iter+=1
    if noise==0:
        # Calculate the gradient of the Gaussian function at the current (x, y, z) point
        grad_x = (gaussian(x + step_size, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z) - gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z)) / step_size
        grad_y = (gaussian(x, y + step_size, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z) - gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z)) / step_size
        grad_z = (gaussian(x, y, z + step_size, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z) - gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z)) / step_size
    else:
        grad_x = (noisy_gaussian(x + step_size, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR) - noisy_gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR)) / step_size
        grad_y = (noisy_gaussian(x, y + step_size, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR) - noisy_gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR)) / step_size
        grad_z = (noisy_gaussian(x, y, z + step_size, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR) - noisy_gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR)) / step_size

    # Update the (x, y, z) point using the gradient and the learning rate
    x += learning_rate * grad_x
    y += learning_rate * grad_y
    z += learning_rate * grad_z

    # Plot the current point on the curve
    #plt.scatter(x, y, c='r')
    if noise==0:
        f = gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z)
    else:
        f = noisy_gaussian(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR)
    trajectory.append((x, y, z))
    
    
    if len(history_f)==0:
        history_f.append(f-stop_steps*2)
        history_f.append(f)
    else:
        history_f.append(f)
    s=len(history_f)
    #if history_f[s-1]-history_f[s-2]<stop_steps:
    #    stop_signal-=1
    #if history_f[s-1]-history_f[s-2]>stop_steps and stop_signal <3:
    #   stop_signal+=1
  # Return the final (x, y) point
  return x, y, z, f, trajectory,tem_iter,history_f
mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z = 0, 0, 0, 1, 1, 1

#x_point = float(input("Enter the x coordinate of the point: "))
#y_point = float(input("Enter the y coordinate of the point: "))
#z_point = float(input("Enter the z coordinate of the point: "))
#x_max, y_max, z_max, f_max, trajectory,total_iter,history_f1 =gradient_ascent(0,1, 1, 1, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=1000)
x_max, y_max, z_max, f_max, trajectory,total_iter,history_f2 =gradient_ascent(1,1, 1, 1, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=1000)
#x_max, y_max, z_max, f_max, trajectory,total_iter,history_f3 =gradient_ascent(0,0.5, 0.5,0.5, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=1000)
#x_max, y_max, z_max, f_max, trajectory,total_iter,history_f4 =gradient_ascent(1,0.5,0.5,0.5, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=1000)
#x_max, y_max, z_max, f_max, trajectory,total_iter,history_f5 =gradient_ascent(0,0.1,0.5,0.5, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=1000)
#x_max, y_max, z_max, f_max, trajectory,total_iter,history_f6 =gradient_ascent(1,0.1,0.5,0.5, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=1000)




x_max, y_max, z_max, f_max, trajectory,total_iter,history_f7 =gradient_ascent(1,1, 1, 1, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=1000)
x_max, y_max, z_max, f_max, trajectory,total_iter,history_f8 =gradient_ascent(1,1, 1, 1, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=500)
x_max, y_max, z_max, f_max, trajectory,total_iter,history_f9 =gradient_ascent(1,1, 1, 1, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=100)
x_max, y_max, z_max, f_max, trajectory,total_iter,history_f10 =gradient_ascent(1,1, 1, 1, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=10)
x_max, y_max, z_max, f_max, trajectory,total_iter,history_f11=gradient_ascent(1,1, 1, 1, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=1)
x_array=[]
for i in range(total_iter+1):
    x_array.append(i)
#plt.plot(history_f1)
#plt.plot(history_f2)
#plt.plot(history_f3)
#plt.plot(history_f4)
#plt.plot(history_f5)
#plt.plot(history_f6)
#plt.plot(history_f7,label = 'SNR = 1000')
#plt.plot(history_f8,label = 'SNR = 500')
#plt.plot(history_f9,label = 'SNR = 100')
#plt.plot(history_f10,label = 'SNR = 10')
#plt.plot(history_f11,label = 'SNR = 1')
#plt.xlabel('iteration')
#plt.ylabel('intensity')
#plt.title('history_f vs. iteration')
#plt.legend()
#plt.show()
#x_max, y_max, z_max, f_max, trajectory,total_iter,history_f =gradient_ascent(1,1,1, 1, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=500)
#plt.plot(history_f)
#plt.xlabel('iteration')
#plt.ylabel('intensity')
#plt.show()
aimpoint = (0,0,0)

def is_success(trajectory, aimpoint):
    covstarting = 0
    distance = 0
    for i in range(len(trajectory)):
        distance = np.sqrt((trajectory[i][0]-aimpoint[0])*(trajectory[i][0]-aimpoint[0])+(trajectory[i][1]-aimpoint[1])*(trajectory[i][1]-aimpoint[1])+(trajectory[i][2]-aimpoint[2])*(trajectory[i][2]-aimpoint[2]))
        if distance<0.225:
           covstarting = i
           break
        if i == len(trajectory)-1 :
            return 0,1000

    countsuc= 0
    countrest = 0
    for j in range(covstarting,len(trajectory)):
        countrest+=1
        if distance<0.225:
            countsuc+=1
    #print(countrest,countsuc)
    if  countsuc/countrest >=0.5:
        return 1,covstarting
    return 0,1000

'''
def is_success(trajectory, history_f):
    covstarting = 0
    for i in range(len(trajectory)-10):
        #distance = np.sqrt((trajectory[i][0]-aimpoint[0])*(trajectory[i][0]-aimpoint[0])+(trajectory[i][1]-aimpoint[1])*(trajectory[i][1]-aimpoint[1])+(trajectory[i][2]-aimpoint[2])*(trajectory[i][2]-aimpoint[2]))
        countsig = 0
        sum_f = 0
        while countsig<10:
           sum_f+= history_f[i+countsig]
           countsig+=1
        ave_f = sum_f/10
        if ave_f>0.9 :
           covstarting = i
           break
        if i == len(trajectory)-11 :
            return False,1000
    countsuc= 0
    countrest = 0
    for j in range(covstarting,len(trajectory)-10):
        countrest+=1
        countsig = 0
        sum_f = 0
        while countsig<10:
           sum_f+= history_f[i+countsig]
           countsig+=1
        ave_f = sum_f/10
        if ave_f>0.9 :
            countsuc+=1
    #print(countrest,countsuc)
    if  countsuc/countrest >=0.5:
        return True,covstarting+5
    return False,1000
'''

# cmd = 
# 1 for prob vs average distance
# 2 for iteration v.s. average distance
# 3 for pro of success v.s. snr(random position
# 4 for average iteration v.s. snr (for sucessful attempts)
# 5 for pro of success v.s. learning rate 
def gradient_draw(cmd):
    if cmd == 1 or cmd ==2:
        # plot prob v.s. average distance to target(snr=1000) 
        # plot iteration v.s. average distance to target(snr=1000) 
        iteration = []
        distance = []
        probability = []
        id_dic = {} #id_dic[distance] = sum of iteration
        pro_dic = {} # pro_dic[distance] = time of success
        pcd_dic = {} #pcd_dic[distance] = number of points
        icd_dic = {} #icd_dic[distance] = number of points but only contain success cases
        for r in np.linspace(0,1.73,10):
            for i in range(100):
                    #r = np.linspace(0,1,100)
                    theta = np.random.rand()*np.pi/2
                    phi = np.random.rand()*np.pi/2
                    x = r*np.sin(phi)*np.cos(theta)
                    y = r*np.sin(phi)*np.sin(theta)
                    z = r*np.cos(phi)
                    x_max, y_max, z_max, f_max, trajectory,total_iter,history_f =gradient_ascent(x,y,z, 1, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=10)
                    result, iterat_count = is_success(trajectory,aimpoint)
                    if r in id_dic :
                        if result ==1:
                            id_dic[r] += iterat_count
                            icd_dic[r] += 1
                        pro_dic[r] += result
                        pcd_dic[r] += 1
                    else:
                        if result ==1:
                            id_dic[r] = iterat_count
                            icd_dic[r] = 1
                        pro_dic[r] = result
                        pcd_dic[r] = 1
        for i in id_dic:
            ave_iterate = id_dic[i]/icd_dic[i]
            ave_prob = pro_dic[i]/pcd_dic[i]
            iteration.append(ave_iterate)
            distance.append(i)
            probability.append(ave_prob)
        if cmd == 2:
            plt.plot(distance,iteration)
            print('distance:',distance,'iteration:',iteration)
            plt.title('iteration v.s. average distance to target')
            plt.ylabel('iteration')
            plt.xlabel('distance from original point to aim point')
            plt.show()
            return


        if cmd == 1:
            print('distance:',distance,'probability:',probability)
            plt.plot(distance,probability)
            plt.title('pro of success v.s. initial average position ')
            plt.ylabel('probalility of success')
            plt.xlabel('distance from original point to aim point')
            plt.show()
            return
    elif cmd == 3 or cmd ==4:
        #pro of success v.s. snr(random position
        #average iteration v.s. snr (for sucessful attempts)
        pro_list = []
        snr_list = []
        iterat_list = []
        #pro_dict = {} #pro_dict[snr] = probability of success
        for i in [0.01,0.04,0.05,0.08,0.1,0.5,1,5,10]:#,50,100,300,400,500,600,700,800,1000]:
            stotal_count = 0
            #itotal_count = 0
            success_count = 0
            iterat_count = 0
            for j in range(100):
                x = random.random()
                y = random.random()
                z = random.random()
                
                
                x_max, y_max, z_max, f_max, trajectory,total_iter,history_f =gradient_ascent(1,x,y,z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=i)
                #plt.plot(history_f)
                #plt.show()
                result, iterat = is_success(trajectory,aimpoint)
                #if(result == 1):
                #    plt.plot(history_f)
                #    plt.show()
                #print(result)
                if result == 1: 
                    #itotal_count += 1
                    iterat_count += iterat
                    #print(itotal_count)
                stotal_count += 1
                success_count += result
                #print(success_count)
                    
            prob = success_count/stotal_count
            if success_count != 0:
                ave_iterat = iterat_count/success_count
                print(iterat_count,success_count)
                iterat_list.append(ave_iterat)
                snr_list.append(i)
            else:
                iterat_list.append(0)
                snr_list.append(i)
            pro_list.append(prob)
        
        #print(snr_list,iterat_list)

        if cmd ==4:
            plt.plot(snr_list,iterat_list)
            print('snr:',snr_list,'iteration',iterat_list)
            plt.title('average iteration v.s. SNR ')
            plt.ylabel('average iteration')
            plt.xlabel('snr')
            plt.show()
            return

        if cmd == 3:
            plt.plot(snr_list,pro_list)
            plt.title('pro of success v.s. snr ')
            plt.ylabel('probalility of success')
            plt.xlabel('snr')
            plt.show()
            return 

    else:
        pro_list = []
        snr_list = []
        iterat_list = []
        #pro_dict = {} #pro_dict[snr] = probability of success
        for i in [0.03,0.01,0.005,0.003,0.0001]:
            stotal_count = 0
            #itotal_count = 0
            success_count = 0
            iterat_count = 0
            for j in range(100):
                x = random.random()
                y = random.random()
                z = random.random()
                
                x_max, y_max, z_max, f_max, trajectory,total_iter,history_f =gradient_ascent(1,x,y,z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,SNR=10,learning_rate=i,step_size=0.04)
                result, iterat = is_success(trajectory,aimpoint)
                #if(result == 1):
                #    plt.plot(history_f)
                #    plt.show()
                #print(result)
                if result == 1: 
                    #itotal_count += 1
                    iterat_count += iterat
                    #print(itotal_count)
                stotal_count += 1
                success_count += result
                #print(success_count)
                    
            prob = success_count/stotal_count
            if success_count != 0:
                ave_iterat = iterat_count/success_count
                print(success_count)
                iterat_list.append(ave_iterat)
                snr_list.append(i)
            else:
                iterat_list.append(0)
                snr_list.append(i)
            pro_list.append(prob)
        plt.plot(snr_list,pro_list)
        plt.title('pro of success v.s. learning rate at SNR10 ')
        plt.ylabel('probalility of success')
        plt.xlabel('learning rate')
        plt.show()
        return



# cmd = 
# 1 for prob vs average distance
# 2 for iteration v.s. average distance
# 3 for pro of success v.s. snr(random position
# 4 for average iteration v.s. snr (for sucessful attempts)
# 5 for pro of success v.s. learning rate 
gradient_draw(5)