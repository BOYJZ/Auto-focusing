import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def gaussian(x, y, z, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  return f

def noisy_gaussian(x, y, z,SNR=1000, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  noisy_f =  f + np.random.normal(0, 1/SNR, f.shape)  
  return noisy_f

def update_velocity(v, x, y, z, pbest, gbest, w, c1, c2):
    r1 = random.random()
    r2 = random.random()
    return w*v + c1*r1*(pbest-np.array([x, y, z])) + c2*r2*(gbest-np.array([x, y, z]))

def update_position(x, y, z, v):
    x = x + v[0]
    y = y + v[1]
    z = z + v[2]
    return x, y, z

def run_pso(evaluate_function, num_particles, num_iterations, w, c1, c2, SNR=1000):
    # Initialize swarm
    swarm_x = np.random.uniform(-np.sqrt(3), np.sqrt(3), num_particles)
    swarm_y = np.random.uniform(-np.sqrt(3),np.sqrt(3), num_particles)
    swarm_z = np.random.uniform(-np.sqrt(3), np.sqrt(3), num_particles)
    initial_position = np.sqrt(swarm_x[0]**2 + swarm_y[0]**2 + swarm_z[0]**2)
    velocities = np.zeros((num_particles, 3))
    pbest = np.array([swarm_x, swarm_y, swarm_z]).T
    pbest_fitness = np.array([evaluate_function(x, y, z, SNR) for x, y, z in zip(pbest[:, 0], pbest[:, 1], pbest[:, 2])])
    gbest = pbest[np.argmax(pbest_fitness)]
    gbest_fitness = evaluate_function(gbest[0], gbest[1], gbest[2], SNR)
    gbest_history = [gbest]
    gbest_value = [gbest_fitness]
    iteration = 0
    phistory=[]
    for i in range(num_particles):
        phistory.append([])
    # Run iterations
    for i in range(num_iterations):
        for j in range(num_particles):
            iteration += 1
            fitness = evaluate_function(swarm_x[j], swarm_y[j], swarm_z[j], SNR)
            phistory[j].append([swarm_x[j], swarm_y[j], swarm_z[j],fitness])
            if fitness > pbest_fitness[j]:
                pbest[j] = np.array([swarm_x[j], swarm_y[j], swarm_z[j]])
                pbest_fitness[j] = fitness
                if fitness > gbest_fitness:
                    gbest = np.array([swarm_x[j], swarm_y[j], swarm_z[j]])
                    gbest_fitness = fitness
            velocities[j] = update_velocity(velocities[j], swarm_x[j], swarm_y[j], swarm_z[j], pbest[j], gbest, w, c1, c2)
            swarm_x[j], swarm_y[j], swarm_z[j] = update_position(swarm_x[j], swarm_y[j], swarm_z[j], velocities[j])

        gbest_history.append(gbest)
        gbest_value.append(gbest_fitness)
    current_position=[0,0,0]
    for i in range(num_particles):
        for j in range(3):
            current_position[j]+=phistory[i][-1][j]
    for j in range(3):
        current_position[j]/=num_particles
    
    return gbest, gbest_history, gbest_value[-1], initial_position, iteration,phistory,current_position



###################################################################
'''set parameter'''
num_iterations=1000
num_particles=15
###################################################################
def example():
    w = 0.5
    c1 = 1.5
    c2 = 2
    SNR = 100
    num_iterations = 10
    num_particles = 15####
    result=run_pso(noisy_gaussian, num_particles, num_iterations, w, c1, c2, 10)
    print('SNR=',SNR)
    print('current average position of all particles',result[-1])
    max_value=0
    for i in range(100):
        tem=noisy_gaussian(result[-1][0], result[-1][1], result[-1][2],SNR=1000, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1)
        if tem>max_value:
            max_value=tem
    print(max_value)
    print('position of global highest value point',result[0])
    print(result[2])

def PSO_animation():
    # Set PSO parameters
    w = 0.5
    c1 = 1.5
    c2 = 2
    SNR = 1000
    num_iterations = 100
    num_particles = 15
    # Run PSO algorithm
    gbest, gbest_history, gbest_fitness, initial_position, iteration, phistory,first_iteration,current_position = run_pso(noisy_gaussian, num_particles, num_iterations, w, c1, c2, SNR)
    
    # Create figure and 3D axes
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    # Define animation update function
    def update(i):
        ax.clear()
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.set_zlim(-1, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Particle Swarm Optimization')
        # Plot best particle in red
        ax.scatter(gbest_history[i][0], gbest_history[i][1], gbest_history[i][2], c='r', marker='o')
        # Plot all particles in blue
        for j in range(num_particles):
            data = np.array(phistory[j])

            ax.scatter(data[i][0], data[i][1], data[i][2], c='b', alpha=0.5)
        # Create animation
    anim = animation.FuncAnimation(fig, update, frames=num_iterations, interval=999999)
    # Show animation
    plt.show()
    anim.save('PSO_animation.gif', fps=999999)

    
    
def draw_success_pro_snr():
    iteration=int(1000/num_particles)
    #asnr=[1,5,10,15,20,22,24,26,28,30,35,38,40]
    asnr=[1,5,10,15,20,30,40]
    y=[]
    for snr in asnr:
        print(snr)
        success=0
        time=100
        for i in range(time):
            success_time=0
            result=run_pso(noisy_gaussian, num_particles, iteration, w=0.5, c1=1.5, c2=2., SNR=snr)
            #cal first iteration
            first=iteration
            j=result[5]
            for k in range(iteration):
                ave_x,ave_y,ave_z=0,0,0
                for s in range(num_particles):
                    ave_x+=j[s][k][0]
                    ave_y+=j[s][k][1]
                    ave_z+=j[s][k][2]
                ave_x/=num_particles
                ave_y/=num_particles
                ave_z/=num_particles
                ave=np.sqrt(ave_x**2+ave_y**2+ave_z**2)
                if ave< 0.225:
                    first=k
                    break
            if iteration==first:
                success_ratio=0
            else:
                #cal the stability
                success_time=0
                for k in range(iteration-first):  
                    ave_x,ave_y,ave_z=0,0,0
                    ave=0
                    for s in range(num_particles):
                        j=result[5][s][k]
                        ave_x+=j[0]
                        ave_y+=j[1]
                        ave_z+=j[2]
                    ave_x/=num_particles
                    ave_y/=num_particles
                    ave_z/=num_particles
                    ave=np.sqrt(ave_x**2+ave_y**2+ave_z**2)
                    if ave< 0.225:
                        success_time+=1
                success_ratio=success_time/(iteration-first)
            if success_ratio>0.5:
                success+=1
        y.append(success/time)
    
    k=[0.03, 0.33, 0.64, 0.82, 0.91, 0.97, 1.0]
    print('GPSO',k)
    print('PSO',y)
    plt.plot(asnr, y, marker='o',color='b')
    plt.plot(asnr,k, marker='o',color='r')
    plt.title('pro of sucess v.s. snr')
    plt.xlabel('snr')
    plt.ylabel('pro of sucess')
    plt.show()

def draw_iteration_success_snr():
    asnr=[5,10,100.300,400,500,600,700,800,1000]
    y=[]
    tem_iteration=int(1000/num_particles)
    for snr in asnr:
        print(snr)
        iteration=0
        success=100
        total=success
        iteration_sum=0
        while(success>0):
            result=run_pso(noisy_gaussian,num_particles, num_iterations, w=0.5, c1=1.5, c2=2.0,SNR=snr)
            #cal first iteration
            first=iteration
            j=result[5]
            for k in range(tem_iteration):
                ave=0
                for z in range(10):
                    ave_x,ave_y,ave_z=0,0,0
                    for s in range(num_particles):
                        ave_x+=j[s][k][0]
                        ave_y+=j[s][k][1]
                        ave_z+=j[s][k][2]
                    ave_x/=num_particles
                    ave_y/=num_particles
                    ave_z/=num_particles
                    ave+=noisy_gaussian(ave_x,ave_y,ave_z,SNR=100, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1)
                if ave>9:
                    first=k
                    #print('fuck',first)
                    iteration_sum+=(first+5)*num_particles
                    success-=1
                    break
        iteration_sum/=total
        y.append(iteration_sum)
    print(y)
    plt.plot(asnr,y)
    plt.title('iteration of success v.s. snr')
    plt.xlabel('snr')
    plt.ylabel('iteration of success')
    plt.show()





def draw_iteration_ini_position():
    x=[0.2,0.4,0.6,0.8,1,1.2,1.4,1.6]
    y=[]
    tem_iteration=int(1000/num_particles)
    for i in range(len(x)):
        signal=20
        total=signal
        iteration_sum=0
        while(signal>0):
            result=run_pso(noisy_gaussian, num_particles, tem_iteration, w=0.5, c1=1.5, c2=2, SNR=100)
            initial_position,iteration=result[3],result[-1]
            if x[i]==x[0]:
                pre=0
            else:
                pre=x[i-1]
            if pre<initial_position<x[i]:
                #cal first iteration
                first=iteration
                j=result[5]
                print(initial_position)
                for k in range(tem_iteration):
                    ave=0
                    for z in range(10):
                        ave_x,ave_y,ave_z=0,0,0
                        for s in range(num_particles):
                            ave_x+=j[s][k][0]
                            ave_y+=j[s][k][1]
                            ave_z+=j[s][k][2]
                        ave_x/=num_particles
                        ave_y/=num_particles
                        ave_z/=num_particles
                        ave+=noisy_gaussian(ave_x,ave_y,ave_z,SNR=100, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1)
                    if ave>9:
                        first=k
                        print('fuck',first)
                        iteration_sum+=(first+5)*num_particles
                        signal-=1
                        break
        ave_iteration=iteration_sum/total
        y.append(ave_iteration)
    print(y)
    plt.plot(x,y)
    plt.title('iteration of first success v.s. initial average position')
    plt.xlabel('initial position')
    plt.ylabel('iteration of first success')
    plt.show()
'''
dic2={}
asnr=[1,10,100,1000]
y=[]
for j in range(len(asnr)):
    hello=[]
    for i in range(1000):
        result=run_pso(noisy_gaussian,num_particles, num_iterations, w=1., c1=2.0, c2=2.0,SNR=asnr[j])
        tem_iteration=result[4]
        hello.append(tem_iteration)
    ave=sum(hello)/len(hello)
    y.append(ave)
plt.plot(asnr,y)
plt.title('iteration v.s. snr')
plt.xlabel('snr')
plt.ylabel('average iteration')
plt.show()
'''
def draw_pro_success_ini_position():
    x=[0.2,0.4,0.6,0.8,1,1.2,1.4,1.6]
    y=[]
    iteration=int(1000/num_particles)
    for i in range(len(x)):
        success=0
        total=0
        while(total<10):
            success_time=0
            result=run_pso(noisy_gaussian, num_particles, iteration, w=0.5, c1=1.5, c2=2, SNR=100)
            initial_position=result[3]
            if x[i]==x[0]:
                pre=0
            else:
                pre=x[i-1]
            if pre<initial_position<x[i]:
                print(initial_position)
                total+=1
                #cal first iteration
                first=iteration
                j=result[5]
                print('f7uck',iteration)
                for k in range(iteration):
                    ave_x,ave_y,ave_z=0,0,0
                    for s in range(num_particles):
                        ave_x+=j[s][k][0]
                        ave_y+=j[s][k][1]
                        ave_z+=j[s][k][2]
                    ave_x/=num_particles
                    ave_y/=num_particles
                    ave_z/=num_particles
                    ave=noisy_gaussian(ave_x,ave_y,ave_z,SNR=100, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1)
                    if ave>0.9:
                        first=k
                        break
                if iteration==first:
                    success_ratio=0
                else:
                    #cal the stability
                    success_time=0
                    for k in range(iteration-first):  
                        ave_x,ave_y,ave_z=0,0,0
                        ave=0
                        for s in range(num_particles):
                            j=result[5][s][k]
                            ave_x+=j[0]
                            ave_y+=j[1]
                            ave_z+=j[2]
                        ave_x/=num_particles
                        ave_y/=num_particles
                        ave_z/=num_particles
                        ave=noisy_gaussian(ave_x,ave_y,ave_z,SNR=100, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1)
                        if ave>0.9:
                            success_time+=1
                    success_ratio=success_time/(iteration-first)
                if success_ratio>0.5:
                    success+=1
                    print('hello')
        y.append(success/total)
    print(y)
    plt.plot(x,y)
    plt.title('pro of success v.s. initial average position')
    plt.xlabel('initial position')
    plt.ylabel('pro of success')
    plt.show()

def draw_intensity_iteration():
    iteration=int(1000/num_particles)
    x=[]
    for i in range(66):
        x.append(i)
    a_snr=[1,10,50,100,300, 500,600,700,800,1000]
    for i in range(len(a_snr)):
        y=[]
        result=run_pso(noisy_gaussian,num_particles, num_iterations, w=.5, c1=1.5, c2=2.0,SNR=a_snr[i])[-2]
        tem_value=0
        for k in range(iteration):
            for j in range(num_particles):
                tem_value+=result[j][k]
            tem_value/=num_particles
            y.append(tem_value)
        print(len(y))
        plt.plot(x,y,label=a_snr[i])
        plt.legend()
        
###################################################################
'''call function'''
#example()
#PSO_animation()
draw_success_pro_snr()
#draw_iteration_success_snr()
#draw_iteration_ini_position()
#draw_pro_success_ini_position()
#draw_intensity_iteration()
###################################################################
