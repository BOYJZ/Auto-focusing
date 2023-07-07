import random
import numpy as np
import matplotlib.pyplot as plt

def noisy_gaussian(x, y, z, SNR=1000, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  noisy_f =  f + np.random.normal(0, 1/SNR, f.shape)  
  return noisy_f

def update_position(x, y, z , v):
    x+=v[0]
    y+=v[1]
    z+=v[2]
    return x,y,z

def fitness_function(x, y, z, SNR,num):
    f=[]
    for i in range(num):
        f.append(noisy_gaussian(x, y, z, SNR))
    #fitness = (1/5) * sum(f) - 0.1 * np.var(f)
    fitness = 1/num* sum(f)
    return fitness

def update_velocity(v, x, y, z, pbest, gbest, c1, c2, gra):
    r1 = random.random()
    r2 = random.random()
    return gra + c1*r1*(pbest-np.array([x, y, z]).T) + c2*r2*(gbest-np.array([x, y, z]).T)

#calculate the gradient part in update velocity
#inputs: pre: previous velocity ; now: velocity rn ; pre_p: previous position ; now_p: position right now
def cal_gra(pre,now,r,pre_p,now_p):
    num=len(pre)
    gra=[]
    for i in range(num):
        tem1=(-pre[i]+now[i])*r
        tem2=-pre_p[i]+now_p[i]
        ave=sum(abs(tem2))
        ave/=3
        #print(tem1,tem2,ave)
        if ave==0:
            grad=[0,0,0]
        else:
            grad=[tem1/tem2[0],tem1/tem2[1],tem1/tem2[2]]
        #print(grad)
        gra.append(grad)
    #print(pre[0],now[0],r,pre_p[0],now_p[0])
    return np.array(gra)

def run_pso(evaluate_function, center, radius, num_particles, num_iterations, learning_rate, c1, c2, SNR, robust_number):
    max_velocity = 0.2 * radius  # Max velocity is a proportion of the search radius
    neighborhood_size = 5
    # Initialize swarm
    swarm_x = np.random.uniform(center[0]-radius, center[0]+radius, num_particles)
    swarm_y = np.random.uniform(center[1]-radius,center[1]+radius, num_particles)
    swarm_z = np.random.uniform(center[2]-radius, center[2]+radius, num_particles)
    initial_position=0
    for i in range(num_particles):
        initial_position += swarm_x[i] + swarm_y[i] + swarm_z[i]
    initial_position/=num_particles
    velocities = np.array(np.zeros((num_particles, 3)))
    pbest = np.array([swarm_x, swarm_y, swarm_z]).T
    pbest_fitness = np.array([noisy_gaussian(x, y, z, SNR=SNR) for x, y, z in pbest])
    gbest_idx = np.argmax(pbest_fitness)
    gbest = pbest[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]
    best_fitnesses = [gbest_fitness] # Keep track of best fitnesses
    best_fitness_position=[gbest]
    pre_fitness=np.array([fitness_function(x, y, z, SNR,1) for x, y, z in np.array([swarm_x, swarm_y, swarm_z]).T])
    gra = np.array(np.zeros((num_particles, 3)))
    pre_p = np.array([swarm_x, swarm_y, swarm_z]).T
    history_max_position=[]
    # Run PSO
    for i in range(num_iterations):
        #w = 1 - (1 - 0.1) * (i / num_iterations)
        #c1 = 2 - (2 - 1) * (i / num_iterations)
        #c2 = 2 + (2 - 1) * (i / num_iterations)
        #num= int(10 - (10-1) * (i/num_iterations))
        num=1
        # Update velocities
        for j in range(num_particles):
            velocities[j] = update_velocity(velocities[j], swarm_x[j], swarm_y[j], swarm_z[j], pbest[j], gbest, c1, c2, gra[j])
            # Apply velocity constraints
            speed = np.linalg.norm(velocities[j])
            if speed > max_velocity:
                velocities[j] = (velocities[j] / speed) * max_velocity
        # Update positions
        for j in range(num_particles):
            swarm_x[j], swarm_y[j], swarm_z[j] = update_position(swarm_x[j], swarm_y[j], swarm_z[j], velocities[j])
        # Evaluate fitness
        current_positions = np.array([swarm_x, swarm_y, swarm_z]).T
        current_fitness = np.array([fitness_function(x, y, z, SNR,num) for x, y, z in current_positions])
        gra = cal_gra(pre_fitness,current_fitness,learning_rate,pre_p,current_positions)
        pre_p=current_positions
        pre_fitness=current_fitness
        # Update personal best and find local best
        for j in range(num_particles):
            start = max(0, j - neighborhood_size // 2)
            end = min(num_particles, start + neighborhood_size)
            local_best_idx = np.argmax(current_fitness[start:end]) + start
            local_best = current_positions[local_best_idx]
            if current_fitness[j] > pbest_fitness[j]:
                pbest[j] = current_positions[j]
                pbest_fitness[j] = current_fitness[j]
            if pbest_fitness[local_best_idx] > gbest_fitness:
                gbest = pbest[local_best_idx]
                gbest_fitness = pbest_fitness[local_best_idx]
        # Record best fitness
        best_fitnesses.append(gbest_fitness)
        best_fitness_position.append(gbest)
        # Death and reproduction
        #if len(swarm_x) >= num_particles and random.random()>mutate_prob:
            #swarm_x, swarm_y, swarm_z,velocities,pbest,pbest_fitness = death_reproduce(swarm_x, swarm_y, swarm_z,velocities,pbest,SNR,mutate_std,pbest_fitness,current_fitness)
        history_max_position.append(gbest)
        #print('added:',gbest)
        #print('history:',history_max_position)
    combined = list(zip(best_fitness_position, best_fitnesses))
    sorted_list = sorted(combined, key=lambda x: x[1])
    sorted_best_fitness_position = [t[0] for t in sorted_list]
    robust_max = np.sort(best_fitnesses)[-int(num_iterations/10):]
    arobust_max_position = sorted_best_fitness_position[len(sorted_best_fitness_position)-robust_number:]
    ave_x,ave_y,ave_z=0,0,0
    for i in range(len(arobust_max_position)):
        ave_x+=arobust_max_position[i][0]
        ave_y+=arobust_max_position[i][1]
        ave_z+=arobust_max_position[i][2]
    robust_max_position=[ave_x/len(arobust_max_position),ave_y/len(arobust_max_position),ave_z/len(arobust_max_position)]
    return initial_position, robust_max, robust_max_position,history_max_position

def calculate_success_probability(center, radius, num_particles, num_iterations, learning_rate, c1, c2, SNR_values, robust_number, num_trials=100):
    success_rates = []
    for SNR in SNR_values:
        num_successes = 0
        for _ in range(num_trials):
            initial_position, robust_max, robust_max_position, history_max_position = run_pso(noisy_gaussian, center, radius, num_particles, num_iterations, learning_rate, c1, c2, SNR, robust_number)
            distance_to_max = np.sqrt((robust_max_position[0]-center[0])**2 + (robust_max_position[1]-center[1])**2 + (robust_max_position[2]-center[2])**2)
            if distance_to_max < 0.44:
                num_successes += 1
        success_rate = num_successes / num_trials
        success_rates.append(success_rate)
        #print(f"SNR: {SNR}, Success Rate: {success_rate}")
    return SNR_values, success_rates

def tune_parameters(center, radius, num_particles_list, num_iterations_list, learning_rate_list, c1_list, c2_list, SNR_values, robust_number):
    best_params = None
    best_success_rate = 0
    least_iterations_particles = float('inf')
    
    for num_particles in num_particles_list:
        for num_iterations in num_iterations_list:
            for learning_rate in learning_rate_list:
                for c1 in c1_list:
                    for c2 in c2_list:
                        SNR_values, success_rates = calculate_success_probability(center, radius, num_particles, num_iterations, learning_rate, c1, c2, SNR_values, robust_number)
                        success_rate = np.mean(success_rates)  # average success rate over different SNRs
                        
                        # keep track of parameters yielding the highest success rate 
                        # and minimum product of number of particles and iterations
                        if success_rate >= best_success_rate and num_particles * num_iterations < least_iterations_particles:
                            best_success_rate = success_rate
                            least_iterations_particles = num_particles * num_iterations
                            best_params = (num_particles, num_iterations, learning_rate, c1, c2)
    
    return best_params

SNR_values = [1]
center = [0, 0, 0]  # center of the search space
radius = np.sqrt(3)  # radius of the search space
robust_number = 5  # number of best positions to average for the robust maximum
num_particles_list = [10, 20, 30, 40, 50]
num_iterations_list = [10, 20, 30, 40, 50]
learning_rate_list = [0.05, 0.1, 0.15, 0.2]
c1_list = [1.5, 2.0, 2.5, 3.0]
c2_list = [1.5, 2.0, 2.5, 3.0]

best_params = tune_parameters(center, radius, num_particles_list, num_iterations_list, learning_rate_list, c1_list, c2_list, SNR_values, robust_number)

print("Best parameters: ")
print("Number of particles: ", best_params[0])
print("Number of iterations: ", best_params[1])
print("Learning rate: ", best_params[2])
print("C1: ", best_params[3])
print("C2: ", best_params[4])
