import random
import numpy as np
import matplotlib.pyplot as plt

def noisy_gaussian(x, y, z, SNR=1000, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  noisy_f =  f + np.random.normal(0, 1/SNR, f.shape)  
  return noisy_f

def update_velocity(v, x, y, z, pbest, gbest, w, c1, c2):
    r1 = random.random()
    r2 = random.random()
    return w*v + c1*r1*(pbest-np.array([x, y, z]).T) + c2*r2*(gbest-np.array([x, y, z]).T)


def update_position(x, y, z , v):
    x+=v[0]
    y+=v[1]
    z+=v[2]
    return x,y,z

def generate_child(parent1, parent2, mutation_rate):
    # Randomly choose a crossover point
    crossover_point=int(random.random()*3)
    crossover_point=0
    # Create the child by combining parent1 and parent2 up to the crossover point
    child = parent1[:crossover_point] + parent2[crossover_point:]
    # Apply mutation to the child with the given mutation rate
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] = random.uniform(-np.sqrt(3), np.sqrt(3))
    return child

def death_reproduce(swarm_x, swarm_y, swarm_z,velocities,pbest,snr,mutation_rate,pbest_fitness,fitness_values):
    fitness_values=np.array(fitness_values)
    # Determine the positions of the two particles with the highest fitness values
    max1_pos = fitness_values.argmax()
    max2_pos = np.argsort(fitness_values)[-2]
    
    # Generate two child particles and add them to the swarm
    child1_x, child1_y, child1_z = generate_child([swarm_x[max1_pos], swarm_y[max1_pos], swarm_z[max1_pos]],
                                                   [swarm_x[max2_pos], swarm_y[max2_pos], swarm_z[max2_pos]],mutation_rate)
    swarm_x = np.append(swarm_x, [child1_x])
    swarm_y = np.append(swarm_y, [child1_y])
    swarm_z = np.append(swarm_z, [child1_z])
    
    child2_x, child2_y, child2_z = generate_child([swarm_x[max1_pos], swarm_y[max1_pos], swarm_z[max1_pos]],
                                                   [swarm_x[max2_pos], swarm_y[max2_pos], swarm_z[max2_pos]],mutation_rate)
    swarm_x = np.append(swarm_x, [child2_x])
    swarm_y = np.append(swarm_y, [child2_y])
    swarm_z = np.append(swarm_z, [child2_z])
    
    #velocities
    velocity1 = generate_child([velocities[max1_pos][0],velocities[max1_pos][1],velocities[max1_pos][2]],
                                [velocities[max2_pos][0],velocities[max2_pos][1],velocities[max2_pos][2]],mutation_rate)
    velocities = np.concatenate((velocities, [velocity1]),axis=0)


    velocity2 = generate_child([velocities[max1_pos][0],velocities[max1_pos][1],velocities[max1_pos][2]],
                                [velocities[max2_pos][0],velocities[max2_pos][1],velocities[max2_pos][2]],mutation_rate)
    velocities = np.concatenate((velocities, [velocity2]),axis=0)


    #random velocities
    #a=np.array([velocities])
    #b=np.array([random.random(), random.random(), random.random()])
    #new_velocities = np.array([a,b])
    #velocities = np.concatenate((velocities, new_velocities))
    
    a=np.array([child1_x, child1_y, child1_z])
    b=np.array([child2_x, child2_y, child2_z])
    new_pbest = np.array([a,b])
    pbest = np.concatenate((pbest,new_pbest))
    
    a=noisy_gaussian(child1_x, child1_y, child1_z, SNR=snr, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1)
    b=noisy_gaussian(child1_x, child1_y, child1_z, SNR=snr, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1)
    new_fitness=np.array([a,b])
    pbest_fitness = np.concatenate((pbest_fitness,new_fitness))

    return swarm_x, swarm_y, swarm_z,velocities,pbest,pbest_fitness

def fitness_function(x, y, z, SNR,num):
    f=[]
    for i in range(num):
        f.append(noisy_gaussian(x, y, z, SNR))
    #fitness = (1/5) * sum(f) - 0.1 * np.var(f)
    fitness = 1/num* sum(f)
    return fitness



def run_pso(evaluate_function, center, radius, num_particles, num_iterations, w, c1, c2, SNR, mutate_prob, mutate_std,robust_number):
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
    
    # Run PSO
    for i in range(num_iterations):
        #w = 1 - (1 - 0.1) * (i / num_iterations)
        #c1 = 2 - (2 - 1) * (i / num_iterations)
        #c2 = 2 + (2 - 1) * (i / num_iterations)
        #num= int(10 - (10-1) * (i/num_iterations))
        num=1
        # Update velocities
        velocities = update_velocity(velocities, swarm_x, swarm_y, swarm_z, pbest, gbest, w, c1, c2)
        # Update positions
        for j in range(num_particles):
            swarm_x[j], swarm_y[j], swarm_z[j] = update_position(swarm_x[j], swarm_y[j], swarm_z[j], velocities[j])
        # Evaluate fitness
        current_positions = np.array([swarm_x, swarm_y, swarm_z]).T
        current_fitness = np.array([fitness_function(x, y, z, SNR,num) for x, y, z in current_positions])
        # Update personal best
        pbest_mask = current_fitness > pbest_fitness
        pbest[pbest_mask] = current_positions[pbest_mask]
        pbest_fitness[pbest_mask] = current_fitness[pbest_mask]
        # Update global best
        if np.max(current_fitness[:num_particles]) > gbest_fitness:
            gbest_idx = np.argmax(current_fitness[:num_particles])
            gbest = current_positions[gbest_idx]
            gbest_fitness = current_fitness[gbest_idx]
        # Record best fitness
        best_fitnesses.append(gbest_fitness)
        best_fitness_position.append(gbest)
        # Death and reproduction
        if len(swarm_x) >= num_particles and random.random()>mutate_prob:
            swarm_x, swarm_y, swarm_z,velocities,pbest,pbest_fitness = death_reproduce(swarm_x, swarm_y, swarm_z,velocities,pbest,SNR,mutate_std,pbest_fitness,current_fitness)
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
    return initial_position, robust_max, robust_max_position


result = run_pso(noisy_gaussian,center=[0,0,0],radius=np.sqrt(3), num_particles=50, num_iterations=100, w=0.7, c1=1.5, c2=1.5, SNR=10, mutate_prob=0.1, mutate_std=0.1,robust_number=3)
print(result[2])



def draw_success_pro_snr():
    iteration=100
    #asnr=[0]
    asnr=[1,10,20,30,50]
    y=[]
    for snr in asnr:
        print(snr)
        success=0
        time=100
        for i in range(time):
            success_time=0
            result=run_pso(noisy_gaussian,center=[0,0,0],radius=np.sqrt(3), num_particles=20, num_iterations=iteration,  w=0.7, c1=2, c2=1, SNR=snr, mutate_prob=0.5, mutate_std=0.01,robust_number=6)
            #cal first iteration
            first=iteration
            gbest_position=result[2]
            for k in range(iteration):
                distance=np.sqrt(gbest_position[0]**2+gbest_position[1]**2+gbest_position[2]**2)
                if distance< 0.225:
                    first=k
                    break
            if iteration==first:
                success_ratio=0
            else:
                #cal the stability
                success_time=0
                for k in range(iteration-first):  
                    distance=np.sqrt(gbest_position[0]**2+gbest_position[1]**2+gbest_position[2]**2)
                    if distance< 0.225:
                        success_time+=1
                success_ratio=success_time/(iteration-first)
            if success_ratio>0.5:
                success+=1
        y.append(success/time)
    #k=[0.023, 0.601, 0.881]
    print('PSO',y)
    plt.plot(asnr, y, marker='o',color='b')
    #plt.plot(asnr, k, marker='o',color='r')
    plt.title('pro of sucess v.s. snr')
    plt.xlabel('snr')
    plt.ylabel('pro of sucess')
    plt.show()


def optimise_parameter():
    iteration=10
    y=[]
    ap=[]
    for r in range(10):
        p=r
        print(p)
        ap.append(p)
        success=0
        time=100
        for i in range(time):
            success_time=0
            result=run_pso(noisy_gaussian, num_particles=20, num_iterations=iteration, w=0.7, c1=2, c2=1, SNR=10, mutate_prob=0.5, mutate_std=0.5,robust_number=6)
            #cal first iteration
            first=iteration
            gbest_position=result[2]
            for k in range(iteration):
                distance=np.sqrt(gbest_position[0]**2+gbest_position[1]**2+gbest_position[2]**2)
                if distance< 0.225:
                    first=k
                    break
            if iteration==first:
                success_ratio=0
            else:
                #cal the stability
                success_time=0
                for k in range(iteration-first):  
                    distance=np.sqrt(gbest_position[0]**2+gbest_position[1]**2+gbest_position[2]**2)
                    if distance< 0.225:
                        success_time+=1
                success_ratio=success_time/(iteration-first)
            if success_ratio>0.5:
                success+=1
        y.append(success/time)
    
    print('PSO',y)
    plt.plot(ap, y, marker='o',color='b')
    plt.show()
    
draw_success_pro_snr()
#optimise_parameter()    
    
    
    
    
    
    