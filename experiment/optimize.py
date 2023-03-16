import random 
import time
import os 
import labview_interface as lvi


def open_file(file):
  '''
  opens file and seeks to the 2nd to the last value
    in the file
  '''
  num_newlines = 0
  n=1 #represents 2nd to last 
  with open(file, 'rb') as f:
      try:  # catch OSError in case of a one line file 
        f.seek(-2, os.SEEK_END)
        while num_newlines < n:
            f.seek(-2, os.SEEK_CUR)
            if f.read(1) == b'\n':
                num_newlines += 1
      except OSError:
        f.seek(0)
      second_last_line = f.readline().decode()
  return float(second_last_line[0:-1] )#gets rid of \n newline



def perform_commands(sheet, delta_x,delta_y, delta_z, step_size):
  lvi.mainWork(sheet, delta_x,delta_y, delta_z, step_size)
  # time.sleep(0.1)
  


def undo_move(sheet, delta_x,delta_y, delta_z, step_size): # may be a problem bc of piezo drift!
    lvi.mainWork(sheet, -delta_x,-delta_y, -delta_z, step_size)
  


def walker(sheet, intensity_init, pos_init, pos_new, delta_x, delta_y, delta_z,\
                                                                step_size, tau, file):
  intensity_new = open_file(file) # reads new intensity from file
  delta_f = intensity_new - intensity_init 

  if delta_f >= 0:  # if delta_f is positive then accept new position
        pos_init = pos_new
  else:   # will move downhill with some probability 
        acceptance_prob = min(1,np.exp(-abs(delta_f)/tau)) # as tau changes so will the prob
        if acceptance_prob > random.uniform(0,1):
            pos_init = pos_new
        else:
            undo_move(sheet,delta_x,delta_y, delta_z, step_size)         
  return pos_init    
                      


def simulated_annealing( x_init, y_init, z_init, tau_init, x_range,y_range, z_range, step_size,
                                                          time_const,iters, tau_iters, sheet2,file ,tolerance):
  '''
 *  Will randomly choose a proposed position
 * if the f(x_proposed) is higher than the previous then accept that new position
 * else accept the new position with some probabilty ~exp(-1/tau) that changes 
     when the temperature parameter tau changes
    * the tau schedule values will be large in the beginning and decrease over time
      -which means that the algorithm will not always 
        seek the max in the beginning but
        will towards the end of the schedule
  returns two lists 
    
  
  '''

  anneal_path_x = []
  anneal_path_y = []
  anneal_path_z = []
  anneal_path_intensity = []
  len_x = len(x_range)
  len_y = len(y_range)
  len_z = len(z_range)

  tau_schedule = [tau_init * np.exp(-t/time_const) for t \
                in range(0,tau_iters)]
  #np.random.seed(0) 
  reTry = 1 # at some point get rid of this requirement
  intensity_init= open_file(file) #initial intensity

  
  for tau in tau_schedule:
    for steps in range(0, iters):
      x_new = x_range[random.randint(0, len_x-1)] #randomly choose x in a range of value
      y_new = y_range[random.randint(0, len_y-1)] #randomly choose y in a range of value
      z_new = z_range[random.randint(0, len_z-1)] #randomly choose z in a range of value

      delta_x = x_new - x_init
      delta_y = y_new - y_init
      delta_z = z_new - z_init

      perform_commands(sheet2, delta_x, 0 ,0, step_size) # will move piezos in xyz by new steps
      x_init = walker(sheet2, intensity_init, x_init, x_new, delta_x, 0, 0, step_size, tau, file)
      intensity_init = open_file(file) #necessary because of possible drift (instead of saying init = new)

      perform_commands(sheet2, 0,delta_y, 0, step_size) # will move piezos in xyz by new steps
      y_init = walker(sheet2, intensity_init, y_init, y_new, 0, delta_y, 0, step_size, tau, file)
      intensity_init = open_file(file) #necessary because of possible drift (instead of saying init = new)

      perform_commands(sheet2, 0, 0, delta_z, step_size) # will move piezos in xyz by new steps   
      z_init = walker(sheet2, intensity_init, z_init, z_new, 0, 0, delta_z, step_size, tau, file)
      intensity_init = open_file(file) #necessary because of possible drift (instead of saying init = new)

    anneal_path_x.append(x_init)
    anneal_path_y.append(y_init)
    anneal_path_z.append(z_init)
    anneal_path_intensity.append(intensity_init)
    # if abs(delta_f) < tolerance and  steps > iters-2:
    #   break
  
  return anneal_path_x, anneal_path_y, anneal_path_z, anneal_path_intensity


'''
 gradiend ascent code
'''

def grad_calc(sheet, delta_x_plus,delta_y_plus,delta_z_plus, delta_x_minus,delta_y_minus, delta_z_minus,file, step_size):
  
  perform_commands(sheet, delta_x_plus,delta_y_plus,delta_z_plus, step_size)
  intensity_plus= open_file(file) 
  undo_move(sheet,delta_x_plus,delta_y_plus,delta_z_plus, step_size)  

  perform_commands(sheet,delta_x_minus,delta_y_minus, delta_z_minus, step_size)
  intensity_minus= open_file(file) 
  undo_move(sheet,delta_x_minus,delta_y_minus, delta_z_minus, step_size)

  grad_pos = (intensity_plus - intensity_minus)/(2*step_size)  # (f(x+h) - f(x-h))/2h
  return grad_pos


def grad_step(sheet, x_init, y_init, z_init, grad_x, grad_y, grad_z, learning_rate, step_size):

  perform_commands(sheet, learning_rate*grad_x, learning_rate*grad_y, learning_rate*grad_z, step_size)
  x_new = x_init + learning_rate*grad_x
  y_new = y_init + learning_rate*grad_y
  z_new = z_init + learning_rate*grad_z

  return x_new, y_new,z_new

def noisy_gradient_ascent_v1(x_init, y_init, z_init, learning_rate, wait_time, step_size, t, sheet, max_steps,file, tolerance):
  
  path_x = [x_init]
  path_y = [y_init]
  path_z = [z_init]

  while t < max_steps:

    grad_x, grad_y, grad_z = 0, 0, 0
    reTry = 1 # at some point get rid of this requirement


    for i in range(wait_time):

      delta_x_plus = x_init + step_size
      delta_y_plus = y_init + step_size
      delta_z_plus =  z_init + step_size
      delta_x_minus = x_init - step_size
      delta_y_minus = y_init - step_size
      delta_z_minus =  z_init - step_size
      
      grad_x += grad_calc(sheet, delta_x_plus,0,0, delta_x_minus,0, 0, file, step_size) 
      grad_y += grad_calc(sheet, 0,delta_y_plus,0, 0,delta_y_minus, 0, file, step_size) 
      grad_z += grad_calc(sheet, 0,0,delta_z_plus, 0,0, delta_z_minus, file, step_size) 
      t += 1


    grad_x, grad_y, grad_z = grad_x/wait_time, grad_y/wait_time, grad_z/wait_time #averaged grads
    x_init, y_init, z_init = grad_step(sheet, x_init, y_init, z_init, grad_x, grad_y, grad_z, learning_rate, step_size)
    path_x.append(x_init)
    path_y.append(y_init)
    path_z.append(z_init)

    if abs(path_x[-1] - path_x[-2]) < tolerance and  abs(path_y[-1] - path_y[-2]) < tolerance   \
                                                and abs(path_z[-1] - path_z[-2]) < tolerance  and t > int(max_steps/2):
      break




  return path_x,path_y,path_z, t







