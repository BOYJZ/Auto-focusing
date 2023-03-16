import numpy as np
import random 
import time
import os 
import labview_interface as lvi


def _open_file(file):
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
  print('in file second to last value', second_last_line)
  return float(second_last_line[0:-1]) #gets rid of \n newline

# def _open_file(file):
#   '''
#   opens file and seeks to the 2nd to the last value
#     in the file
#   '''
#   with open(file) as f:
#     my_lines = f.read().splitlines()
#     last_value = my_lines[-1]
#     #print('last value', my_lines)
#   return float(last_value)



def _choose_dim(delta_pos,label):

  if label == 'x':
    delta_x = delta_pos
    delta_y = 0
    delta_z = 0

  elif label == 'y':
    delta_x = 0
    delta_y = delta_pos
    delta_z = 0

  elif label == 'z':
    delta_x = 0
    delta_y = 0
    delta_z = delta_pos
  else:
    print('did not write label in grad_calc')
    exit()
  return delta_x,delta_y,delta_z




def _perform_commands(sheet, delta_x=None,delta_y=None, delta_z=None, step_size=None):
  '''
  follows instructions as dictated by excel sheet
  '''
  
  if step_size:
    lvi.mainWork(sheet,  delta_x,delta_y, delta_z, step_size)
  else:
    lvi.mainWork(sheet )
  # time.sleep(0.1)
  return


def _undo_move(sheet, delta_x,delta_y, delta_z, step_size): # may be a problem bc of piezo drift!
  lvi.mainWork(sheet, -delta_x,-delta_y, -delta_z, step_size)
  return


def _walker(sheet1, sheet2, sheet3,  intensity_init,  pos_init, pos_new, step_size, tau, wait_time, file, label):

  delta_pos = pos_new - pos_init #will dictate how much piezo will move
  delta_x,delta_y,delta_z =_choose_dim(delta_pos,label)

  for _ in range( wait_time): # for time avg intensity
    _perform_commands(sheet1 ) #saves intensity to file
    intensity_init += _open_file(file) #initial intensity
    time.sleep(0.5)

  _perform_commands(sheet3, delta_x, delta_y, delta_z, step_size) # will move piezos in x or y or z 

  for _ in range( wait_time): # used for time avg intensity
    _perform_commands(sheet1 ) #saves intensity to file
    intensity_new += _open_file(file) 
    time.sleep(0.5)

  delta_f = (intensity_new - intensity_init)/wait_time  #time average of delta intensity

  if delta_f >= 0:  # if delta_f is positive then accept new position
        pos_init = pos_new
  else:   # will move downhill with some probability 
        acceptance_prob = min(1,np.exp(-abs(delta_f)/tau)) # as tau changes so will the prob
        if acceptance_prob > random.uniform(0,1):
            pos_init = pos_new # accepts a non greedy movement
        else:
            _undo_move(sheet3, delta_x,delta_y, delta_z, step_size) # rejects movement and goes back to original pos

  return pos_init,delta_f    
                      


def simulated_annealing( sheet1, sheet2, sheet3, pos_init, pos_range, tau_init, step_size, wait_time, time_const,iters, 
                                                          tau_iters,file, tolerance, label) :
  '''
 *  Will randomly choose a proposed position
 * if the f(x_proposed) is higher than the previous then accept that new position
 * else accept the new position with some probabilty ~exp(-delta/tau) that changes 
     when the temperature parameter tau changes
    * the tau schedule values will be large in the beginning and decrease over time
      -which means that the algorithm will not always seek the max in the beginning but
        will towards the end of the schedule
  returns one lists
    
  
  '''

  anneal_path = []
  anneal_path_intensity = []
  len_pos = len(pos_range)

  intensity_init = 0
  intensity_new = 0

  tau_schedule = [tau_init * np.exp(-t/time_const) for t \
                                    in range(0,tau_iters)]
  
  for tau in tau_schedule:
    for steps in range(0, iters):
      pos_new = pos_range[random.randint(0, len_pos-1)] #randomly choose x in a range of value
      
      pos_init, delta_f  = _walker(sheet1, sheet2, sheet3, intensity_init, pos_init, pos_new, 
                                                      step_size, tau, wait_time, file, label)
      intensity_init = _open_file(file) #necessary because of possible drift (instead of saying init = new)

    anneal_path.append(pos_init)
    anneal_path_intensity.append(intensity_init)

    if abs(delta_f) < tolerance and  steps > iters-2:
      break
  
  return anneal_path


'''
 gradiend ascent code--------------------------------------------------------------------------------------
'''


def _grad_calc(sheet1,sheet2, sheet3, delta_pos, wait_time, file, step_size, label):

  '''
  Calculates the gradient by averging intensity over n number of seconds and 
    finding the derivative

    the first for loop sums up n seconds of intensity readings
    we move by the smallest step size
    the second for loop sums up n seconds of intensity
    then the derivative is calculates and multipled by 1/n
  returns average gradient (float)
  '''


  intensity_1 = 0
  intensity_2 = 0
  delta_x,delta_y,delta_z = _choose_dim(delta_pos,label)

  for _ in range(wait_time):
    _perform_commands(sheet1 ) #saves intensity to file
    intensity_2 += _open_file(file)
    print('intensity_2', intensity_2)
   # time.sleep(1.0) 

  _perform_commands(sheet3,  delta_x,delta_y,delta_z, step_size) #makes a move 

  for _ in range(wait_time):
    _perform_commands(sheet1 ) #saves intensity to file
    intensity_1 += _open_file(file)
    print('intensity_1',intensity_1)
    #time.sleep(1.0) #waits 1 second

  _undo_move(sheet3, delta_x,delta_y,delta_z, step_size) # go back to original position 

  grad_avg = (intensity_1 - intensity_2)/(wait_time * delta_pos) # avg of (f(x+h) - f(x))/h
  print('grad_avg')
  return grad_avg

  

def _grad_step(sheet1,sheet2, sheet3, pos_init, delta_pos, wait_time, learning_rate, file, step_size, label):

  '''
  Makes a step towards the max
  '''

  grad_avg = _grad_calc(sheet1,sheet2, sheet3, delta_pos, wait_time, file, step_size, label) #calc avg grad
  if grad_avg:
    delta_x,delta_y,delta_z = _choose_dim(learning_rate*grad_avg,label) #chooses the dim to be optimized
    print('learning_rate*grad_avg',learning_rate*grad_avg)
    _perform_commands(sheet2,  delta_x,delta_y,delta_z, step_size) #move and save to file
    pos_new = pos_init + learning_rate*grad_avg # position update
    print('new position ', pos_new)
  else:
    print('did not move, there is a problem with inputs to _grad_step()  it is none' )
    exit()

  return pos_new

def noisy_gradient_ascent_v2( sheet1, sheet2,   sheet3, pos_init, learning_rate, wait_time,  max_steps, tolerance, file, step_size,label):
  '''
  sheet1 sheet2: excel sheets from cmd_v2
  pos_init: init position float  
  learning_rate: will dictate how fast gradient converges (float)
  wait_time: how many seconds of intensity avg (int)
  max_steps: number of max steps in grad ascent until it stops (int) 
  tolerance: tells grad when to stop (float)
  file: the file where intensity is being stored (string)
  step_size: the smallest step piezo can take (float)
  label: will tell whether grad ascent is wrt x or y or z (string)
  

  function does time averaged gradient ascent
  will stop if intensity changes less than the tolerance
  returns the path and the number of steps to reach convergence

  '''

  grad_path = [pos_init]
  for t in range( max_steps):
    delta_pos = 5*step_size # used to make a small change in intensity for the derivate calc
    pos_init = _grad_step(sheet1,sheet2,sheet3, pos_init, delta_pos, wait_time, learning_rate, file, step_size, label)
    grad_path.append(pos_init)
    print('grad_path', grad_path)

    if abs(grad_path[-1] - grad_path[-2]) < tolerance  and t > 2:
      break

  return grad_path, t










