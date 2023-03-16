import pyautogui
import time
import xlrd
import pyperclip
import numpy as np
import labview_interface as lvi
import optimize_v2 as opt





#n = 1000

x_init, y_init, z_init = 0, 0, 0
step_size = .020 #this is 20 nm
x_range = np.arange(-2, 2, step_size)
y_range = np.arange(-2, 2, step_size)
z_range = np.arange(-2, 2, step_size)

tau_init = 100
time_const = 8
tau_iters = 100
iters = 30
tolerance = 5*10**3
optimeize_iter = 2

learning_rate = 5*10**-7
eps = 10**-3

max_steps = 1000
wait_time = 1

file = 'intensity_values.txt'
file2 = 'cmd_v2.xls'
wb = xlrd.open_workbook(filename=file2) #open a file
sheet1 = wb.sheet_by_index(0) #accesses the first sheet
sheet2 = wb.sheet_by_index(1) #accesses the second sheet
sheet3 = wb.sheet_by_index(2) #accesses the second sheet

key=input('Select algorithm: 1. Gradient ascent 2. Simulated annealing\n') 
path_tot_x = []
path_tot_y = []
path_tot_z = []

'''
Will perform simulated annealing or gradient descent with respect to z then y then x

repeats the procedure again n number of times using the final position value of 
  previous run as the starting position of next run.
  
stores the trajectory as a list for each dimension

'''


if key =='1':

  #lvi.mainWork(sheet1) #save current intensity to a file does not move piezos
  for i in range(optimeize_iter): #will optimize z 'optimize_iter' number of times
    path_z, tz = opt.noisy_gradient_ascent_v2( sheet1, sheet2,sheet3, z_init, learning_rate,
                               wait_time, max_steps, tolerance, file, step_size, 'z')
    z_init = path_z[-1] 
    path_tot_z += path_z

    if i < 1:    
      path_y, ty = opt.noisy_gradient_ascent_v2( sheet1, sheet2, sheet3, y_init, learning_rate, 
                                  wait_time, max_steps, tolerance, file, step_size, 'y')
      y_init = path_y[-1] 
      path_tot_y += path_y


      path_x, tx = opt.noisy_gradient_ascent_v2( sheet1, sheet2,sheet3, x_init, learning_rate,
                                wait_time, max_steps, tolerance, file, step_size, 'x')
      x_init = path_x[-1] 
      path_tot_x += path_x

elif key=='2':

  lvi.mainWork(sheet1) #save current intensity to a file does not move piezos
  for i in range(optimeize_iter): #will optimize z 'optimize_iter' number of times
    anneal_path_z  = opt.simulated_annealing( sheet1, sheet2, sheet3, z_init, z_range, tau_init,
                            step_size, wait_time ,time_const,iters,tau_iters,file, tolerance, 'z')
    z_init = anneal_path_z[-1] 
    path_tot_z += anneal_path_z

    if i < 1:
      anneal_path_y = opt.simulated_annealing( sheet1, sheet2, sheet3, y_init, y_range, tau_init,
                              step_size, wait_time, time_const,iters,tau_iters,file, tolerance, 'y')
      y_init = anneal_path_y[-1] 
      path_tot_y += anneal_path_y

      anneal_path_x = opt.simulated_annealing( sheet1, sheet2, sheet3, x_init, x_range, tau_init,
                              step_size, wait_time, time_const,iters,tau_iters,file, tolerance, 'x')
      x_init = anneal_path_x[-1] 
      path_tot_x += anneal_path_x
      
      
elif key=='3':
    
    lvi.mainWork(sheet1) #save current intensity to a file does not move piezos
    for i in range(optimeize_iter): #will optimize z 'optimize_iter' number of times
      anneal_path_z  = opt.pso( sheet1, sheet2, sheet3, z_init, z_range, tau_init,
                              step_size, wait_time ,time_const,iters,tau_iters,file, tolerance, 'z')
      z_init = anneal_path_z[-1] 
      path_tot_z += anneal_path_z

      if i < 1:
        anneal_path_y = opt.pso( sheet1, sheet2, sheet3, y_init, y_range, tau_init,
                                step_size, wait_time, time_const,iters,tau_iters,file, tolerance, 'y')
        y_init = anneal_path_y[-1] 
        path_tot_y += anneal_path_y

        anneal_path_x = opt.pso( sheet1, sheet2, sheet3, x_init, x_range, tau_init,
                                step_size, wait_time, time_const,iters,tau_iters,file, tolerance, 'x')
        x_init = anneal_path_x[-1] 
        path_tot_x += anneal_path_x


