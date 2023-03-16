import pyautogui
import time
import xlrd
import pyperclip
import numpy as np
import labview_interface as lvi
import optimize as opt



def perform_commands(sheet):
  lvi.mainWork(sheet)
  # time.sleep(0.1)
  print(f"wait 0.1 seconds, On iteration {i}")

def gaussian(x, mu_x, sigma_x):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2)))
  return f

def fixed_noisy_gaussian(x, mu_x, sigma_x):
  noise = .1
  f = gaussian(x, mu_x, sigma_x)
  return f + np.random.normal(0, noise, f.shape) 





#np.random.seed(0)
step_size = .001 #let's say this is a micron
batch_size =100
n = 1000

x_init, mu_x, sigma_x = 0, 0, 1
y_init, mu_y, sigma_y = 0, 0, 1
z_init, mu_z, sigma_z = 0, 0, 1
x_range = np.arange(-2, 2, step_size)
y_range = np.arange(-2, 2, step_size)
z_range = np.arange(-2, 2, step_size)

intensity_list = [fixed_noisy_gaussian(x, mu_x, sigma_x) for x in x_range]
tau_init = 100
time_const = 8
tau_iters = 100
iters = 30
tolerance = 10**-6

learning_rate = .1
eps = 10**-3
t = 0
max_steps = 1000
wait_time = 1



file = 'intensity_values.txt'
file2 = 'cmd_v2.xls'
wb = xlrd.open_workbook(filename=file2) #open a file
sheet1 = wb.sheet_by_index(0) #accesses the first sheet
sheet2 = wb.sheet_by_index(1) #accesses the second sheet







key=input('Select algorithm: 1. Gradient ascent 2. Simulated annealing\n') 

if key =='1':
	lvi.mainWork(sheet1) #save current intensity to a file does not move piezos
	opt.noisy_gradient_ascent_v1(x_init, y_init, z_init, learning_rate, wait_time, step_size, t, sheet2, max_steps,file, tolerance)


elif key=='2':
	lvi.mainWork(sheet1) #save current intensity to a file does not move piezos
	anneal_path_x, anneal_path_y, anneal_path_z, anneal_path_intensity = opt.simulated_annealing( x_init, y_init, z_init,
														 				tau_init, x_range,y_range, z_range, step_size,
                                                         					time_const,iters, tau_iters, sheet2,file,tolerance)




