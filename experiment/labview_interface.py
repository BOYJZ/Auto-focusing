import pyautogui
import time
import xlrd
import pyperclip
import numpy as np



#define mouse events

#Other uses of the pyautogui library https://blog.csdn.net/qingfengxd1/article/details/108270159
'''
    clickTimes = number (type:int) of times you want it to click. This could be 1 single click or 2 for double
    lOrR = stands for left or right clicks (type string)
    img = image name (type string)
    reTry = number of retries (type int)

    * Locates the center of the image (obtained from img file) on the current screen
    * for reTry  equal to 1
        -if image is found then click the center
        -if the image is not found it will delay for .1 seconds and try again infinite amount of times
    * if reTry is greater than 1
        -then the function will repeat the operation reTry-number of times

    note: images must be in the same directory as program
    
'''
def mouseClick(clickTimes,lOrR,img,reTry): #mouseClick(1,"left",img,reTry)

    if reTry == 1:
        while True:
            location=pyautogui.locateCenterOnScreen(img,confidence=0.8)
            current_location =  pyautogui.position()
            if current_location.x > 3600:
                print('mouse out of bounds shutting off...') # emergency shut off
                exit()

            elif location is not None:
                pyautogui.click(location.x+10,location.y,clicks=clickTimes,interval=0.2,duration=0.2,button=lOrR)
                break
            
            print("No matching images found,Retry after 0.1 seconds")
            time.sleep(.1)
    elif reTry > 1: 
        i = 1
        while i < reTry + 1:
            location=pyautogui.locateCenterOnScreen(img,confidence=0.8)
            current_location =  pyautogui.position()
            if current_location.x > 3600:
                print('mouse out of bounds shutting off...') # emergency shut off
                exit()
            elif location is not None:
                pyautogui.click(location.x+10,location.y,clicks=clickTimes,interval=0.2,duration=0.2,button=lOrR)
                #print("repeat click")
                time.sleep(.1)
                i += 1
     
    elif reTry == 0:
        pass
        time.sleep(.2)
    else:
        print('Error with retry')
        exit()


def mainWork(sheet, x_step=None,y_step=None,z_step=None,step_size=None):
    reTry = 1 
    i = 1
    while i < sheet.nrows:
        
        cmdType = sheet.row(i)[0] #save data from first col  ith row

        if cmdType.value == 1.0:
            img = sheet.row(i)[1].value #take picture name from second col ith row                 
            mouseClick(1,"left",img,reTry) #left clicks the image on the screen
            print("left click",img)
            
        
        elif cmdType.value == 2.0:   #2 means double-click the left button
            img = sheet.row(i)[1].value #take picture name in second col            
            mouseClick(2,"left",img,reTry) #left clicks the image on the scree
            print("Double left button",img)
        
        elif cmdType.value == 4.0: #4 stands for enter
            inputValue = sheet.row(i)[1].value
            pyperclip.copy(inputValue) # allows you to copy inputValue to clipboard
            pyautogui.hotkey('ctrl','v')# pastes information from the clipboard 
            time.sleep(0.5)
            pyautogui.press('enter')
            print("input:",inputValue)
                                                  

        elif cmdType.value == 8.0: # only makes piezo movements
            if x_step and step_size: 
                x_button = 1 if x_step <= 0 else 2   # 1 is west 2 is east
                #x_reTry = abs(int(x_step/step_size)) # if x_button < step_size then reTry = 0
                x_reTry = abs(int(x_step/step_size)) if abs(int(x_step/step_size)) < 10 else 10
                x_img = f'direction{x_button}.png'  #string which will be a file name 
                print('number of button presses in the '+str(x_button)+' direction', x_reTry)
                mouseClick(1,"left",x_img,x_reTry) #will move piezo in x directions (x,y,z)

            elif y_step and step_size:
                y_button = 3 if y_step <= 0 else 4 # 3 is south 4 is north
                #y_reTry = abs(int(y_step/step_size))
                y_reTry = abs(int(y_step/step_size)) if abs(int(y_step/step_size)) < 10 else 10
                y_img = f'direction{y_button}.png' 
                print('number of button presses in the '+str(y_button)+' direction', y_reTry)
                mouseClick(1,"left",y_img,y_reTry)
             


            elif z_step and step_size:
                z_button = 6 if z_step <= 0 else 5 #6 is going up(in) 5 is going down (out)
						#odd is positive direction even is neg 	
                #z_reTry = abs(int(z_step/step_size)) 
                z_reTry = abs(int(z_step/step_size)) if abs(int(z_step/step_size)) < 10 else 10
                z_img = f'direction{z_button}.png' 
                print('number of button presses in the '+str(z_button)+' direction', z_reTry)
                mouseClick(1,"left",z_img,z_reTry)
            


        elif cmdType.value == 9.0:
            #mouseClick(2,"left",img,reTry)
            pyautogui.doubleClick(450, 720, button='left')  #these cordinates are where we copy
                                                            #in order to record the PDM! trace counts
                                                           #to the text file on the right hand side of the screen.
                                                           # goes to these specific coordinates and double clicks

            pyautogui.hotkey('ctrl','c') #copies to clipboard
            time.sleep(.2)
            print("copied")  
            
        elif cmdType.value == 10.0:
            pyautogui.hotkey('ctrl','v') #pastes from the clipboard
            pyautogui.press('enter')
            print("pasted")    
           

        elif cmdType.value == 11.0:
            pyautogui.hotkey('ctrl','s') #saves file
            print("saved") 
                                    
        i += 1




