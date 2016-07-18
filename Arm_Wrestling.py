# Authors: Matteo Greco, Stephen Bono, Nicolò Grassi, Karym El Kniba

#Set "Audio-Video" configuration

from pynq import Overlay
​
ol = Overlay("audiovideo.bit")
ol.download()

#Data acquisition from EMG sensor

%matplotlib inline
from pynq.pmods import PMOD_ADC
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from pynq.board import Button, LED
​
def trova_pos(stringa, num):
    i=0
    while (i<len(stringa)):
        if(stringa[i] == num):
            return i
        i+=1
    return -1
​
def mediaMax(Stringa):
    valoremax=max(Stringa)
    media=0
    acc=0
    count=0
    indice = trova_pos(Stringa, valoremax)
    for i in range (indice, min(indice+11, len(Stringa)), 1):
        acc += Stringa[i]
        count +=1
    for i in range (indice-1, max(indice-12, 0), -1):
        acc += Stringa[i]
        count +=1
    media=acc/count
    return media
​
# ADC configuration
pmod_adc = PMOD_ADC(1)
​
i=0
players = []
sample_freq = 0
​
# Leds&Button assignements
button = Button(0)
led1 = LED(0)
led2 = LED(1)
​
# We have 2 players -- i<2
while (i<2):
    
    # Status Led: Ready to start
    if(i==0):
        led1.on()
    else:
        led2.on()
    
    # Wait button push
    while(button.read() == 0):
        time.sleep(0.1)
    
    # Check if correct
    if(i==0):
        led1.off()
    else:
        led2.off()
​
    values = []
    sampling_time = 5
    start = time.time()
    end = start
​
    while(end - start < sampling_time):
        value = pmod_adc.read(0)
        values.append(value)
        end = time.time()
​
    num_samples = len(values)
    sampling_time = end - start
    sampling_freq = num_samples / sampling_time
    sample_freq = sampling_freq
    time_step_ms = (1 / sampling_freq) * 1000
​
    nyq_rate = sampling_freq / 2.0
    cutoff_hz = 2.0
    nsamples = len(values)
    t = np.arange(nsamples) / sampling_freq
    N = 60
​
    taps = signal.firwin(N, cutoff_hz/nyq_rate)
    filtered_x = signal.filtfilt(taps, 1.0, values)
​
    if(i==0):
        player1 = filtered_x
    if(i==1):
        player2 = filtered_x
​
    i+=1
​
# draw graph
line1, = plt.plot(np.arange(len(player1))*(1/sample_freq), player1, 'r')
line2, = plt.plot(np.arange(len(player2))*(1/sample_freq), player2, 'b')
plt.legend([line1, line2], ['Player1','Player2'])
plt.title('ARM  WRESTLING COMPARISON')
plt.ylabel("Bicept Contraction [V]")
plt.xlabel("Time [s]")
plt.ylim(0.5, 2.5)
plt.show()
​
strenght1 = mediaMax(player1)
strenght2 = mediaMax(player2)
​
endurance1 = sum(player1)/len(player1)
endurance2 = sum(player2)/len(player2)
​
print("Player1 --> strenght: " + str(strenght1))
print("\t    endurance: " + str(endurance1))
​
print("Player2 --> strenght: " + str(strenght2))
print("\t    endurance: " + str(endurance2))
​
if(strenght1>strenght2):
    value1 = strenght1
    value2 = strenght2
    v1 = 1
else:
    value1 = strenght2
    value2 = strenght1
    v1 = 2

if(value1 > (value2*1.1)):
    v1_wins = True
elif (endurance1 > endurance2):
    v1_wins = True
else:
    v1_wins = False

if(v1==1 and v1_wins == True):
    print("\nWe predict that the winner will be Player 1")
else:
    print("\nWe predict that the winner will be Player 2")


#Set the IMU sensor and connect the Monitor to have a real time feedback of the match

from time import sleep
from pynq.video import Frame, vga
from IPython.display import Image
from pynq.pmods import Grove_IMU
from IPython import display
from ipywidgets import widgets
import cv2
import numpy as np
import time

#IMU
imu = Grove_IMU(1, 4)
​
directions = ["X", "Y", "Z"]
acclBars = []
​
for d in directions:
    bar = widgets.FloatSlider(min=-2, max=2, description="Accelerometer " + d)
    acclBars.append(bar)
    display.display(bar)
​
​
# monitor configuration
video_out_res_mode = 0 # 640x480 @ 60Hz
frame_out_w = 1920
frame_out_h = 1080
​
# image configuration
frame_w = 640
frame_h = 480
​
# initialize monitor
videoOut = vga.VGA('out')
resolution = videoOut.mode(video_out_res_mode)
print("selected video resolution: " + resolution)
videoOut.start()
videoOut.frame_index(0)
​
# initialize support variables
temp = np.zeros((frame_h, frame_w), dtype=np.uint8)
​
frameOut = videoOut.frame(0)
frameCount = 0
start = time.time()
​
img = cv2.imread('image.jpg')
​
try:
    while(True):
​
    # get data from accelerometer
    motion = imu.get_motion()
        
        accl = (motion[0], motion[1], motion[2])
        
        for i in range(3):
            acclBars[i].value = accl[i]
        
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        
        shape = img.shape
        imgW = shape[1]
        imgH = shape[0]
        
        M = cv2.getRotationMatrix2D((imgW/2,imgH/2),(accl[2]*90),1)
        img_new = cv2.warpAffine(img,M,(imgW,imgH))
        
        frame[100:imgH+100,190:imgW+190] = img_new[0:imgH,0:imgW]
​
    shape = frame.shape
        frame_h = shape[0]
        frame_w = shape[1]
        
        # write frame to video output
        frame.resize(frame_w*frame_h*3)
        for y in range(frame_h):
            frameOut.frame[y*frame_out_w*3:y*frame_out_w*3+frame_w*3] = frame.data[y*frame_w*3:(y+1)*frame_w*3]

    videoOut.frame(0, frameOut)
        frameCount = frameCount + 1
​
except KeyboardInterrupt:
    # end gracefully if the kernel is interrupted
    pass
​
finally:
    videoOut.stop()