import cv2 
import numpy as np
import time
from PIL import ImageDraw, ImageFont, Image
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import keras
sys.stderr = stderr
from keras.models import load_model

from matplotlib import pyplot

modelname = h5py.File('../my_model.h5','r+')
model = load_model(modelname)

def get_expression(custom):
    objects = ['rabbia', 'paura', 'felicità', 'tristezza', 'sorpresa', 'normale']
    m=0.000000000000000000001
    a=custom[0]
    for i in range(0,len(a)):
        if a[i]>m:
            m=a[i]
            ind=i
    return objects[ind]

def get_img(original):
    frame = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)#scambio i rossi con i blu perchè erano invertiti quindi porto da BGR in RGB
    '''pyplot.imshow(frame)
    pyplot.show()'''
    #x:48/640=0.075 y:48/480=0.1
    
    frame = cv2.resize(frame, (0,0), fx=0.097, fy=0.1)
    frame = frame[0:frame.shape[1], 7:frame.shape[1]-7]#taglio 7 pixel a sx e 7 a dx
    '''pyplot.imshow(frame)
    pyplot.show()'''
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    '''pyplot.imshow(frame)
    pyplot.show()'''
    frame = np.expand_dims(frame, axis = -1)
    frame = np.expand_dims(frame, axis = 0)
    frame = frame.astype("float32")
    frame /= 255
    return frame

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)

# Write some Text
fontpath = "./Ubuntu-R.ttf"
font1                   = ImageFont.truetype(fontpath, 32) #cv2.FONT_HERSHEY_SIMPLEX
font2                   = ImageFont.truetype(fontpath, 16) 
fontScale              = 1
fontColor              = (255,255,255,0)
lineType               = 2
expression=""
while True:
    try:
        check, frame = webcam.read()
        # get boundary of this text
        
        #textsize = cv2.getTextSize(expression, font, 1, 2)[0]
        frame = cv2.rectangle(frame, (0,480-40), (640,480), (0,0,0),-1)
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        draw.text((20,480-40),expression, font=font1, fill= fontColor)
        draw.text((500,480-30), "Premi Q per uscire", font=font2, fill= fontColor)
        frame = np.array(frame_pil)
        cv2.imshow("Capturing", frame)
        i, d = divmod(time.time(), 1)
        
        if(d>0.9):
            x=get_img(frame)
            custom = model.predict(x)
            #print("Espressione: ",get_expression(custom), " (q per uscire)", end="\r", flush=True)
            expression=get_expression(custom)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
      
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break