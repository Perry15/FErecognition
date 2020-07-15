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
from tensorflow.keras.models import load_model
sys.stderr = stderr
from matplotlib import pyplot

'''modelname = h5py.File('../my_model.h5','r+')
model = load_model(modelname)'''
#models loading
file_old = h5py.File('./modelli/m_old.h5','r+')
file_vgg_v0 = h5py.File('./modelli/m_vgg_v0.h5','r+')
file_vgg_v1 = h5py.File('./modelli/m_vgg_v1.h5','r+')
file_vgg_v2 = h5py.File('./modelli/m_vgg_v2.h5','r+')
file_vgg_v3 = h5py.File('./modelli/m_vgg_v3.h5','r+')
file_vgg_aut_v4 = h5py.File('./modelli/m_vgg_aut_v4.h5','r+')
file_inception = h5py.File('./modelli/m_inception.h5','r+')
file_resnet_v0 = h5py.File('./modelli/m_resnet_v0.h5','r+')
file_resnet_v2 = h5py.File('./modelli/m_resnet_v2.h5','r+')
file_resnet_v3 = h5py.File('./modelli/m_resnet_v3.h5','r+')
file_resnet_v4 = h5py.File('./modelli/m_resnet_v4.h5','r+')
m_old = load_model(file_old)
m_vgg_v0 = load_model(file_vgg_v0)
m_vgg_v1 = load_model(file_vgg_v1)
m_vgg_v2 = load_model(file_vgg_v2)
m_vgg_v3 = load_model(file_vgg_v3)
m_vgg_aut_v4 = load_model(file_vgg_aut_v4)
m_inception = load_model(file_inception)
m_resnet_v0 = load_model(file_resnet_v0)
m_resnet_v2 = load_model(file_resnet_v2)
m_resnet_v3 = load_model(file_resnet_v3)
m_resnet_v4 = load_model(file_resnet_v4)
import pickle 
filehandler = open('./objects.obj', 'rb') #errors='ignore'
objects = pickle.load(filehandler)
X_test_norm = objects['X_test_norm']
y_test = objects['y_test']
weights = objects['weights']
#models accuracies
'''scores = m_old.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_old.name, m_old.metrics_names[1], scores[1]*100))
scores = m_vgg_v0.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_vgg_v0.name, m_vgg_v0.metrics_names[1], scores[1]*100))
scores = m_vgg_v1.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_vgg_v1.name, m_vgg_v1.metrics_names[1], scores[1]*100))
scores = m_vgg_v2.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_vgg_v2.name, m_vgg_v2.metrics_names[1], scores[1]*100))
scores = m_vgg_v3.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_vgg_v3.name, m_vgg_v3.metrics_names[1], scores[1]*100))
scores = m_vgg_aut_v4.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_vgg_aut_v4.name, m_vgg_aut_v4.metrics_names[1], scores[1]*100))
scores = m_inception.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_inception.name, m_inception.metrics_names[1], scores[1]*100))
scores = m_resnet_v0.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_resnet_v0.name, m_resnet_v0.metrics_names[1], scores[1]*100))
scores = m_resnet_v2.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_resnet_v2.name, m_resnet_v2.metrics_names[1], scores[1]*100))
scores = m_resnet_v3.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_resnet_v3.name, m_resnet_v3.metrics_names[1], scores[1]*100))
scores = m_resnet_v4.evaluate(X_test_norm, y_test, verbose=2)
print("%s - %s: %.2f%%" % (m_resnet_v4.name, m_resnet_v4.metrics_names[1], scores[1]*100))'''


def predict_with_ensamble(X_test, weights, l_models): # X_test of shape (n_sample, 48, 48, 6) credo
  #VALUTAZIONE ENSEMBLE SUL TEST SET
  c=0
  yhats = np.empty((len(l_models),X_test.shape[0],6))
  #yhats = np.empty((4,3589))
  for model in l_models:
    tmp = model.predict(X_test)
    tmp = np.array(tmp)
    #print("tmp",tmp)
    #tmp = tmp*weights[c] #reweighting
    yhats[c] = tmp
    #yhats[c] = np.argmax(tmp,axis=1)
    c+=1

  for i in range(len(l_models)):
    yhats[i] = yhats[i]*weights[i]

  media = np.mean(yhats,axis=0)

  y_ensemble = np.argmax(media, axis=1)
  return y_ensemble

def get_expression(custom):
    objects = ['rabbia', 'paura', 'felicità', 'tristezza', 'sorpresa', 'normale']
    m=0.000000000000000000001
    a=custom[0]
    for i in range(0,len(a)):
        if a[i]>m:
            m=a[i]
            ind=i
    return objects[ind]

def new_get_expression(custom):
    objects = ['rabbia', 'paura', 'felicità', 'tristezza', 'sorpresa', 'normale']
    return objects[custom]

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

models = [m_vgg_v0, m_vgg_v1, m_vgg_v2, m_vgg_v3, m_vgg_aut_v4, m_inception, m_resnet_v2, m_resnet_v3, m_resnet_v4]
# predizione sul test
'''y_pred = predict_with_ensamble(X_test_norm, weights, models)
y_test_modified = np.argmax(y_test, axis=1)
from sklearn import metrics
print("Accuracy: %.3f" % (metrics.accuracy_score(y_test_modified, y_pred)*100))'''


# singola predizione
'''y_pred = predict_with_ensamble(np.array([X_test_norm[1]]), weights, models)
y_test_modified = np.argmax(y_test, axis=1)
print("y_pred:",y_pred,"y_test",[y_test_modified[1]])
from sklearn import metrics
print("Accuracy: %.3f" % (metrics.accuracy_score([y_test_modified[1]], y_pred)*100))'''

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
        i, d = divmod(time.time(), 4)
        #print("i: ",i, "d:", d, end="\n")
        if(d>3.9): # ogni 4 secondi
            x=get_img(frame)
            custom = predict_with_ensamble(x, weights, models)
            #custom = model.predict(x)
            expression=new_get_expression(custom[0])
            #expression=get_expression(custom)
            print("Espressione: ",expression, " (q per uscire)", end="\n", flush=True)
        
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