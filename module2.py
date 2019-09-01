
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
import keras
from keras.models import model_from_json
from keras.layers import *
import pyaudio
import wave    
from gtts import gTTS
import vlc
import librosa
import librosa.display

from playsound import playsound
import speech_recognition as sr0
import os



def countfaces():
    camera = cv2.VideoCapture(0)
    while True:
        #time.sleep(10)
        return_value,image = camera.read()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',gray)
        time.sleep(5)
        #if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite('test.jpg',image)
        #break

        time.sleep(5)
        txtfiles = [] 
        for file in glob.glob("*.jpg"):
            txtfiles.append(file)
    
        for ix in txtfiles:
            img = cv2.imread(ix,cv2.IMREAD_COLOR)
            imgtest1 = img.copy()
            imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)   
            facecascade = cv2.CascadeClassifier('D:\Kinshuk Gupta\Documents\Hack\haarcascade_frontalface_default.xml')    
            eye_cascade = cv2.CascadeClassifier('D:\Kinshuk Gupta\Documents\Hack\haarcascade_eye.xml')
   
            faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.2, minNeighbors=5) 
            print("Total number of Faces found",len(faces))
    
            if(len(faces)>1):
                print("Make sure there is no one else around you")
                time.sleep(5)

            for (x, y, w, h) in faces:
                face_detect = cv2.rectangle(imgtest, (x, y), (x+w, y+h), (255, 0, 255), 2)
                roi_gray = imgtest[y:y+h, x:x+w]
                roi_color = imgtest[y:y+h, x:x+w]        
                plt.imshow(face_detect)
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    eye_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
                    plt.imshow(eye_detect)

        if cv2.waitKey(1)& 0xFF == ord('s'):
            camera.release()
            cv2.destroyAllWindows()




def loadModel():
    opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model/aug_noiseNshift_6class6_np.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

def runModel():
    #livedf= pd.DataFrame(columns=['feature'])
    data, sampling_rate = librosa.load('output10.wav')
    X, sample_rate = librosa.load('output10.wav', res_type='kaiser_fast',duration=3.0,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    livepreds = loaded_model.predict(twodim,
                         batch_size=16, 
                         verbose=1)
    livepreds1=livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    livepredictions = (lb.inverse_transform((liveabc)))
    

def AudioRecorder():
   # The text that you want to convert to audio 
    mytext = ["Welcome to INTERVIEW BOT!","First question from our side what is java","Is multiple inheritance supported in java","what are classes and objects in java","HOW java is different from other languages"]
  
    # Language in which you want to convert 
    language = 'en'


    def audiototext(audio):
        r = sr.Recognizer()
        with sr.AudioFile(audio) as source:
            audio = r.record(source)
            print ('Done!') 
        text = r.recognize_google(audio)
    
        return text
    
    lst1=["good","better than others","uses classes and objects","objects","classes","better","android development","android"]
    lst2=["no","ambiguity","not supported","not"]
    lst3=["classes are blocks","blocks","structure","data structure"," instance","reference"]
    lst4=["oops","object oriented","various uses"]
    lstans=[lst1,lst2,lst3,lst4]
    def anscheck(ans,count):
        success=0
        for i in lstans[count]:
            if(i in ans):
                success+=1
        print(success)     
    
    def recordaudio1(i):
    
        CHUNK = 1024 
        FORMAT = pyaudio.paInt16 #paInt8
        CHANNELS = 2 
        RATE = 44100 #sample rate
        RECORD_SECONDS = 10
        str="answered"+i+".wav"
        WAVE_OUTPUT_FILENAME ="output10.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer

        print("* recording")
 
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data) # 2 bytes(16 bits) per channel

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
        return str

  
    # Passing the text and language to the engine,  
    # here we have marked slow=False. Which tells  
    # the module that the converted audio should  
    # have a high speed 
    count=0
    for i in mytext:
        myobj = gTTS(text=i, lang=language, slow=False) 
         # Saving the converted audio in a mp3 file named 
         # welcome 
        str="welcome"+i+".mp3"
        myobj.save(str) 
         # Playing the converted file 
    

        playsound(str)    
        # os.system("welcome.mp3")
        if(i!="Welcome to INTERVIEW BOT!"):
            s=recordaudio1(i)
            ans=audiototext(s)
            print(ans)
            anscheck(ans,count)
            count+=1 







def Text_to_Speech():
    tts = gTTS(text='Good morning', lang='en')
    tts.save("good.mp3")
    player=vlc.MediaPlayer("good.mp3")
    player.play()

from threading import Thread



if __name__ == '__main__':
    loadModel()
    Text_to_Speech()
    AudioRecorder()
    Thread(target = runModel).start()
    Thread(target = countfaces).start()
