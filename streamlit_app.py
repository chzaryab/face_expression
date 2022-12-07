import keras
import os
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI
import cv2
from pydantic import BaseModel
import random
import base64
import string
import io
from PIL import Image
outputlist=['angry','fear','happy','neutral','sad','surprise']
model=keras.models.load_model('facial_reg_inv3.h5',compile=False)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  
app=FastAPI()


class Msg(BaseModel):
    base64string: str


def cv2operation(frame):
    gray_frame = frame
    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    hash = random.getrandbits(28)
    hash=str(hash)
    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (150, 150)), -1), 0)

        # predict the emotions
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, outputlist[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(f'dddd{hash}.jpg',frame)
    return frame,hash






def base64str_to_PILImage(base64string):
    base64_img_bytes = base64string.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

@app.post('/modeloutput')
async def get_model(item:Msg):

    #nparr = np.fromstring(item.dict().get('encoded_img'), np.uint8)
    # f=open('base64.txt', 'rb')
    # print(f.read())
    # nparr = np.fromstring(f.read(), np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   
    # _, encoded_img = cv2.imencode('.PNG', return_img)
    # encoded_img = base64.b64encode(encoded_img)
   

    image_64_decode = base64str_to_PILImage(item.base64string)
  
    return_img,_ = cv2operation(image_64_decode)
    # create a writable image and write the decoding result
    image_64_encode = base64.b64encode(return_img)
    return{
      
        'image':image_64_encode

    }
