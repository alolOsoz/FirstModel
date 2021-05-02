import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import os, shutil, stat
import glob as gb
import cv2
import tensorflow as tf
import keras
from flask import Flask , render_template,request
import numpy as np



#import  function to reuse the model
from tensorflow.keras.models import load_model
model_path="E:\SASUniversityEdition\Machine\MODEL\saveModel.h5"
 
# load model
import efficientnet.keras as efn   
model=load_model(model_path)

import cv2
# fuction to do some configration on image

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/answer",methods=['POST'])
def answer():
    img = request.files['img']
    img.save('static/asd.jpg')
    #read image
    img=plt.imread(img)
    #resize image
    img=cv2.resize(img,(224,224))
    #normlize image
    img=img/255.0
    # expand dim
    img=np.expand_dims(img , axis=0) # two image as (1,224,224,3)or use reshape method
    prediction = model.predict(img)
    if(prediction>=0.5):
        result= 'abnormal'
    else:
        result= 'normal'
    return render_template('answer.html', data=result)

# @deploymodel_app.route('/load_img')
# def load_img():
#     return send_from_directory('static', "asd.jpg")
 


if __name__=="__main__":
    app.run(debug=True)