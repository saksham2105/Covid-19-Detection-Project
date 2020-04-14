import pyrebase

config = {
    "apiKey": "AIzaSyCmV8sCbek_w54QklHvvinjw_DvBoxMQ_U",
    "authDomain" : "covidetect19.firebaseapp.com",
    "databaseURL" : "https://covidetect19.firebaseio.com",
    "projectId" : "covidetect19",
    "storageBucket" : "covidetect19.appspot.com",
    "messagingSenderId" : "60972454754",
    "appId" : "1:60972454754:web:a5bf174d3e98ef5645923b"
}

firebase = pyrebase.initialize_app(config)

storage = firebase.storage()




from flask import Flask,render_template, flash, request, redirect, url_for

from werkzeug.utils import secure_filename
import tensorflow.keras as keras
import cv2
import os
import json

import numpy as np


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/ping', methods=['GET'])
def heart_beat():
    return 'PONG'

@app.route('/sitemap.xml', methods=['GET'])
def sitemap():
    return render_template('sitemap.xml')

def normalize_image(data):
    # normalize input
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = data
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    inputX = []
    inputX.append(image)
    inputX = np.array(inputX)/255.0
    return inputX

def get_model(path):
    model = keras.models.load_model(path)
    return model

@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)


            model = get_model('models/covid19.model')
            upload = request.files['file']
            
            
            file_data = request.files['file'].read()
            data = cv2.imdecode(np.fromstring(file_data, np.uint8), cv2.IMREAD_COLOR)
            norm_data = normalize_image(data)
            # result is [postive_prob , negative_prob]
            result = model.predict(norm_data)[0]
            # create result
            result = {
                'positive': result[0],
                'negative': result[1]
            }
            resu = result['positive']
            if resu > 0.5:
                 storage.child('covid/' + upload.filename).put(upload)
            else:
                 storage.child('Normal/' + upload.filename).put(upload)    
            return  render_template('res.html',res= result)
        else:
            return '<h1> bad</h1>'
    return  render_template('index.html')



if __name__ == '__main__':
    
    app.run(host='127.0.0.1',port=8080,debug=True)