from flask import Flask,render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import infer
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


@app.route('/', methods=['GET', 'POST'])
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


            model = infer.get_model('../models/covid19.model')
            
            file_data = request.files['file'].read()
            data = cv2.imdecode(np.fromstring(file_data, np.uint8), cv2.IMREAD_COLOR)
            norm_data = infer.normalize_image(data)
            # result is [postive_prob , negative_prob]
            result = model.predict(norm_data)[0]
            # create result
            result = {
                "positive": result[0],
                "negative": result[1]
            }
            return  render_template('res.html',res= result)
        else:
            return '<h1> bad</h1>'
    return  render_template('index.html')
    
    

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080,debug=True)
