# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:38:58 2019

@author: Abhishek Tripathi
"""

import flask
from flask import request, render_template,send_file
import base64
import cv2
app = flask.Flask(__name__)
#import applyOnImage


@app.route('/logCount', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':  
        f = request.files['image']
        print('saving the uploaded image ')
        f.save('var1/' + 'logInput.jpg')
 #       path = applyOnImage.apply('var1/logInput.jpg','var1/logmodel/best_model_trancos_ResFCN_logs_tanny.pth','ResFCN')
        img1 = cv2.imread('var1/' + 'logInput.jpg')
        print('image 1 read ')
        _, img_encoded = cv2.imencode('.jpg', img1)
        print('sending data back')
        return base64.b64encode(img_encoded)

if __name__ == '__main__':
    app.run(port=5001,debug=False, threaded=False)
''' app.run(port=5000, debug=True)'''

