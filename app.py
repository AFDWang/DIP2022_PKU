# encoding:utf-8
# !/usr/bin/env python
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import datetime
import random
import HistEq as he
import FFT as Fft
import GLPF as glpf


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static','upload')


class Pic_str():
    def create_uuid(self): #生成唯一的图片的名称字符串，防止图片显示时的重名问题
        nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 生成当前时间
        randomNum = random.randint(0, 100)  # 生成的随机整数n，其中0<=n<=100
        if randomNum <= 10:
            randomNum = str(0) + str(randomNum)
        uniqueNum = str(nowTime) + str(randomNum)
        return uniqueNum


def add_border(img):
    # add white thin border
    if len(img.shape)==2:
        return cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=255)
    else:
        return cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=[255,255,255])

def resize(img):
    # resize the final image to a proper size for showing in the webpage.
    # scale
    height_max = 600
    width_max = 1200
    shape_now = img.shape
    scale = min(height_max / shape_now[0], width_max / shape_now[1])

    # resize image
    height = int(shape_now[0] * scale)
    width = int(shape_now[1] * scale)
    shape_resize = (width, height)
    return cv2.resize(img, shape_resize, interpolation=cv2.INTER_AREA)

def add_title(img,text='add_text'):
    # add Enlgish title in the top of the image
    if len(img.shape)==2:
        color=255
    else:
        color=(255,255,255)
    img=cv2.copyMakeBorder(img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=color)
    if len(img.shape)==2:
        return cv2.putText(img,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color= 0,thickness=7,lineType=cv2.LINE_AA)
    else:
        return cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,color= (0,0,0),thickness=7,lineType=cv2.LINE_AA)


@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/HistEq', methods=['POST'], strict_slashes=False)
def HistEq():
    file_dir = app.config['UPLOAD_FOLDER']
    save_eq_dir=''
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    try:
        f = request.files["photo"]
        fname = secure_filename(f.filename)
        _, ext = os.path.splitext(fname)
        new_filename = Pic_str().create_uuid()
        save_dir = os.path.join(file_dir, new_filename + ext)
        f.save(save_dir)
        # forward to processing image
        if request.form['mode'] == 'gray':
            img = cv2.imread(save_dir, 0)
            dst = he.MyMethod(img, 'gray')
            save_eq_dir = os.path.join(file_dir, new_filename + '_eq.jpg')
            cv2.imwrite(save_eq_dir, resize(np.concatenate([add_title(add_border(img),text='Origin') , add_title(add_border(dst),text='HistEq') ], 1)))
        else:
            img = cv2.imread(save_dir, 1)
            dsts = he.MyMethod(img, 'color')
            save_eq_dir = os.path.join(file_dir, new_filename + '_eq.jpg')
            texts=['Origin','RGB_HistEq','HSI_HistEq','MyMethod']
            cv2.imwrite(save_eq_dir,  resize(np.concatenate([ add_title(add_border(dsts[i]),text=texts[i]) for i in range(4)], 1)))
    except:
        pass

    return render_template('task1_HistEq.html', dir_result=save_eq_dir)







@app.route('/FFT', methods=['POST'], strict_slashes=False)
def FFT():
    file_dir = app.config['UPLOAD_FOLDER']
    save_eq_dir = ''
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    try:
        f = request.files["photo"]
        fname = secure_filename(f.filename)
        _, ext = os.path.splitext(fname)
        new_filename = Pic_str().create_uuid()
        save_dir = os.path.join(file_dir, new_filename + ext)
        f.save(save_dir)
        # forward to processing image
        if request.form['mode'] == 'gray':
            save_eq_dir = os.path.join(file_dir, new_filename + '_FFT.jpg')
            img = cv2.imread(save_dir, 0)
            Fft.FFT(img,'gray',save_eq_dir)
        else:
            save_eq_dir = os.path.join(file_dir, new_filename + '_FFT.jpg')
            img = cv2.imread(save_dir, 1)
            Fft.FFT(img, 'color', save_eq_dir)
    except:
        pass


    return render_template('task2_FFT.html',dir_result=save_eq_dir)



@app.route('/GLPF', methods=['POST'], strict_slashes=False)
def GLPF():
    file_dir = app.config['UPLOAD_FOLDER']
    save_eq_dir = ''
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    try:
        f = request.files["photo"]
        fname = secure_filename(f.filename)
        _, ext = os.path.splitext(fname)
        new_filename = Pic_str().create_uuid()
        save_dir = os.path.join(file_dir, new_filename + ext)
        f.save(save_dir)
        # forward to processing image
        if request.form['mode'] == 'gray':
            save_eq_dir = os.path.join(file_dir, new_filename + '_GLPF.jpg')
            img = cv2.imread(save_dir, 0)
            fig = glpf.img_generate(img,'gray')
            fig.savefig(save_eq_dir)
        else:
            save_eq_dir = os.path.join(file_dir, new_filename + '_GLPF.jpg')
            img = cv2.imread(save_dir, 1)
            fig = glpf.img_generate(img, 'color')
            fig.savefig(save_eq_dir)
    except:
        pass

    return render_template('task3_GLPF.html',dir_result=save_eq_dir)


if __name__ == '__main__':
    app.run(debug=True)