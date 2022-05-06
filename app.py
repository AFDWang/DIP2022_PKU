# encoding:utf-8
# !/usr/bin/env python
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import datetime
import random
from HistEq import run_histeq
from Morph import run_morph, run_morph_multi_kernel_size
from Sharpen import run_sharpen

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

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/HistEq', methods=['POST'], strict_slashes=False)
def HistEq():
    file_dir = app.config['UPLOAD_FOLDER']
    result_dir=''
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    try:
        f = request.files["photo"]
        fname = secure_filename(f.filename)
        _, ext = os.path.splitext(fname)
        new_filename = Pic_str().create_uuid()
        save_dir = os.path.join(file_dir, new_filename + ext)
        f.save(save_dir)
        img_dir = file_dir+'/'
        img_name = new_filename + ext
        # forward to processing image
        if request.form['mode'] == 'gray':
            result_dir = run_histeq(img_dir, img_dir, img_name, type='gray', resize=True)
        else:
            result_dir = run_histeq(img_dir, img_dir, img_name, type='color', resize=True)
    except:
        pass

    return render_template('task1_HistEq.html', dir_result=result_dir)

@app.route('/Morph', methods=['POST'], strict_slashes=False)
def Morph():
    file_dir = app.config['UPLOAD_FOLDER']
    result_dir = ''
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    try:
        f = request.files["photo"]
        fname = secure_filename(f.filename)
        _, ext = os.path.splitext(fname)
        new_filename = Pic_str().create_uuid()
        save_dir = os.path.join(file_dir, new_filename + ext)
        f.save(save_dir)
        img_dir = file_dir+'/'
        img_name = new_filename + ext
        # forward to processing image
        if request.form['mode'] == 'gray':
            result_dir = run_morph_multi_kernel_size(img_dir, img_dir, img_name, type='gray', kernel_type = 1, resize=True)
        else:
            result_dir = run_morph_multi_kernel_size(img_dir, img_dir, img_name, type='color', color_type='rgb', kernel_type = 1, resize=True)
    except:
        pass


    return render_template('task2_Morph.html',dir_result=result_dir)

@app.route('/Sharpen', methods=['POST'], strict_slashes=False)
def Sharpen():
    file_dir = app.config['UPLOAD_FOLDER']
    result_dir = ''
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    try:
        f = request.files["photo"]
        fname = secure_filename(f.filename)
        _, ext = os.path.splitext(fname)
        new_filename = Pic_str().create_uuid()
        save_dir = os.path.join(file_dir, new_filename + ext)
        f.save(save_dir)
        img_dir = file_dir+'/'
        img_name = new_filename + ext
        # forward to processing image
        if request.form['mode'] == 'gray':
            result_dir = run_sharpen(img_dir, img_dir, img_name, type='gray', resize=True)
        else:
            result_dir = run_sharpen(img_dir, img_dir, img_name, type='color', resize=True)
    except:
        pass

    return render_template('task3_Sharpen.html',dir_result=result_dir)


if __name__ == '__main__':
    app.run(debug=True)