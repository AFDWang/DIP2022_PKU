import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from utils import cvtColor, add_border, resize_img, add_title
import time

class Laplacian_sharpening:
    def __init__(self, kernel_type='base'):
        if kernel_type == 'base':
            self.kernel = np.array([[0,1,0],
                                    [1,-4,1],
                                    [0,1,0]])
        elif kernel_type == 'extension':
            self.kernel = np.array([[1,1,1],
                                    [1,-8,1],
                                    [1,1,1]])
        self.kernel_size = 3
        self.h = 1
    
    def pad_edge(self, img):
        a, b = img.shape
        re = np.zeros((a+2*self.h, b+2*self.h), dtype=np.uint8)
        re[:a, self.h:b+self.h] = img
        re[2*self.h:a+2*self.h, self.h:b+self.h] = img
        re[self.h:a+self.h, :b] = img
        re[self.h:a+self.h, 2*self.h:b+2*self.h] = img
        corner = ([0,0,-1,-1], [0,-1,0,-1])
        re[corner] = img[corner]
        re[self.h:a+self.h, self.h:b+self.h] = img
        return re

    def sharpen(self, img):
        a, b = img.shape
        re = np.zeros_like(img, dtype=np.uint8)
        laplacian = np.zeros_like(img, dtype=np.uint8)
        img_padded = self.pad_edge(img)

        ### This is the FASTEST implementation (in my work) using full vectorization!
        img_list = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                img_list.append(img_padded[i:a+i, j:b+j]*self.kernel[i,j])
        img_sum = np.sum(np.stack(img_list, axis=0), axis=0)
        laplacian[:,:] = np.clip(img_sum,0,255)
        re[:,:] = np.clip(img-img_sum,0,255)

        # ### This is a FASTER implementation using vectorization for kernel!
        # for i in range(a):
        #     for j in range(b):
        #         sum_val = np.sum(img_padded[i:i+2*self.h+1, j:j+2*self.h+1]*self.kernel)
        #         laplacian[i,j] = np.clip(sum_val,0,255)
        #         re[i,j] = np.clip(img[i,j]-sum_val,0,255)

        # ### This is a SLOW implementation using loops!
        # for i in range(a):
        #     for j in range(b):
        #         temp = img_padded[i:i+2*self.h+1, j:j+2*self.h+1]
        #         sum_val = 0
        #         for p in range(self.kernel_size):
        #             for q in range(self.kernel_size):
        #                 sum_val += self.kernel[p,q]*temp[p,q]
        #         laplacian[i,j] = np.clip(sum_val,0,255)
        #         re[i,j] = np.clip(img[i,j]-sum_val,0,255)

        return re, laplacian

    def sharpen_color(self, img, type='rgb'):
        assert len(img.shape) == 3
        if type == 'rgb':
            re = img.copy()
            laplacian = img.copy()
            re[:,:,0], laplacian[:,:,0] = self.sharpen(img[:,:,0])
            re[:,:,1], laplacian[:,:,1] = self.sharpen(img[:,:,1])
            re[:,:,2], laplacian[:,:,2] = self.sharpen(img[:,:,2])
        elif type == 'hls':
            img_hls = cvtColor(img, type='BGR2HLS')
            re = img_hls.copy()
            laplacian = img_hls[:,:,1].copy()
            re[:,:,1], laplacian[:,:] = self.sharpen(img_hls[:,:,1])
            re = cvtColor(re, type='HLS2BGR')
        return re, laplacian

def save_all_sharpen_gray(results, save_path, resize = False):
    texts=['Origin', 'Sharpen_base', 'Laplacian_base', 'Sharpen_ext', 'Laplacian_ext']
    imgs = [add_title(add_border(results[i]),text=texts[i]) for i in range(len(texts))]
    pad = np.full_like(imgs[0],255,dtype=np.uint8)
    imgs = [np.concatenate(imgs[:3],1), np.concatenate([pad]+imgs[3:],1)]
    img = np.concatenate(imgs, 0)
    if resize:
        img = resize_img(img)
    cv2.imwrite(save_path, img)

def save_all_sharpen_color(results, save_path, resize = False):
    texts=['Origin', 'Sharpen_base_rgb', 'Sharpen_ext_rgb', 'Sharpen_base_hls', 'Sharpen_ext_hls']
    imgs = [add_title(add_border(results[i]),text=texts[i]) for i in range(len(texts))]
    pad = np.full_like(imgs[0],255,dtype=np.uint8)
    imgs = [np.concatenate(imgs[:3],1), np.concatenate([pad]+imgs[3:],1)]
    img = np.concatenate(imgs, 0)
    if resize:
        img = resize_img(img)
    cv2.imwrite(save_path, img)

def run_sharpen(img_dir, save_dir, img_name, type='gray', resize = False):
    filter_base = Laplacian_sharpening('base')
    filter_ext = Laplacian_sharpening('extension')
    task = 'Sharpen'
    print("Running...")

    if type == 'gray':
        img=cv2.imread(img_dir+img_name, cv2.IMREAD_GRAYSCALE)
        re_base, lap_base = filter_base.sharpen(img)
        re_ext, lap_ext = filter_ext.sharpen(img)
        results = [img, re_base, lap_base, re_ext, lap_ext]
        save_path = save_dir+gen_result_name(img_name, task)
        save_all_sharpen_gray(results, save_path, resize=resize)
    elif type == 'color':
        img=cv2.imread(img_dir+img_name,cv2.IMREAD_COLOR)    
        re_base_rgb, _ = filter_base.sharpen_color(img, type='rgb')
        re_ext_rgb, _ = filter_ext.sharpen_color(img, type='rgb')
        re_base_hls, _ = filter_base.sharpen_color(img, type='hls')
        re_ext_hls, _ = filter_ext.sharpen_color(img, type='hls')
        results = [img, re_base_rgb, re_ext_rgb, re_base_hls, re_ext_hls]
        save_path = save_dir+gen_result_name(img_name, task)
        save_all_sharpen_color(results, save_path, resize=resize)
    print("Done! Save to %s"%save_path)
    return save_path

def gen_result_name(img_name, task):
    l = img_name.split('.')
    return '%s_%s.%s' % (l[0], task, 'jpg')

if __name__ == '__main__':
    task = 'Sharpen'
    dir = 'dataset/%s/'%task
    save_dir = 'results/%s/'%task

    # filter_base = Laplacian_sharpening('base')
    # filter_ext = Laplacian_sharpening('extension')

    # ### Test single channel Laplacian
    # img_name = 'moon.tif'
    # re_name = gen_result_name(img_name, task)
    # img=cv2.imread(dir+img_name,cv2.IMREAD_GRAYSCALE)

    # re_base, lap_base = filter_base.sharpen(img)
    # re_ext, lap_ext = filter_ext.sharpen(img)
    # results = [img, re_base, lap_base, re_ext, lap_ext]
    # save_all_sharpen_gray(results, save_dir+gen_result_name(img_name, task))


    # filter_base = Laplacian_sharpening('base')
    # filter_ext = Laplacian_sharpening('extension')
    # img_name = 'moon.tif'
    # img=cv2.imread(dir+img_name,cv2.IMREAD_GRAYSCALE)
    # print(img.shape)
    # time0 = time.time()
    # for i in range(1):
    #     filter_base.sharpen(img)
    # time1 = time.time()
    # for i in range(1):
    #     filter_ext.sharpen(img)
    # time2 = time.time()
    # print(time1-time0,time2-time1)
    # exit(0)

    # ### Test multi channel Laplacian
    # img_name = 'histeqColor.jpg'
    # re_name = gen_result_name(img_name, task)
    # img=cv2.imread(dir+img_name,cv2.IMREAD_COLOR)

    # re_base_rgb, _ = filter_base.sharpen_color(img, type='rgb')
    # re_ext_rgb, _ = filter_ext.sharpen_color(img, type='rgb')
    # re_base_hls, _ = filter_base.sharpen_color(img, type='hls')
    # re_ext_hls, _ = filter_ext.sharpen_color(img, type='hls')
    # results = [img, re_base_rgb, re_ext_rgb, re_base_hls, re_ext_hls]
    # save_all_sharpen_color(results, save_dir+gen_result_name(img_name, task))

    run_sharpen(dir, save_dir, 'moon.tif', type='gray')
    run_sharpen(dir, save_dir, 'histeqColor.jpg', type='color')