import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from utils import cvtColor, add_border, resize_img, add_title
import time

class Morphlogy:
    def __init__(self, kernel=np.ones((3,3))):
        self.kernel = kernel
        assert kernel.shape[0] == kernel.shape[1]
        self.kernel_size = kernel.shape[0]
        assert self.kernel_size%2 == 1
        self.h = self.kernel_size//2

    def Erose(self, img):
        a, b = img.shape
        re = np.zeros_like(img, dtype=np.uint8)

        ### This is the FASTEST implementation (in my work) using full vectorization!
        img_list = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if self.kernel[i,j]:
                    img_list.append(img[i:a-2*self.h+i, j:b-2*self.h+j])
        img_min = np.min(np.stack(img_list, axis=0), axis=0)
        re[self.h:a-self.h, self.h:b-self.h] = img_min

        # ### This is a FASTER implementation using vectorization for kernel!
        # kernel = self.kernel.copy()
        # kernel[np.where(kernel==0)]=999999
        # for i in range(self.h, a-self.h):
        #     for j in range(self.h, b-self.h):
        #         re[i,j] = np.min(img[i-self.h:i+self.h+1, j-self.h:j+self.h+1]*kernel)

        # ### This is a SLOW implementation using loops!
        # for i in range(self.h, a-self.h):
        #     for j in range(self.h, b-self.h):
        #         temp = img[i-self.h:i+self.h+1, j-self.h:j+self.h+1]
        #         min_val = np.inf
        #         for p in range(self.kernel_size):
        #             for q in range(self.kernel_size):
        #                 if self.kernel[p,q]:
        #                     min_val = min(min_val, temp[p,q])
        #         re[i,j] = min_val
        return re

    def Dilate(self, img):
        a, b = img.shape
        re = np.zeros_like(img, dtype=np.uint8)

        ### This is the FASTEST implementation (in my work) using full vectorization!
        img_list = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if self.kernel[i,j]:
                    img_list.append(img[i:a-2*self.h+i, j:b-2*self.h+j])
        img_max = np.max(np.stack(img_list, axis=0), axis=0)
        re[self.h:a-self.h, self.h:b-self.h] = img_max

        # ### This is a FASTER implementation using vectorization for kernel!
        # kernel = self.kernel.copy()
        # kernel[np.where(kernel==0)] = -999999
        # for i in range(self.h, a-self.h):
        #     for j in range(self.h, b-self.h):
        #         re[i,j] = np.max(img[i-self.h:i+self.h+1, j-self.h:j+self.h+1]*kernel)

        # ### This is a SLOW implementation using loops!
        # for i in range(self.h, a-self.h):
        #     for j in range(self.h, b-self.h):
        #         temp = img[i-self.h:i+self.h+1, j-self.h:j+self.h+1]
        #         max_val = -np.inf
        #         for p in range(self.kernel_size):
        #             for q in range(self.kernel_size):
        #                 if self.kernel[p,q]:
        #                     max_val = max(max_val, temp[p,q])
        #         re[i,j] = max_val
        return re

    def Open(self, img):
        return self.Dilate(self.Erose(img))

    def Close(self, img):
        return self.Erose(self.Dilate(img))

    def Bound(self, img):
        return img - self.Erose(img)

    def gen_all_morph(self, img):
        img_erose = self.Erose(img)
        img_dilate = self.Dilate(img)
        img_open = self.Dilate(img_erose)
        img_close = self.Erose(img_dilate)
        img_bound = img - img_erose
        return img_erose, img_dilate, img_open, img_close, img_bound
        # return self.Erose(img), self.Dilate(img), self.Open(img), self.Close(img), self.Bound(img)

    def gen_all_morph_color(self, img, type='rgb'):
        assert len(img.shape) == 3
        if type == 'rgb':
            results = [img.copy() for i in range(5)]
            channel_0 = self.gen_all_morph(img[:,:,0])
            channel_1 = self.gen_all_morph(img[:,:,1])
            channel_2 = self.gen_all_morph(img[:,:,2])
            for i in range(5):
                results[i][:,:,0] = channel_0[i]
                results[i][:,:,1] = channel_1[i]
                results[i][:,:,2] = channel_2[i]
            return results
        elif type == 'hls':
            img_hls = cvtColor(img, type='BGR2HLS')
            img_l = img_hls[:,:,1].copy()
            re_imgs_l = self.gen_all_morph(img_l)
            results = []
            for re_img in re_imgs_l:
                re = img_hls.copy()
                re[:,:,1]=re_img
                re = cvtColor(re, type='HLS2BGR')
                results.append(re)
        return results

def save_all_morph(results, save_path, resize = False):
    texts=['Origin', 'Erose', 'Dilate', 'Open', 'Close', 'Bound']
    imgs = [add_title(add_border(results[i]),text=texts[i]) for i in range(len(texts))]
    imgs = [np.concatenate(imgs[:3], 1), np.concatenate(imgs[3:], 1)]
    img = np.concatenate(imgs, 0)
    if resize:
        img = resize_img(img)
    cv2.imwrite(save_path, img)

def save_all_morph_multi_kernel_size(results, save_path, resize = False):
    texts=['Origin', 'Erose_3', 'Dilate_3', 'Open_3', 'Close_3', 'Bound_3', 'Erose_5', 'Dilate_5', 'Open_5', 'Close_5', 'Bound_5']
    imgs = [add_title(add_border(results[i]),text=texts[i]) for i in range(len(texts))]
    pad = np.full_like(imgs[0],255,dtype=np.uint8)
    imgs = [np.concatenate(imgs[:3], 1), np.concatenate(imgs[3:6], 1), np.concatenate([pad]+imgs[6:8], 1), np.concatenate(imgs[8:11], 1)]
    img = np.concatenate(imgs, 0)
    if resize:
        img = resize_img(img)
    cv2.imwrite(save_path, img)

def gen_result_name(img_name, task):
    l = img_name.split('.')
    return '%s_%s.%s' % (l[0], task, 'jpg')

def run_morph_multi_kernel_size(img_dir, save_dir, img_name, type='gray', resize = False, color_type='rgb', kernel_type = 1):
    kernel0_3 = np.array([  [1,1,1],
                            [1,1,1],
                            [1,1,1]])
    kernel1_3 = np.array([  [0,1,0],
                            [1,1,1],
                            [0,1,0]])
    kernel0_5 = np.array([  [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1]])
    kernel1_5 = np.array([  [0,0,1,0,0],
                            [0,1,1,1,0],
                            [1,1,1,1,1],
                            [0,1,1,1,0],
                            [0,0,1,0,0]])

    kernels = {3:[kernel0_3, kernel1_3], 5:[kernel0_5, kernel1_5]}
    morph_3 = Morphlogy(kernels[3][kernel_type])
    morph_5 = Morphlogy(kernels[5][kernel_type])
    task = 'Morph'
    print("Running...")

    if type == 'gray':
        img=cv2.imread(img_dir+img_name, cv2.IMREAD_GRAYSCALE)
        results_3 = morph_3.gen_all_morph(img)
        results_5 = morph_5.gen_all_morph(img)
        save_path = save_dir+gen_result_name(img_name,task+'_35_kernel%d'%kernel_type)
        save_all_morph_multi_kernel_size([img]+list(results_3)+list(results_5), save_path=save_path, resize=resize)
    elif type == 'color':
        img=cv2.imread(img_dir+img_name,cv2.IMREAD_COLOR)
        results_3 = morph_3.gen_all_morph_color(img, type=color_type)
        results_5 = morph_5.gen_all_morph_color(img, type=color_type)
        save_path = save_dir+gen_result_name(img_name,task+'_35_kernel%d_%s'%(kernel_type, color_type))
        save_all_morph_multi_kernel_size([img]+list(results_3)+list(results_5), save_path=save_path, resize=resize)
    print("Done! Save to %s"%save_path)
    return save_path

def run_morph(img_dir, save_dir, img_name, type='gray', resize = False, color_type='rgb', kernel_type = 1):
    kernel0_3 = np.array([  [1,1,1],
                            [1,1,1],
                            [1,1,1]])
    kernel1_3 = np.array([  [0,1,0],
                            [1,1,1],
                            [0,1,0]])
    kernel0_5 = np.array([  [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1]])
    kernel1_5 = np.array([  [0,0,1,0,0],
                            [0,1,1,1,0],
                            [1,1,1,1,1],
                            [0,1,1,1,0],
                            [0,0,1,0,0]])

    kernels = {3:[kernel0_3, kernel1_3], 5:[kernel0_5, kernel1_5]}
    morph_3 = Morphlogy(kernels[3][kernel_type])
    morph_5 = Morphlogy(kernels[5][kernel_type])
    task = 'Morph'
    print("Running...")

    if type == 'gray':
        img=cv2.imread(img_dir+img_name, cv2.IMREAD_GRAYSCALE)
        if img.shape[0]>500 or img.shape[1]>500:
            morph = morph_5
        else:
            morph = morph_3
        results = morph.gen_all_morph(img)
        save_path = save_dir+gen_result_name(img_name,task+'_kernel%d'%kernel_type)
        save_all_morph([img]+list(results), save_path=save_path, resize=resize)
    elif type == 'color':
        img=cv2.imread(img_dir+img_name,cv2.IMREAD_COLOR)
        if img.shape[0]>500 or img.shape[1]>500:
            morph = morph_5
        else:
            morph = morph_3
        results = morph.gen_all_morph_color(img, type=color_type)
        save_path = save_dir+gen_result_name(img_name,task+'_kernel%d_%s'%(kernel_type, color_type))
        save_all_morph([img]+list(results), save_path=save_path, resize=resize)
    print("Done! Save to %s"%save_path)
    return save_path

if __name__ == '__main__':
    task = 'Morph'
    dir = 'dataset/%s/'%task
    save_dir = 'results/%s/'%task

    # kernel0 = np.array([[1,1,1],
    #                     [1,1,1],
    #                     [1,1,1]])
    # kernel1 = np.array([[0,1,0],
    #                     [1,1,1],
    #                     [0,1,0]])
    # morph = Morphlogy(kernel1)

    # img_name = 'dct5_gray.jpg'
    # img=cv2.imread(dir+img_name,cv2.IMREAD_GRAYSCALE)
    # results = morph.gen_all_morph(img)
    # save_all_morph([img]+list(results), save_path=save_dir+gen_result_name(img_name,task))

    # kernel0_3 = np.array([  [1,1,1],
    #                         [1,1,1],
    #                         [1,1,1]])
    # kernel0_5 = np.array([  [1,1,1,1,1],
    #                         [1,1,1,1,1],
    #                         [1,1,1,1,1],
    #                         [1,1,1,1,1],
    #                         [1,1,1,1,1]])
    # morph_3 = Morphlogy(kernel0_3)
    # morph_5 = Morphlogy(kernel0_5)
    # img_name = 'dct5_gray.jpg'
    # img=cv2.imread(dir+img_name,cv2.IMREAD_GRAYSCALE)
    # print(img.shape)
    # time0 = time.time()
    # for i in range(1):
    #     results = morph_3.gen_all_morph(img)
    # time1 = time.time()
    # for i in range(1):
    #     results = morph_5.gen_all_morph(img)
    # time2 = time.time()
    # print(time1-time0,time2-time1)
    # exit(0)

    # img_name = 'word_bw.bmp'
    # img=cv2.imread(dir+img_name,cv2.IMREAD_GRAYSCALE)
    # results = morph.gen_all_morph(img)
    # save_all_morph([img]+list(results), save_path=save_dir+gen_result_name(img_name,task))

    # img_name = 'histeqColor.jpg'
    # img=cv2.imread(dir+img_name,cv2.IMREAD_COLOR)
    # results = morph.gen_all_morph_color(img, type='hls')
    # save_all_morph([img]+list(results), save_path=save_dir+gen_result_name(img_name,task+'_hls'))

    # results = morph.gen_all_morph_color(img, type='rgb')
    # save_all_morph([img]+list(results), save_path=save_dir+gen_result_name(img_name,task+'_rgb'))

    # Test run_sample
    run_morph(dir, save_dir, 'dct5.jpg', type='gray', kernel_type = 1)
    run_morph(dir, save_dir, 'dct5.jpg', type='gray', kernel_type = 0)
    run_morph(dir, save_dir, 'word_bw.bmp', type='gray', kernel_type = 1)
    run_morph(dir, save_dir, 'word_bw.bmp', type='gray', kernel_type = 0)

    run_morph(dir, save_dir, 'histeqColor.jpg', type='color', color_type='rgb', kernel_type = 0)
    run_morph(dir, save_dir, 'histeqColor.jpg', type='color', color_type='hls', kernel_type = 0)
