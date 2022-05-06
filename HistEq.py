import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from utils import cvtColor, add_border, resize_img, add_title, resize_img_width
import time

def HistEq_single_channel(img):
    a, b = img.shape
    n = a*b
    img = img.astype(np.uint8)
    # Build Histogram
    hist = np.histogram(img, 256, (0,256))[0]
    # Calculate Prob
    prob = hist / n
    # Calculate Cumulative Prob
    cum_prob = np.zeros(256)
    temp = 0.0
    for i in range(256):
        temp += prob[i]
        cum_prob[i] = temp
    # Generate mapping
    mapping = (np.around(cum_prob*255)).astype(np.uint8)
    # Generate result
    re = mapping[img].astype(np.uint8)
    return re

def HistEq_rgb_avg(img):
    a, b, _ = img.shape
    n = a*b
    img = img.astype(np.uint8)
    # Build Histogram
    hist_list = []
    for k in range(3):
        hist_list.append(np.histogram(img[:,:,k], 256, (0,256))[0])
    hist = np.stack(hist_list, axis=0)
    # Calculate Prob
    prob = hist / n
    # Calculate Avg prob
    prob = np.mean(prob, axis=0)
    # Calculate Cumulative Prob
    cum_prob = np.zeros(256)
    temp = 0.0
    for i in range(256):
        temp += prob[i]
        cum_prob[i] = temp
    # Generate mapping
    mapping = (np.around(cum_prob*255)).astype(np.uint8)
    # Generate result
    re = mapping[img].astype(np.uint8)
    return re

def show_hist(img, save_path):
    hist = get_hist(img)
    plt.clf()
    plt.figure(figsize=(5,5))
    plt.plot(hist)
    plt.grid()
    plt.xlabel('r_k', fontsize=10)
    plt.ylabel('h(r_k)', fontsize=10)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.savefig(save_path)

def get_hist(img):
    hist = np.histogram(img, 256, (0,256))[0]
    return hist

def HistEq_rgb(img):
    assert len(img.shape)==3
    # equalizeHist in BGR
    img_BGR_eq=img.copy()
    img_BGR_eq[:, :, 0] = HistEq_single_channel(img_BGR_eq[:, :, 0])
    img_BGR_eq[:, :, 1] = HistEq_single_channel(img_BGR_eq[:, :, 1])
    img_BGR_eq[:, :, 2] = HistEq_single_channel(img_BGR_eq[:, :, 2])
    return img_BGR_eq

def HistEq_hls(img_hls):
    # equlizeHist in lightness
    img_HLS_eq = img_hls.copy()
    img_HLS_eq[:,:,1]=HistEq_single_channel(img_HLS_eq[:,:,1])
    return img_HLS_eq

def HistEq_hls_ls(img_hls, deg=350):
    # equlizeHist in lightness ; equlizeHist and exp in saturation
    img_HLS_eq = img_hls.copy()
    img_HLS_eq[:,:,1] = HistEq_single_channel(img_HLS_eq[:,:,1])
    img_HLS_eq[:,:,2] = np.uint8(np.clip(img_HLS_eq[:,:,2]/255*deg,0,255))
    return img_HLS_eq

def HistEq_rgb_avg_hls_ls(img):
    img_rgb_avg = HistEq_rgb_avg(img)
    img_rgb_avg_hls_ls = HistEq_hls_ls(cvtColor(img_rgb_avg, type='BGR2HLS'), deg=450)
    return img_rgb_avg_hls_ls

def gen_all_HistEq_color(img):
    assert len(img.shape)==3
    start = time.time()
    img_hls = cvtColor(img, type='BGR2HLS')
    re_rgb = HistEq_rgb(img)
    print("Generated rgb!")
    re_hls = cvtColor(HistEq_hls(img_hls), type='HLS2BGR')
    print("Generated hls!")
    re_rgb_avg = HistEq_rgb_avg(img)
    print("Generated rgb_avg!")
    re_hls_ls = cvtColor(HistEq_hls_ls(img_hls), type='HLS2BGR')
    print("Generated hls_ls!")
    re_rgb_avg_hls_ls = cvtColor(HistEq_rgb_avg_hls_ls(img), type='HLS2BGR')
    print("Generated rgb_avg+hls_ls!")
    end = time.time()
    # print(end-start)
    return re_rgb,re_hls,re_rgb_avg,re_hls_ls,re_rgb_avg_hls_ls

def save_all_HistEq_color(results, save_path, resize = False):
    texts=['Origin', 'RGB', 'HSI', 'RGB_Avg', 'HSI_Sat', 'RGB_Avg+HSI_Sat']
    imgs = [add_title(add_border(results[i]),text=texts[i]) for i in range(len(texts))]
    imgs = [np.concatenate(imgs[:3], 1), np.concatenate(imgs[3:], 1)]
    img = np.concatenate(imgs, 0)
    if resize:
        img = resize_img(img)
    cv2.imwrite(save_path, img)

def gen_all_HistEq_gray(img):
    assert len(img.shape)==2
    re = HistEq_single_channel(img)
    return [re]

def save_all_HistEq_gray(results, save_path, resize = False):
    show_hist(results[0], save_path)
    hist_ori = cv2.imread(save_path,cv2.IMREAD_GRAYSCALE)
    show_hist(results[1], save_path)
    hist_eq = cv2.imread(save_path,cv2.IMREAD_GRAYSCALE)
    hist_ori = resize_img_width(hist_ori, results[0].shape[1])
    hist_eq = resize_img_width(hist_eq, results[0].shape[1])
    results = results+[hist_ori,hist_eq]
    texts = ['Origin', 'HistEq', 'Histogram_Ori', 'Histogram_Eq']
    imgs = [add_title(add_border(results[i]),text=texts[i]) for i in range(len(texts))]
    imgs = [np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)]
    img = np.concatenate(imgs, 0)
    if resize:
        img = resize_img(img)
    cv2.imwrite(save_path, img)

def run_histeq(img_dir, save_dir, img_name, type='gray', resize = False):
    task = 'HistEq'
    print("Running...")
    if type == 'gray':
        img=cv2.imread(img_dir+img_name, cv2.IMREAD_GRAYSCALE)
        results = gen_all_HistEq_gray(img)
        save_path = save_dir+gen_result_name(img_name, task)
        save_all_HistEq_gray([img]+list(results), save_path=save_path, resize=resize)
    elif type == 'color':
        img=cv2.imread(img_dir+img_name,cv2.IMREAD_COLOR)
        results = gen_all_HistEq_color(img)
        save_path = save_dir+gen_result_name(img_name, task)
        save_all_HistEq_color([img]+list(results), save_path=save_path, resize=resize)
    print("Done! Save to %s"%save_path)
    return save_path

def gen_result_name(img_name, task):
    l = img_name.split('.')
    return '%s_%s.%s' % (l[0], task, 'jpg')

if __name__ == '__main__':
    task = 'HistEq'
    dir = 'dataset/%s/'%task
    save_dir = 'results/%s/'%task

    ### Test single channel HistEq
    # img_name = 'histeq1.jpg'
    # re_name = gen_result_name(img_name, task)
    # img=cv2.imread(dir+img_name,cv2.IMREAD_GRAYSCALE)
    # show_hist(img, save_dir+'hist_origin.jpg')
    # re = HistEq_single_channel(img)
    # cv2.imwrite(save_dir+re_name,re)
    # show_hist(re, save_dir+'hist_eq.jpg')
    # # call cv2 function only for testing the correctness of my algorithm!
    # re_cv2 = cv2.equalizeHist(img)

    # ### Test multi channel HistEq
    # img_name = 'histeqColor.jpg'
    # re_name = gen_result_name(img_name, task)
    # img=cv2.imread(dir+img_name,cv2.IMREAD_COLOR)
    # print(img.shape)

    # # Test img cvt func
    # time0 = time.time()
    # for i in range(10):
    #     img_hls = cvtColor(img, type='BGR2HLS')
    # time1 = time.time()
    # # call cv2 function only for testing the correctness of my algorithm!
    # #img_hls_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # for i in range(10):
    #     img_gbr = cvtColor(img_hls, type='HLS2BGR')
    # time2 = time.time()
    # print(time1-time0, time2-time1)
    # #cv2.imwrite(save_dir+'color_rgb2hls2rgb.jpg',img_gbr)

    # # Test rgb histEq
    # re_rgb = HistEq_rgb(img)
    # cv2.imwrite(save_dir+'color_rgb.jpg',re_rgb)

    # # Test rgb avg histEq
    # re_rgb_avg = HistEq_rgb_avg(img)
    # cv2.imwrite(save_dir+'color_rgb_avg.jpg',re_rgb_avg)

    # # Test hls histEq
    # re_hls = HistEq_hls(cvtColor(img, type='BGR2HLS'))
    # cv2.imwrite(save_dir+'color_hls.jpg',cvtColor(re_hls, type='HLS2BGR'))

    # time0 = time.time()
    # for i in range(1):
    #     cvtColor(HistEq_hls(cvtColor(img, type='BGR2HLS')), type='HLS2BGR')
    # time1 = time.time()
    # print(time1-time0)

    # # Test hls_ls histEq
    # re_hls_ls = HistEq_hls_ls(cvtColor(img, type='BGR2HLS'))
    # cv2.imwrite(save_dir+'color_hls_ls.jpg',cvtColor(re_hls_ls, type='HLS2BGR'))

    # # Test rgb_avg_hls_ls histEq
    # re_rgb_avg_hls_ls = HistEq_rgb_avg_hls_ls(img)
    # cv2.imwrite(save_dir+'color_rgb_avg_hls_ls.jpg',cvtColor(re_rgb_avg_hls_ls, type='HLS2BGR'))


    # img_name = 'histeq1.jpg'
    # img=cv2.imread(dir+img_name,cv2.IMREAD_GRAYSCALE)
    # results = gen_all_HistEq_gray(img)
    # save_all_HistEq_gray([img]+list(results), save_path=save_dir+gen_result_name(img_name, task))

    # img_name = 'histeqColor.jpg'
    # img=cv2.imread(dir+img_name,cv2.IMREAD_COLOR)
    # results = gen_all_HistEq_color(img)
    # save_all_HistEq_color([img]+list(results), save_path=save_dir+gen_result_name(img_name, task))

    run_histeq(dir, save_dir, 'histeq1.jpg', type='gray')
    run_histeq(dir, save_dir, 'histeq2.jpg', type='gray')
    run_histeq(dir, save_dir, 'histeq3.jpg', type='gray')
    run_histeq(dir, save_dir, 'histeq4.jpg', type='gray')
    run_histeq(dir, save_dir, 'histeqColor.jpg', type='color')