import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

PI = 3.1415926535

def cvtColor(img, type='HLS2BGR'):
    if type == 'HLS2BGR':
        a, b, _ = img.shape
        img_hls = img.astype(float)
        H, I, S = img_hls[:,:,0]*2*(2*PI)/360, img_hls[:,:,1]/255, img_hls[:,:,2]/255
        B, G, R = np.zeros((a,b), dtype=float), np.zeros((a,b), dtype=float), np.zeros((a,b), dtype=float)

        ### This is a FAST implementation using vectorization!
        cond_1 = np.where(H<2*PI/3)
        R[cond_1] = I[cond_1]*(1+S[cond_1]*np.cos(H[cond_1])/np.cos(2*PI/6-H[cond_1]))
        B[cond_1] = I[cond_1]*(1-S[cond_1])
        G[cond_1] = 3*I[cond_1]-(R[cond_1]+B[cond_1])
        cond_2 = np.where(np.logical_and(H>=2*PI/3, H<4*PI/3))
        G[cond_2] = I[cond_2]*(1+S[cond_2]*np.cos(H[cond_2]-2*PI/3)/np.cos(2*PI/6-H[cond_2]+2*PI/3))
        R[cond_2] = I[cond_2]*(1-S[cond_2])
        B[cond_2] = 3*I[cond_2]-(R[cond_2]+G[cond_2])
        cond_3 = np.where(np.logical_and(H>=4*PI/3, H<6*PI/3))
        B[cond_3] = I[cond_3]*(1+S[cond_3]*np.cos(H[cond_3]-4*PI/3)/np.cos(2*PI/6-H[cond_3]+4*PI/3))
        G[cond_3] = I[cond_3]*(1-S[cond_3])
        R[cond_3] = 3*I[cond_3]-(G[cond_3]+B[cond_3])

        # ### This is a SLOW implementation using loops!
        # for i in range(a):
        #     for j in range(b):
        #         assert I[i,j]<=1 and I[i,j]>=0 and S[i,j]<=1 and S[i,j]>=0 and H[i,j]>=0 and H[i,j]<2*PI
        #         if H[i,j] < 2*PI/3:
        #             R[i,j] = I[i,j]*(1+S[i,j]*np.cos(H[i,j])/np.cos(2*PI/6-H[i,j]))
        #             B[i,j] = I[i,j]*(1-S[i,j])
        #             G[i,j] = 3*I[i,j]-(R[i,j]+B[i,j])
        #         elif H[i,j] < 4*PI/3:
        #             G[i,j] = I[i,j]*(1+S[i,j]*np.cos(H[i,j]-2*PI/3)/np.cos(2*PI/6-H[i,j]+2*PI/3))
        #             R[i,j] = I[i,j]*(1-S[i,j])
        #             B[i,j] = 3*I[i,j]-(R[i,j]+G[i,j])
        #         elif H[i,j] < 6*PI/3:
        #             B[i,j] = I[i,j]*(1+S[i,j]*np.cos(H[i,j]-4*PI/3)/np.cos(2*PI/6-H[i,j]+4*PI/3))
        #             G[i,j] = I[i,j]*(1-S[i,j])
        #             R[i,j] = 3*I[i,j]-(G[i,j]+B[i,j])

        B, G, R = np.clip(B,0,1), np.clip(G,0,1), np.clip(R,0,1)
        img_bgr = np.zeros((a,b,3), dtype=np.uint8)
        img_bgr[:,:,0], img_bgr[:,:,1], img_bgr[:,:,2] = B*255, G*255, R*255
        return img_bgr

    elif type == 'BGR2HLS':
        a, b, _ = img.shape
        img_bgr = img.astype(float)/255
        B, G, R = img_bgr[:,:,0], img_bgr[:,:,1], img_bgr[:,:,2]
        H, I, S = np.zeros((a,b), dtype=float), np.zeros((a,b), dtype=float), np.zeros((a,b), dtype=float)

        ### This is a FAST implementation using vectorization!
        # Calculate I
        RGB_sum = R+G+B
        I = 1/3*RGB_sum
        # Calculate S
        RGB_min = np.min(np.stack((R,G,B)), axis = 0)
        cond_S=np.where(RGB_sum!=0)
        S[cond_S] = 1-3/RGB_sum[cond_S]*RGB_min[cond_S]
        # Calculate H
        val = ((R-G)**2+(R-B)*(G-B))**0.5
        cond_H = np.where(val!=0)
        theta = np.arccos(np.clip(1/2*(R-G+R-B)[cond_H]/val[cond_H], a_min=-1, a_max=1))*360/(2*PI)
        cond_H2 = B[cond_H] > G[cond_H]
        theta[cond_H2] = 360 - theta[cond_H2]
        H[cond_H] = theta

        # ### This is a SLOW implementation using loops!
        # for i in range(a):
        #     for j in range(b):
        #         I[i,j] = 1/3*(R[i,j]+G[i,j]+B[i,j])
        #         if R[i,j]+G[i,j]+B[i,j] != 0:
        #             S[i,j] = 1-3/(R[i,j]+G[i,j]+B[i,j])*(min(R[i,j],G[i,j],B[i,j]))
        #         val = ((R[i,j]-G[i,j])**2+(R[i,j]-B[i,j])*(G[i,j]-B[i,j]))**0.5
        #         if val != 0:
        #             theta_val = np.arccos(np.clip(1/2*(R[i,j]-G[i,j]+R[i,j]-B[i,j])/val, a_min=-1, a_max=1))*360/(2*PI)
        #             if B[i,j] <= G[i,j]:
        #                 H[i,j] = theta_val
        #             else:
        #                 H[i,j] = 360 - theta_val

        img_hls = np.zeros((a,b,3), dtype=np.uint8)
        img_hls[:,:,0], img_hls[:,:,1], img_hls[:,:,2] = H/2, I*255, S*255
        return img_hls

# utils for display
def add_border(img):
    # add white thin border
    if len(img.shape)==2:
        return cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=255)
    else:
        return cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=[255,255,255])

def resize_img(img):
    # resize the final image to a proper size for showing in the webpage.
    # scale
    height_max = 1300
    width_max = 1300
    shape_now = img.shape
    print(shape_now)
    scale = min(height_max / shape_now[0], width_max / shape_now[1])

    # resize image
    height = int(shape_now[0] * scale)
    width = int(shape_now[1] * scale)
    shape_resize = (width, height)
    return cv2.resize(img, shape_resize, interpolation=cv2.INTER_AREA)

def resize_img_width(img, width):
    # resize the final image to a proper size for showing in the webpage.
    # scale
    shape_now = img.shape
    scale = width / shape_now[1]

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
        return cv2.putText(img,text,(0,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.3,color= 0,thickness=3,lineType=cv2.LINE_AA)
    else:
        return cv2.putText(img, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.3,color= (0,0,0),thickness=3,lineType=cv2.LINE_AA)
