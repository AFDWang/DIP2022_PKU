import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def GaussianFrequencyFilter(img, sigma):
    height, width = img.shape

    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)

    # for i in range(height):
    #     for j in range(width):
    #         fft[i, j] *= np.exp(-((i - (height - 1) / 2) ** 2 + (j - (width - 1) / 2) ** 2) / 2 / sigma ** 2)

    h_v=np.exp(-((np.arange(height) - (height - 1) / 2) ** 2 ) / 2 / sigma ** 2)
    w_v=np.exp(-((np.arange(width) - (width - 1) / 2) ** 2) / 2 / sigma ** 2)
    weights=np.outer(h_v,w_v)
    fft=np.multiply(fft,weights)



    fft = np.fft.ifftshift(fft)
    fft = np.fft.ifft2(fft)

    fft = np.real(fft)
    img_filter=np.uint8(np.clip(fft, 0, 255))

    return img_filter

def Filter(img,mode,sigma):
    if mode=='gray':
        return GaussianFrequencyFilter(img,sigma)
    else:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        img[:,:,1]=GaussianFrequencyFilter(img[:,:,1],sigma)
        img=cv2.cvtColor(img,cv2.COLOR_HLS2BGR)
        return img

def img_generate(img,mode):
    sigmas=[5, 15, 30, 80, 230]
    plt.ioff()
    fig = plt.figure(frameon=False,figsize=(10, 6))


    if mode=='gray':
        plt.subplot(2, 3, 1), plt.imshow(img, mode), plt.title('original image'), plt.axis('off')
    else:
        plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)), plt.title('original image'), plt.axis('off')
    for i, sigma in enumerate(sigmas):
        img_filter = Filter(img, mode, sigma)
        if mode=='gray':
            plt.subplot(2, 3, i + 2), plt.imshow(img_filter, 'gray'), \
            plt.title('sigma={:.0f}'.format(sigma)), plt.axis('off')
        else:
            plt.subplot(2, 3, i + 2), plt.imshow(cv2.cvtColor(img_filter,cv2.COLOR_BGR2RGB) ), \
            plt.title('sigma={:.0f}'.format(sigma)), plt.axis('off')

    plt.tight_layout()
    plt.close()
    return fig












