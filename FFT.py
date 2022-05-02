import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def FFT(img,mode,save_path):
    # if mode='color',only BGR mode is accepted.
    if mode=='gray':
        f = np.fft.fft2(img)
        img_back = np.uint8(np.clip(np.abs(np.fft.ifft2(f)), 0, 255))
        img_error = np.abs(img_back + 0. - img)
        spectrogram = np.uint8(np.clip(20. * np.log(1 + np.abs(np.fft.fftshift(f))), 0, 255))
        plt.ioff()
        plt.figure(figsize=(10,6))
        plt.subplot(221), plt.imshow(img, 'gray', vmin=0, vmax=255), plt.title('original image'), plt.axis('off')
        plt.subplot(222), plt.imshow(spectrogram, 'gray', vmin=0, vmax=255), plt.title('spectrogram'), plt.axis('off')
        plt.subplot(223), plt.imshow(img_back, 'gray', vmin=0, vmax=255), plt.title('restored image'), plt.axis('off')
        plt.subplot(224), plt.imshow(np.uint8(np.clip(img_error, 0, 255)), 'gray', vmin=0, vmax=255), \
        plt.title('max of error={:.0f}'.format(np.max(img_error))), \
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # spectrogram for fft
        spectrogram = np.zeros_like(img)
        f = [0] * 3
        for i in range(3):
            f[i] = np.fft.fft2(img[:, :, i])
            spectrogram[:, :, i] = np.uint8(np.clip(20. * np.log(1 + np.abs(np.fft.fftshift(f[i]))), 0, 255))
        # restore with ifft
        img_back = np.zeros_like(img)
        for i in range(3):
            img_back[:, :, i] = np.uint8(np.clip(np.abs(np.fft.ifft2(f[i])), 0, 255))
        # differences between original images and restored images
        img_error = np.abs(img_back + 0. - img)
        plt.ioff()
        plt.figure(figsize=(10,6))
        plt.subplot(221), plt.imshow(img, vmin=0, vmax=255), plt.title('original image'), plt.axis('off')
        plt.subplot(222), plt.imshow(spectrogram, vmin=0, vmax=255), plt.title('spectrogram'), plt.axis('off')
        plt.subplot(223), plt.imshow(img_back, vmin=0, vmax=255), plt.title('restored image'), plt.axis('off')
        plt.subplot(224), plt.imshow(np.uint8(np.clip(img_error, 0, 255)), vmin=0, vmax=255), \
        plt.title('max of error={:.0f}'.format(np.max(img_error))), \
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
















