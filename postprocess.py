import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
from IPython import embed
from refine import normal2depth

def gradxy2(x):
    h, w = x.shape
    Dx = np.zeros_like(x)
    Dy = np.zeros_like(x)
    Dx[:, 1:-1] = x[:, 2:] - x[:, :-2]
    Dy[1:-1, :] = x[2:, :] - x[:-2, :]
    Dx[:, 0] = Dx[:, 1]
    Dx[:, -1] = Dx[:, -2]
    Dy[0, :] = Dy[1, :]
    Dy[-1, :] = Dy[-2, :]
    return Dx, Dy

def gradxy(x):
    h, w = x.shape
    Dx = np.zeros_like(x)
    Dy = np.zeros_like(x)
    Dx[:, :-1] = x[:, 1:] - x[:, :-1]
    Dy[:-1, :] = x[1:, :] - x[:-1, :]
    Dx[:, -1] = Dx[:, -2]
    Dy[-1, :] = Dy[-2, :]
    return Dx, Dy

def depthFromGrad(depth, Dx, Dy):
    dxy = np.zeros_like(depth)
    dyx = np.zeros_like(depth)
    dxy[0, 0], dyx[0, 0] = depth[0, 0], depth[0, 0]
    for i in range(1, Dx.shape[1]-1):     # first dx, then dy
        dxy[0, i] = dxy[0, i-1] + Dx[0, i-1]
    for j in range(1, Dy.shape[0]-1):
        dxy[j, :] = dxy[j-1, :] + Dy[j-1, :]

    for j in range(1, Dy.shape[0]-1):
        dyx[j, 0] = dyx[j-1, 0] + Dy[j-1, 0]
    for i in range(1, Dx.shape[1]-1):     # first dx, then dy
        dyx[:, i] = dyx[:, i-1] + Dx[:, i-1]

    return dxy, dyx

def depthFromGrad2(depth, Dx, Dy):
    dx = np.zeros_like(depth)
    dy = np.zeros_like(depth)
    dx[:, 0], dy[0, :] = depth[:, 0], depth[0, :]
    for i in range(1, Dx.shape[1]-1):
        dx[:, i] = dx[:, i-1] + Dx[:, i-1]
    for j in range(1, Dy.shape[0]-1):
        dy[j, :] = dy[j-1, :] + Dy[j-1, :]
    return dx, dy

def gradRefineDepth(depth, Dx, Dy, k=3):
    d = np.zeros_like(depth)
    th = int(k/2)
    h, w = depth.shape
    for y in range(h):
        for x in range(w):
            d0 = depth[y,x]
            for yy in range(y-th, y+th+1):
                if yy < 0 or y >= h:
                    continue
                for xx in range(x-th, x+th+1):
                    if xx < 0 or y >= w:
                        continue
                    


# path = '/Users/anshijie/ylab/2.18/landscape/landscape/'

# for root ,subdir, files in os.walk(path):
#     for fname in files:
#         if not fname.endswith('txt'):
#             continue
#         txt_path = os.path.join(root, fname)
#         img_path = txt_path.replace('txt', 'jpg')
#         img = cv2.imread(img_path, 0)
#         depth = np.loadtxt(txt_path)
#         h, w = img.shape
#         depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
#         depth = (255*(depth - depth.min())/(depth.max()-depth.min())).astype('uint8')
#         # embed()
#         depth = cv2.ximgproc.jointBilateralFilter(img, depth, -1, 10, 50)
#         # depth = cv2.ximgproc.amFilter(img, depth, 6, 0.1)
#         plt.imsave(img_path.replace('.jpg', '-am-5-1.png'), depth, cmap='jet', format='png')
#         print('processing: ', fname)

path = '/Users/anshijie/ylab/3/imgs-out/landscape/'

for root ,subdir, files in os.walk(path):
    for fname in files:
        if not fname.endswith('12-d.png'):
            continue
        depth_path = os.path.join(root, fname)
        img_path = depth_path.replace('-d.png', '.png')
        img = cv2.imread(img_path)
        depth = cv2.imread(depth_path, -1)
        h, w, _ = img.shape
        # depth = cv2.resize(depth, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
        # img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        # depth = (255*(depth - depth.min())/(depth.max()-depth.min())).astype('uint8')
        # embed()
        # depth = cv2.ximgproc.jointBilateralFilter(img, depth, -1, 20, 20)
        # depth = cv2.ximgproc.amFilter(img, depth, 6, 0.1)
        # depth = cv2.ximgproc.dtFilter(img, depth, 300, 500)
        # plt.imsave(img_path.replace('.png', '-jbf.png'), depth, cmap='jet', format='png')
        # depth = normal2depth(depth, img, th=3)
        depth = depth.astype('float32')
        Dx, Dy = gradxy2(depth)
        plt.figure(1)
        plt.imshow(Dx)
        plt.set_cmap('jet')
        plt.colorbar()
        plt.figure(2)
        plt.imshow(Dy)
        plt.set_cmap('jet')
        plt.colorbar()
        Dx = cv2.ximgproc.dtFilter(img, Dx, 300, 500)
        Dy = cv2.ximgproc.dtFilter(img, Dy, 300, 500)
        plt.figure(3)
        plt.imshow(Dx)
        plt.set_cmap('jet')
        plt.colorbar()
        plt.figure(4)
        plt.imshow(Dy)
        plt.set_cmap('jet')
        plt.colorbar()
        dxy, dyx = depthFromGrad2(depth, Dx, Dy)
        plt.figure(5)
        plt.imshow(dxy)
        plt.set_cmap('jet')
        plt.colorbar()
        plt.figure(6)
        plt.imshow(dyx)
        plt.set_cmap('jet')
        plt.colorbar()
        plt.figure(7)
        plt.imshow(depth)
        plt.set_cmap('jet')
        plt.colorbar()
        plt.figure(8)
        plt.imshow((dxy+dyx)/2)
        plt.set_cmap('jet')
        plt.colorbar()
        plt.show()
        print('processing: ', fname)
        # cv2.imwrite(img_path.replace('.png', '-dt-100-1000.png'), depth)