import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
from IPython import embed

from get_soft_normal import get_normal_vector

fx = 592.750635
fy = 592.750635
cx = 480
cy = 270
downSize = 1
path = '/Users/anshijie/ylab/3/imgs-out/landscape/'

def padding(x):
    h, w, c = x.shape
    y = np.zeros((h + 2, w + 2, c), dtype=x.dtype)
    y[1:-1, 1:-1] = x
    y[0, :] = y[1, :]
    y[-1, :] = y[-2, :]
    y[:, 0] = y[:, 1]
    y[:, -1] = y[:, -2]
    return y

def depth2mesh(depth):
    h, w = depth.shape
    mesh = np.zeros((h, w, 3), dtype=depth.dtype)
    DirectX = np.zeros(depth.shape, dtype=depth.dtype)
    DirectY = np.zeros(depth.shape, dtype=depth.dtype)
    for i in range(DirectX.shape[1]):
        DirectX[:, i] = i
    for j in range(DirectY.shape[0]):
        DirectY[j, :] = j
    mesh[:, :, 0] = (DirectX - cx) / fx
    mesh[:, :, 1] = (DirectY - cy) / fy
    mesh[:, :, 2] = 1
    mesh = mesh * depth.reshape(h, w, -1)
    return mesh

def normal2depth(depth, img, th=3, iteration=2, sigma=0.95):
    h, w = depth.shape
    mesh = depth2mesh(depth)
    normal_vector = get_normal_vector(mesh)
    normal_vector = normal_vector / (np.sqrt(np.sum(normal_vector ** 2, axis=2, keepdims=True)) + 1e-32)
    normal_vector = padding(normal_vector)
    d = np.zeros_like(depth)
    for n in range(iteration):
        for yi in range(h):
            for xi in range(w):
                s = 0
                t = 0
                for yj in range(yi - th, yi + th + 1):
                    if yj < 0 or yj >= h:
                        continue
                    for xj in range(xi - th, xi + th + 1):
                        if xj < 0 or xj >= w:
                            continue
                        nix, niy, niz = normal_vector[yi, xi][0], normal_vector[yi, xi][1], normal_vector[yi, xi][2]
                        njx, njy, njz = normal_vector[yj, xj][0], normal_vector[yj, xj][1], normal_vector[yj, xj][2]
                        coef = nix * njx + niy * njy + niz * njz
                        pxj, pyj, pzj = mesh[yj, xj, 0], mesh[yj, xj, 1], mesh[yj, xj, 2]
                        a = mesh[yi, xi] / depth[yi, xi]
                        dd = (nix * pxj + niy * pyj + niz * pzj) / (a[0] * nix + a[1] * niy + a[2] * niz)
                        if np.abs(coef) > sigma:
                            s += np.abs(coef)
                            t += np.abs(coef) * dd
                try:
                    d[yi, xi] = t / s
                except:
                    embed()
    return d


for root ,subdir, files in os.walk(path):
    for fname in files:
        if not fname.endswith('txt'):
            continue
        txt_path = os.path.join(root, fname)
        img_path = txt_path.replace('txt', 'jpg')
        img = cv2.imread(img_path)
        depth = np.loadtxt(txt_path)
        depth = normal2depth(depth, img)
        d = normal2depth(depth, img, th=3)
        out_path = img_path.replace('.png', '-r.png')
        plt.imsave(out_path, d, cmap='jet', format='png')
        cv2.imwrite(img_path.replace('.png', '-rr.png'), d.astype('uint8'))
        print('Saving: ', img_path)
