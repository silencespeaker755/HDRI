import numpy as np
from matplotlib import pyplot as plt 
import cv2
import math

class ToneMapping:

    def photographic_global(hdr, a=0.18):
        Lw = 0.06 * hdr[:,:,0] + 0.67 * hdr[:,:,1] + 0.27 * hdr[:,:,2]
        Lw_average = np.exp(np.mean(np.log(1e-8 + Lw)))
        Lm = a / Lw_average * Lw
        Lwhite = np.max(Lm)
        Ld = Lm * (1 + (Lm / (Lwhite * Lwhite))) / (1 + Lm)

        ldr = np.zeros(hdr.shape)
        for c in range(3):
            ldr[:,:,c] = hdr[:,:,c] * Ld / Lw

        ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
        return ldr

    def photographic_local(hdr, a=0.18, p=8.0, scale_max=8, epsilon=0.05):
        # Ld = Lm / (1 + V1)
        # Vi(x,y,s) = Gaussian(Lm, (s,s))
        # V(x,y,s) = (V1 - V2) / (2^p * a / s / s + V1)
        Lw = 0.06 * hdr[:,:,0] + 0.67 * hdr[:,:,1] + 0.27 * hdr[:,:,2]
        Lw_average = np.exp(np.mean(np.log(1e-8 + Lw)))
        Lm = a / Lw_average * Lw
        center_surround_ratio = 1.6
        # s2 = s1 * center_surround_ratio
        # calculate all V(x,y,s) for s = 1~scale_max
            # find sm where |V(x,y,s)| < epsilon
        s = np.full(scale_max, center_surround_ratio)
        s_power = np.arange(scale_max) #s=[0,1,2,3,...7]
        s = 2 * np.ceil(np.power(s, s_power)).astype(np.uint8) + 1
        Vm = Ld = np.zeros(Lm.shape)
        done = np.full(Lm.shape, False)
        for i in range(0, scale_max):
            V1 = cv2.GaussianBlur(Lm, (s[i], s[i]), 0)
            if(i == 0): V1 = Lm
            if(i == scale_max - 1):
                update = np.where(done == False)
                Vm[update] = V1[update]
                break
            V2 = cv2.GaussianBlur(Lm, (s[i+1], s[i+1]), 0)
            V = (V1 - V2) / ((2 ** p) * a / s[i] / s[i] + V1)
            # if V < epsilon and not done yet: update 
            update = np.where(np.logical_and(np.abs(V) >= epsilon, np.logical_not(done)))
            Vm[update] = V1[update]
            done[update] = True

        Ld = Lm / (1 + Vm)
        ldr = np.zeros(hdr.shape)
        for c in range(3):
            ldr[:,:,c] = hdr[:,:,c] * Ld / Lw

        ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
        return ldr
