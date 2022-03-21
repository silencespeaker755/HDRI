import numpy as np
from matplotlib import pyplot as plt 
import cv2

def photographic_global(hdr, a=0.3):
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
    
if __name__ == '__main__':
    # combine 3 channel Ei values to HDR
    Ec = []
    dir = 'RobertsonData_tiger/'
    Ec.append(np.loadtxt(dir+'Ei_b.txt'))
    Ec.append(np.loadtxt(dir+'Ei_g.txt'))
    Ec.append(np.loadtxt(dir+'Ei_r.txt'))

    hdr = np.zeros((Ec[0].shape[0], Ec[0].shape[1], 3))
    for c in range(3):
        hdr[:,:,c] = Ec[c]
    # img = cv2.imread('test.hdr')
    ldr = photographic_global(hdr, a=0.7)
    # ldr = Reinhard_global(hdr)
    cv2.imwrite(dir + 'ldr.jpg', ldr)
