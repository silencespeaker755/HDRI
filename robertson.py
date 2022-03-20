import numpy as np
import cv2
from matplotlib import pyplot as plt 
from imageIO import read_images

def optimize_E(Z, height, width, g_func, weight, expo_time, channel): 
    # Z: images of channel at different time, datatype:list[time][y][x][channel]
    # time_seq: exposure time, datatype:list[time]
    # Ei = sum(w(Zij) * g(Zij) * t) / sum(w(Zij) * t^2)
    print('Start optimize_E')
    # print(type(height))
    Ei = np.zeros((height, width)) 
    for y in range(height):
        for x in range(width):
            sum1 = 0
            sum2 = 0
            for t_idx in range(len(expo_time)):
                z_pixel = Z[t_idx][y][x][channel]
                w = weight[z_pixel]
                t = expo_time[t_idx]
                sum1 += w * g_func[z_pixel] * t
                sum2 += w * t * t
            Ei[y][x] = sum1 / sum2
    # print('Ei:', Ei)
   
    return Ei

def optimize_g(Z, height, width, E_func, expo_time, channel):
    # g(z_pixel) = 1/cnt * sum(all pixel's E value * t in time t)
    print('Start optimize_g')
    gm = np.zeros(256) 
    Em = np.zeros((256, 2))
    for t_idx in range(len(expo_time)):
        # print('time: {}'.format(time_seq[t_idx]))
        for y in range(height):
            for x in range(width):
                # print('pixel: {}, {}: {}'.format(y, x, E_func[y][x]))
                m = Z[t_idx][y][x][channel]
                Em[m][0] += E_func[y][x] * expo_time[t_idx] #sum
                Em[m][1] += 1 # cnt

    for m in range(256):
        if(Em[m][1] != 0): gm[m] = Em[m][0] / Em[m][1]
    
    gm /= gm[128] #normalize

    # print(gm)
    return gm

import os
def my_read_images(image_dir): # for reading memorial images (exposure time was not written in properties)
    paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]
    LDR_images = []
    for path in paths:
        # read image and append images into LDR list
        img = cv2.imread(path)
        LDR_images.append(img)

    # transform LDR_images into np array
    LDR_images = np.array(LDR_images)
    exposure_time = [0] * 16 # 2^5 ~ 2^-10
    exposure_time[0] = 32
    for i in range(1, 16, 1):
        exposure_time[i] = exposure_time[i-1] / 2
    
    return LDR_images, exposure_time

if __name__ == '__main__':
   
    # initial g function is chosen as a linear function with g(128) = 0
    initial_g = np.arange(256) / 128
    # setup weighting function: w(Zij) = exp(-4 * (Zij - 127.5)^2 / 127.5^2)
    weight = np.exp(-4 *  np.square(np.arange(256) - 127.5) / 127.5 / 127.5)
    plt.plot(np.arange(256), initial_g) 
    plt.title('Initial g')
    plt.savefig('RobertsonData/initial_g.png')
    plt.clf()
    
    plt.plot(np.arange(256), weight)
    plt.title('Weight')
    plt.savefig('RobertsonData/weight.png') 
    plt.clf()
    # read images: i images of differet exposure time, each with 3 channels (Z[i][y][x][channel])
    LDR_images, exposure_times = my_read_images('Photos/memorial/')
    # LDR_images, exposure_times = read_images('Photos/JPG/')
    print(LDR_images.shape)
    
    height = LDR_images.shape[1]
    width = LDR_images.shape[2]
    #scale down to speed up
    # height = (int)(height / 8)
    # width = (int)(width / 8)
    # print('width:{}, height:{}'.format(width, height))
    # LDR_images_quarter = []

    # for i in range(len(LDR_images)):
    #     img = LDR_images[i]
    #     img = cv2.resize(img, dsize = (width, height), interpolation=cv2.INTER_NEAREST)
    #     LDR_images_quarter.append(img)
        
    epoch = 8
    Ec = np.zeros((3, height, width))

    channel_str = ['b', 'g', 'r']
    for c in range(3):
        print('\n=====channel:{}'.format(c))
        Ei = np.zeros((height, width)) 
        gm = initial_g #first epoch: use initial g
        for i in range(epoch):
            print('\n=====epoch:{}'.format(i))
            Ei = optimize_E(LDR_images, height, width, gm, weight, exposure_times, channel = c)
            gm = optimize_g(LDR_images, height, width, Ei, exposure_times, channel = c)      

        # save Ei
        title = 'Ei_{}'.format(channel_str[c])
        plt.title(title)
        plt.imsave('RobertsonData/' + title + '.png', Ei, cmap = 'jet')
        np.savetxt('RobertsonData/' + title + '.txt', Ei)
        plt.clf()

        #save g curve
        title = 'gm_{}'.format(channel_str[c])
        plt.plot(np.arange(256),gm) 
        plt.title(title)
        plt.savefig('RobertsonData/' + title + '.png')
        np.savetxt('RobertsonData/' + title + '.txt', gm)
        plt.clf()
        
        # put Ei into channel
        Ec[c] = Ei
    
    # combine 3 channel Ei values to HDR
    hdr = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            for c in range(3):
               hdr[y][x][c] = Ec[c][y][x]

    # save hdr image
    cv2.imwrite('test.hdr',hdr.astype(np.float32))


