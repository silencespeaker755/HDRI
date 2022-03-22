import math
import numpy as np
import cv2
from matplotlib import pyplot as plt 
from imageIO import read_images
from toneMapping import ToneMapping
import os
def my_read_images(image_dir): # for reading memorial images (exposure time was not written in properties)
    paths = [os.path.join(image_dir, file) for file in sorted(os.listdir(image_dir)) if os.path.isfile(os.path.join(image_dir, file))]
    LDR_images = []
    for path in paths:
        # read image and append images into LDR list
        img = cv2.imread(path)
        LDR_images.append(img)
        print(path)

    # transform LDR_images into np array
    LDR_images = np.array(LDR_images)
    exposure_time = [0] * 16 # 2^5 ~ 2^-10
    exposure_time[0] = 32
    for i in range(1, 16, 1):
        exposure_time[i] = exposure_time[i-1] / 2
    print(exposure_time)
    return LDR_images, exposure_time

class RobertsonHDR:
    def __init__(self, images, exposure_time, ldr_size):
        self.Z = images
        self.expo_time = exposure_time
        self.ldr_size = ldr_size
        self.height = images.shape[1]
        self.width = images.shape[2]
        self.weight = self.setup_weight(ldr_size)
        plt.plot(np.arange(256), self.weight)
        plt.show()
        self.radianceMaps = np.zeros((3, self.height, self.width))
        self.gCurves = np.full((3, ldr_size), np.nan)

    def setup_weight(self, ldr_size): # referenced from opencv source code
        q = (ldr_size - 1) / 4
        value = np.arange(ldr_size) / q - 2
        return 1 / (math.exp(4) - 1) * (math.exp(4) * np.exp(- np.power(value, 2)) - 1)

    def optimize_E(self, g_func, channel): 
        # Z: images of channel at different time, datatype:list[time][y][x][channel]
        # time_seq: exposure time, datatype:list[time]
        # Ei = sum(w(Zij) * g(Zij) * t) / sum(w(Zij) * t^2)
        print('Start optimize_E')
        Ei = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                sum1 = 0
                sum2 = 0
                for t_idx in range(len(self.expo_time)):
                    z_pixel = self.Z[t_idx][y][x][channel]
                    w = self.weight[z_pixel]
                    t = self.expo_time[t_idx]
                    sum1 += w * g_func[z_pixel] * t
                    sum2 += w * t * t
                Ei[y][x] = sum1 / sum2
        return Ei
       

    def optimize_g(self, E_func, channel):
        # g(z_pixel) = 1/cnt * sum(all pixel's E value * t in time t)
        print('Start optimize_g')
        gm = np.zeros(self.ldr_size) 
        Em = np.zeros((self.ldr_size, 2))
        for t_idx in range(len(self.expo_time)):
            # print('time: {}'.format(time_seq[t_idx]))
            for y in range(self.height):
                for x in range(self.width):
                    # print('pixel: {}, {}: {}'.format(y, x, E_func[y][x]))
                    m = self.Z[t_idx][y][x][channel]
                    Em[m][0] += E_func[y][x] * self.expo_time[t_idx] #sum
                    Em[m][1] += 1 # cnt
        
        for m in range(self.ldr_size):
            gm[m] = Em[m][0] / (Em[m][1] + 1e-8)
        gm /= gm[int(self.ldr_size / 2)] #normalize

        # print(gm)
        return gm

    def solve(self, channel, epoch = 8): #optimize g, Ei
        
        print('\n=====solve channel:{}'.format(channel))
        Ei = np.zeros((self.height, self.width))
        gm = np.arange(self.ldr_size) / self.ldr_size / 2 # initial g function is chosen as a linear function with g(128) = 0
        for i in range(epoch):
            print('\n=====epoch:{}'.format(i))
            Ei = self.optimize_E(gm, channel)
            gm = self.optimize_g(Ei, channel)      

        self.gCurves[channel] = gm
        self.radianceMaps[channel] = Ei
    
    def load_radiance_maps_from_file(self, files):
        channel = 0
        for path in files:
            self.radianceMaps[channel] = np.load(path)
            channel += 1

    def load_gCurves_from_file(self, files):
        channel = 0
        for path in files:
            self.gCurves[channel] = np.load(path)
            channel += 1

    def process_radiance_map(self, epoch = 10):
        for c in range(3):
            if(np.any(np.isnan(self.gCurves[c]))):
                self.solve(c, epoch)
            else:
                self.radianceMaps[c] = self.optimize_E(self.gCurves[c], c)

    def get_HDR_image(self):
        # combine 3 channel Ei values to HDR
        hdr = np.zeros((height, width, 3))
        for c in range(3):
            hdr[:,:,c] = self.radianceMaps[c]
        
        return hdr

if __name__ == '__main__':
    
    # initial g function is chosen as a linear function with g(128) = 0
    # initial_g = np.arange(256) / 128
    # setup weighting function: w(Zij) = exp(-4 * (Zij - 127.5)^2 / 127.5^2)
    # weight = np.exp(-4 *  np.square(np.arange(256) - 127.5) / 127.5 / 127.5)

    # read images: i images of differet exposure time, each with 3 channels (Z[i][y][x][channel])
    # LDR_images, exposure_times = my_read_images('Photos/memorial/')
    LDR_images, exposure_times = read_images('Photos/JPG/')

    height = LDR_images.shape[1]
    width = LDR_images.shape[2]
    #scale down to speed up
    height = (int)(height / 8)
    width = (int)(width / 8)
    print('width:{}, height:{}'.format(width, height))
    LDR_images_quarter = []

    for i in range(len(LDR_images)):
        img = LDR_images[i]
        img = cv2.resize(img, dsize = (width, height), interpolation=cv2.INTER_NEAREST)
        LDR_images_quarter.append(img)
    LDR_images_quarter = np.array(LDR_images_quarter)
    
    rb = RobertsonHDR(LDR_images_quarter, exposure_times, 256)
    dir = 'RobertsonDatas/tiger_40epoch/'
    # rb.process_radiance_map(epoch = 2)
    rb.load_gCurves_from_file([dir + 'gm_b.npy',dir + 'gm_g.npy',dir + 'gm_r.npy' ])
    # rb.load_radiance_maps_from_file([dir + 'Ei_b.npy',dir + 'Ei_g.npy',dir + 'Ei_r.npy' ])
    rb.process_radiance_map()
    channel_str = ['b', 'g', 'r']

    for c in range(3):
        # save Ei
        title = 'Ei_{}'.format(channel_str[c])
        plt.title(title)
        plt.imsave(dir + title + '.png', np.log(rb.radianceMaps[c] + 1e-8), cmap = 'jet')
        np.save(dir + title + '.npy', rb.radianceMaps[c])
        plt.clf()
        #save g curve
        title = 'gm_{}'.format(channel_str[c])
        plt.plot(np.arange(256),np.log(rb.gCurves[c] + 1e-8)) 
        plt.title(title)
        plt.savefig(dir + title + '.png')
        np.save(dir + title + '.npy', rb.gCurves[c])
        plt.clf()

    hdr = rb.get_HDR_image()
    # save hdr image
    cv2.imwrite(dir + 'test.hdr',hdr.astype(np.float32))
    ldr = ToneMapping.photographic_global(hdr, a=0.5)
    cv2.imwrite(dir + 'Ldr_photographic_global.jpg', ldr)
    ldr = ToneMapping.photographic_local(hdr, a=0.7, epsilon=0.01, scale_max=25, p=20.0)
    cv2.imwrite(dir + 'Ldr_photographic_local.jpg', ldr)