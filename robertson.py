import numpy as np
import cv2
from matplotlib import pyplot as plt 
from imageIO import read_images

def optimize_E(Z, height, width, g_func, weight, time_seq, channel): 
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
            for t_idx in range(len(time_seq)):
                z_pixel = Z[t_idx][y][x][channel]
                w = weight[z_pixel]
                t = time_seq[t_idx]
                sum1 += w * g_func[z_pixel] * t
                sum2 += w * t * t
            Ei[y][x] = sum1 / sum2
    # print('Ei:', Ei)
   
    return Ei

def optimize_g(Z, height, width, E_func, time_seq, channel):
    # g(z_pixel) = 1/cnt * sum(all pixel's E value * t in time t)
    # print('3:\n', E_func)
    # for y in range(height):
    #     for x in range(width):
    #         if(E_func[y][x] < 0):
                # print(E_func[y][x])
    print('Start optimize_g')
    gm = np.zeros(256) 
    for m in range(256):
        # print('\n====find g({})'.format(m))
        sum = 0
        cnt = 0
        for t_idx in range(len(time_seq)):
            # print('time: {}'.format(time_seq[t_idx]))
            for y in range(height):
                for x in range(width):
                    # print('pixel: {}, {}: {}'.format(y, x, E_func[y][x]))
                    if(m == Z[t_idx][y][x][channel]):
                        cnt += 1
                        sum += E_func[y][x] * time_seq[t_idx]
                        # print('add sum:{} * {} = {}'.format(E_func[y][x], time_seq[t_idx], sum))
        # print('cnt:{}, sum:{}'.format(cnt, sum))
        # input()
        if(cnt == 0): gm[m] = 0
        else:    gm[m] = sum / cnt
        print('g{}: {}'.format(m, gm[m]))
    
    print(gm)
    return gm

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
    LDR_images, exposure_times = read_images('Photos/JPG/')
    print(LDR_images.shape)

    width = (int)(LDR_images.shape[2] / 8)
    height = (int)(LDR_images.shape[1] / 8)
    print('width:{}, height:{}'.format(width, height))
    LDR_images_quarter = []

    for i in range(len(LDR_images)):
        img = LDR_images[i]
        img = cv2.resize(img, dsize = (width, height), interpolation=cv2.INTER_NEAREST)
        LDR_images_quarter.append(img)
    # plt.imshow(LDR_images_quarter[0][:,:,::-1])

    epoch = 10
    for c in range(3):
        print('\n=====channel:{}'.format(c))
        Ei = np.zeros((height, width)) 
        gm = initial_g #first epoch: use initial g
        for i in range(epoch):
            print('\n=====epoch:{}'.format(i))

            Ei = optimize_E(LDR_images_quarter, height, width, gm, weight, exposure_times, channel = c)
            title = 'Ei_{}_c{}'.format(i, c)
            plt.title(title)
            plt.imsave('RobertsonData/' + title + '.png', Ei)
            np.savetxt('RobertsonData/' + title + '.txt', Ei)
            plt.clf()

            gm = optimize_g(LDR_images_quarter, height, width, Ei, exposure_times, channel = c)
            title = 'gm_{}_c{}'.format(i, c)
            plt.plot(np.arange(256),gm) 
            plt.title(title)
            plt.savefig('RobertsonData/' + title + '.png')
            np.savetxt('RobertsonData/' + title + '.txt', gm)
            plt.clf()
            

