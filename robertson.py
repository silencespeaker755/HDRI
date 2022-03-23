import math
import numpy as np
import cv2
from threading import Thread

class RobertsonHDR:
    def __init__(self, images, exposure_time, ldr_size):
        self.Z = images
        self.expo_time = exposure_time
        self.ldr_size = ldr_size
        self.height = images.shape[1]
        self.width = images.shape[2]
        self.weight = self.setup_weight(ldr_size)
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

    def threaded(fn):
        def wrapper(*args):
            thread = Thread(target=fn, args=args)
            thread.daemon = True
            return thread
        return wrapper
    
    @threaded
    def solve(self, channel, epoch = 8): #optimize g, Ei
        
        Ei = np.zeros((self.height, self.width))
        gm = np.arange(self.ldr_size) / self.ldr_size / 2 # initial g function is chosen as a linear function with g(128) = 0
      
        for i in range(epoch):
            print('=====channel:{} epoch:{}'.format(channel, i))
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

    def process_radiance_map(self, savefiles, epoch = 10):
        threading_missions = []
        for c in range(3):
            mission = self.solve(c, epoch)
            threading_missions.append(mission)
        
        # execute thread mission
        for mission in threading_missions:
            mission.start()
        
        for mission in threading_missions:
            mission.join()
        
        for c in range(3):        
            np.save(savefiles[c], self.radianceMaps[c])

    def get_HDR_image(self):
        # combine 3 channel Ei values to HDR
        B, G, R = self.radianceMaps
        hdr = cv2.merge([B, G, R])
        
        return hdr