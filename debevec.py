import numpy as np
import cv2
import random
from threading import Thread
from tqdm import tqdm

class Debevec:
    def __init__(self, images, exposure_times) -> None:
        self.images = images
        self.exposure_times = exposure_times
        self.shape = images[0].shape[:2]

    def weighting(self, pixel, min=1, max=254):
        # calculate weight coefficient according to pixel
        if pixel > (min+max)/2:
            return abs(max - pixel)
        return abs(pixel - min)

    def split_BGR_images(self):
        B = []
        G = []
        R = []

        for image in self.images:
            b, g, r = cv2.split(image)

            # flat the whole 2d array into 1d array
            B.append(np.array(b).reshape(-1))
            G.append(np.array(g).reshape(-1))
            R.append(np.array(r).reshape(-1))

        return np.array(B), np.array(G), np.array(R)
    
    def pick_evaluation_points(self, total_number, select_number):
        self.select_point = random.sample(range(total_number), select_number)
        return self.select_point

    def generate_inverse_response_curve(self, sample_points):
        # initialize the matrix
        image_number, n = sample_points.shape
        A = np.zeros(shape=(n*image_number+1+254, 256+n))
        B = np.zeros(A.shape[0])

        exposure_time_ln = [np.log(p) for p in self.exposure_times]

        current_row = 0
        for i in range(image_number):
            delta_time = exposure_time_ln[i]
            for j in range(n):
                point = sample_points[i][j]
                weight_pixel = self.weighting(point)
                A[current_row, point] = weight_pixel
                A[current_row, 256+j] = -weight_pixel
                B[current_row] = weight_pixel * delta_time

                #update current row index
                current_row += 1
        
        # fix the curve by setting middle point's value to 0
        A[current_row, 127] = 1
        current_row += 1

        # delta time section
        for i in range(254):
            weight_pixel = self.weighting(i+1)
            A[current_row, i]   = weight_pixel
            A[current_row, i+1] = weight_pixel * -2
            A[current_row, i+2] = weight_pixel

            current_row += 1

        x = np.linalg.lstsq(A, B, rcond=None)[0]
    
        return x[:256]
    
    def threaded(fn):
        def wrapper(*args):
            thread = Thread(target=fn, args=args)
            thread.daemon = True
            return thread
        return wrapper
    
    @threaded
    def reconstruct_irradiance_image(self, image, irradiance_map, output):
        exposure_time_ln = [np.log(p) for p in self.exposure_times]
        

        image = image.T
        pixels, samples = image.shape
        result = []
        with tqdm(total=pixels) as pbar: 
            for i in range(pixels):
                total_Ei = 0
                total_weight = 0
                for j in range(samples):
                    ei = image[i][j]
                    weight_ei = self.weighting(ei)
                    total_Ei += weight_ei * (irradiance_map[ei] - exposure_time_ln[j])
                    total_weight += weight_ei

                result.append(total_Ei / total_weight)
                pbar.update(1)
        
        result = np.array(np.exp(result))
        
        np.save(output, result.reshape(self.shape))
