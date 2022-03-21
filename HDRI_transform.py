import numpy as np
import argparse
import os
from threading import Thread
from imageIO import read_images

from debevec import Debevec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--image_dir")
    parser.add_argument("-o", "--output", default="output.jpg")
    args = parser.parse_args()

    # for jpg type
    dir_path = os.path.join(args.image_dir, "JPG")
    LDR_images, exposure_times = read_images(dir_path)

    debevec = Debevec(images=LDR_images, exposure_times=exposure_times)
    
    B, G, R = debevec.split_BGR_images()
    image_number, pixels = B.shape

    points = debevec.pick_evaluation_points(pixels, 100)

    irradiance_map = debevec.generate_irradiance_map(sample_points=B[:, points])
    print(irradiance_map)
    # print(np.array(irradiance_map).shape)

    debevec.reconstruct_irradiance_image(B, irradiance_map, "HDR_B.npy").join
    debevec.reconstruct_irradiance_image(G, irradiance_map, "HDR_G.npy").join
    debevec.reconstruct_irradiance_image(R, irradiance_map, "HDR_R.npy").join








    