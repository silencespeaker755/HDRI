import numpy as np
import argparse
import os
from utils import draw_irradiance_map
from imageIO import read_images

from debevec import Debevec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir")
    parser.add_argument("-d", "--save_dir", default="./DebevecData")
    args = parser.parse_args()

    # for jpg type
    dir_path = os.path.join(args.source_dir)
    LDR_images, exposure_times = read_images(dir_path)

    debevec = Debevec(images=LDR_images, exposure_times=exposure_times)
    
    B, G, R = debevec.split_BGR_images()
    image_number, pixels = B.shape

    points = debevec.pick_evaluation_points(pixels, 100)

    # get report's data store directory
    target_dir = args.save_dir
    irradiance_map_B = debevec.generate_irradiance_map(sample_points=B[:, points])
    draw_irradiance_map(irradiance_map_B, os.path.join(target_dir , "Irradiance_B.png"), color="blue")

    irradiance_map_G = debevec.generate_irradiance_map(sample_points=G[:, points])
    draw_irradiance_map(irradiance_map_G, os.path.join(target_dir , "Irradiance_G.png"), color="green")

    irradiance_map_R = debevec.generate_irradiance_map(sample_points=R[:, points])
    draw_irradiance_map(irradiance_map_R, os.path.join(target_dir , "Irradiance_R.png"), color="red")


    debevec.reconstruct_irradiance_image(B, irradiance_map_B, "HDR_B.npy").join
    debevec.reconstruct_irradiance_image(G, irradiance_map_G, "HDR_G.npy").join
    debevec.reconstruct_irradiance_image(R, irradiance_map_R, "HDR_R.npy").join








    