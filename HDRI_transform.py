import numpy as np
import argparse
import os
from imageIO import read_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--image_dir")
    parser.add_argument("-o", "--output", default="output.jpg")
    args = parser.parse_args()

    # for jpg type
    dir_path = os.path.join(args.image_dir, "JPG")
    LDR_images, exposure_times = read_images(dir_path)




    