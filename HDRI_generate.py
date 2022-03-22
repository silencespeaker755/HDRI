import numpy as np
import argparse
import cv2, os
from imageIO import save_HDR_images
from matplotlib import pyplot as plt
from utils import draw_radiance_map

def combine_save_BGR_files(B, G, R, output):
    HDR_image = cv2.merge([B, G, R])
    save_HDR_images(HDR_image, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blue"   , default="./DebevecData/Radiance_B.npy")
    parser.add_argument("-g", "--green"  , default="./DebevecData/Radiance_G.npy")
    parser.add_argument("-r", "--red"    , default="./DebevecData/Radiance_R.npy")
    parser.add_argument("-d", "--dir"    , default="./DebevecData")
    parser.add_argument("-o", "--output" , default="HDR.hdr")
    args = parser.parse_args()

    target_dir = args.dir
    B = np.load(args.blue)
    G = np.load(args.green)
    R = np.load(args.red)

    # draw radiance map according to each color's data
    dict = {"blue": B, "green": G, "red": R}
    draw_radiance_map(image_dict=dict, store_dir=target_dir)

    store_path = os.path.join(target_dir, args.output)
    combine_save_BGR_files(B, G, R, store_path)
