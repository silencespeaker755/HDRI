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
    parser.add_argument("-b", "--blue"   , default="./Inverse_CRF_B.npy")
    parser.add_argument("-g", "--green"  , default="./Inverse_CRF_G.npy")
    parser.add_argument("-r", "--red"    , default="./Inverse_CRF_R.npy")
    parser.add_argument("-d", "--dir"    , default="./DebevecData")
    parser.add_argument("-o", "--output" , default="HDR.hdr")
    args = parser.parse_args()

    target_dir = args.dir
    B = np.load(os.path.join(target_dir, args.blue))
    G = np.load(os.path.join(target_dir, args.green))
    R = np.load(os.path.join(target_dir, args.red))

    # draw radiance map according to each color's data
    dict = {"blue": B, "green": G, "red": R}
    draw_radiance_map(image_dict=dict, store_dir=args.dir)

    store_path = os.path.join(target_dir, args.output)
    combine_save_BGR_files(B, G, R, store_path)
