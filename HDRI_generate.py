import numpy as np
import argparse
import cv2
from imageIO import save_HDR_images
# from matplotlib import pyplot as plt

def combine_BGR_files(B, G, R, output):
    HDR_image = cv2.merge([B, G, R])
    print(HDR_image)
    save_HDR_images(HDR_image, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blue", default="./HDR_B.npy")
    parser.add_argument("-g", "--green", default="./HDR_G.npy")
    parser.add_argument("-r", "--red", default="./HDR_R.npy")
    parser.add_argument("-o", "--output", default="HDR.hdr")
    args = parser.parse_args()

    B = np.load(args.blue)
    G = np.load(args.green)
    R = np.load(args.red)

    B = np.exp(np.log2(B))
    G = np.exp(np.log2(G))
    R = np.exp(np.log2(R))

    combine_BGR_files(B.reshape(3456, 4608), G.reshape(3456, 4608), R.reshape(3456, 4608), args.output)
