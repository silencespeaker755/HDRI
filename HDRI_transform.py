import numpy as np
import argparse
import cv2, os
from toneMapping import ToneMapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blue"   , default="DebevecData/Radiance_B.npy")
    parser.add_argument("-g", "--green"  , default="DebevecData/Radiance_G.npy")
    parser.add_argument("-r", "--red"    , default="DebevecData/Radiance_R.npy")
    parser.add_argument("-d", "--dir"    , default="DebevecData")
    parser.add_argument("-m", "--method"    , default="global")
    parser.add_argument("-o", "--output" , default="LDR.png")
    args = parser.parse_args()

    target_dir = args.dir
    method = args.method

    B = np.load(args.blue)
    G = np.load(args.green)
    R = np.load(args.red)

    hdr = cv2.merge([B, G, R])

    # save hdr image
    output = args.output
    if method == "global":
        ldr = ToneMapping.photographic_global(hdr, a=0.5)
        cv2.imwrite(os.path.join(target_dir, output), ldr)
    if method == "local":
        ldr = ToneMapping.photographic_local(hdr, a=0.7, epsilon=0.01, scale_max=25, p=20.0)
        cv2.imwrite(os.path.join(target_dir, output), ldr)
