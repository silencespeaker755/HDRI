import numpy as np
import argparse
import cv2, os
from toneMapping import ToneMapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blue"   , default="./Inverse_CRF_B.npy")
    parser.add_argument("-g", "--green"  , default="./Inverse_CRF_G.npy")
    parser.add_argument("-r", "--red"    , default="./Inverse_CRF_R.npy")
    parser.add_argument("-d", "--dir"    , default="./DebevecData")
    parser.add_argument("-m", "--method"    , default="global")
    parser.add_argument("-o", "--output" , default="HDR.hdr")
    args = parser.parse_args()

    target_dir = args.dir
    method = args.method

    B = np.load(os.path.join(target_dir, args.blue))
    G = np.load(os.path.join(target_dir, args.green))
    R = np.load(os.path.join(target_dir, args.red))

    hdr = cv2.merge([B, G, R])

    # save hdr image
    output = args.output
    if method == "global":
        ldr = ToneMapping.photographic_global(hdr, a=0.5)
        cv2.imwrite(os.path.join(target_dir, output), ldr)
    if method == "local":
        ldr = ToneMapping.photographic_local(hdr, a=0.7, epsilon=0.01, scale_max=25, p=20.0)
        cv2.imwrite(os.path.join(target_dir, output), ldr)