import numpy as np
import argparse
import cv2, os
from toneMapping import ToneMapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blue"   , default="RobertsonData/tiger_final/Radiance_B.npy")
    parser.add_argument("-g", "--green"  , default="RobertsonData/tiger_final/Radiance_G.npy")
    parser.add_argument("-r", "--red"    , default="RobertsonData/tiger_final/Radiance_R.npy")
    parser.add_argument("-d", "--dir"    , default="RobertsonData/tiger_final/")
    parser.add_argument("-m", "--method"    , default="global")
    parser.add_argument("-o", "--output" , default="LDR.png")
    parser.add_argument("-a", "--a" , default=0.7)
    parser.add_argument("-e", "--epsilon" , default=0.01)
    parser.add_argument("-s", "--scale" , default=20)
    parser.add_argument("-p", "--phi" , default=20)
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
        ldr = ToneMapping.photographic_global(hdr, a=float(args.a))
        cv2.imwrite(os.path.join(target_dir, output), ldr)
    if method == "local":
        ldr = ToneMapping.photographic_local(hdr, a=float(args.a), epsilon=float(args.epsilon), scale_max=int(args.scale), p=float(args.phi))
        cv2.imwrite(os.path.join(target_dir, output), ldr)
