import numpy as np
import argparse
import os
from utils import draw_inverse_response_curve
from imageIO import read_images

from debevec import Debevec
from robertson import RobertsonHDR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", default="Photos/JPG")
    parser.add_argument("-d", "--save_dir", default="DebevecData")
    parser.add_argument("-m", "--method", default="debevec")
    parser.add_argument("-r", "--radiance", default="Radiance")
    parser.add_argument("-g", "--inverse_response_curve", default="Inverse_CRF")
    parser.add_argument("-e", "--epoch", default=5)
    args = parser.parse_args()

    target_dir = args.save_dir
    dir_path = os.path.join(args.source_dir)
    LDR_images, exposure_times = read_images(dir_path)

    # for output radiance map
    prefix = args.radiance
    crf = args.inverse_response_curve
    outputs_format = [  {"CRF": f"{crf}_B.png", "radiance": f"{prefix}_B.npy", "color": "blue"},  \
                        {"CRF": f"{crf}_G.png", "radiance": f"{prefix}_G.npy", "color": "green"}, \
                        {"CRF": f"{crf}_R.png", "radiance": f"{prefix}_R.npy", "color": "red"} ]
    outputs_format = np.array([{key:(row[key] if key=="color" else os.path.join(target_dir, row[key])) for key in row} for row in outputs_format])

    if args.method == "debevec":
        debevec = Debevec(images=LDR_images, exposure_times=exposure_times)

        B, G, R = debevec.split_BGR_images()
        image_number, pixels = B.shape

        points = debevec.pick_evaluation_points(pixels, 100)
        
        inverse_CRF_B = debevec.generate_inverse_response_curve(sample_points=B[:, points])
        draw_inverse_response_curve(inverse_CRF_B, outputs_format[0]["CRF"], color=outputs_format[0]["color"])
        
        inverse_CRF_G = debevec.generate_inverse_response_curve(sample_points=G[:, points])
        draw_inverse_response_curve(inverse_CRF_G, outputs_format[1]["CRF"], color=outputs_format[1]["color"])
        
        inverse_CRF_R = debevec.generate_inverse_response_curve(sample_points=R[:, points])
        draw_inverse_response_curve(inverse_CRF_R, outputs_format[2]["CRF"], color=outputs_format[2]["color"])
        
        threads = [None] * 3
        print(threads)
        threads[0] = debevec.reconstruct_irradiance_image(B, inverse_CRF_B, outputs_format[0]["radiance"])
        
        threads[1] = debevec.reconstruct_irradiance_image(G, inverse_CRF_G, outputs_format[1]["radiance"])

        threads[2] = debevec.reconstruct_irradiance_image(R, inverse_CRF_R, outputs_format[2]["radiance"])

        for thread in threads:
            thread.join()

    else:
        rb = RobertsonHDR(LDR_images, exposure_times, 256)

        rb.process_radiance_map([t["radiance"] for t in outputs_format], epoch = int(args.epoch))
        for index, output in enumerate(outputs_format):
            draw_inverse_response_curve(np.log(rb.gCurves[index]), output["CRF"], color=output["color"]) 








    
