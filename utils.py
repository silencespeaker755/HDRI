import numpy as np
import os
from matplotlib import pyplot as plt

def draw_inverse_response_curve(irradiance_map, output_file, color):
    plt.plot(np.arange(256), irradiance_map, color=color)
    plt.title('Irradience')
    plt.xlabel("pixel value")
    plt.ylabel("ln(g(Z))")
    plt.savefig(output_file) 
    plt.clf()

def draw_radiance_map(image_dict, store_dir):
    for key in image_dict:
        plt.title(f"Ei of {key}")
        plt.imsave(os.path.join(store_dir, f"Ei_{key}.png"), np.log(image_dict[key] + 1e-8), cmap = 'jet')
        plt.clf()