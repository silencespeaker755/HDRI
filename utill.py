import numpy as np
from matplotlib import pyplot as plt

def draw_irradiance_map(irradiance_map, output_file, color):
    plt.plot(np.arange(256), irradiance_map, color=color)
    plt.title('Irradiance')
    plt.xlabel("pixel value")
    plt.ylabel("ln(Z)")
    plt.savefig(f'DebevecData/{output_file}') 
    plt.clf()