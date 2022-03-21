import numpy as np
import cv2, exifread
import os

def read_images(image_dir):
    paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]
    LDR_images = []
    exposure_times = []

    for path in paths:
        # read image and append images into LDR list
        img = cv2.imread(path)
        LDR_images.append(img)

        # get exposure time and append it into list
        time = get_exposure_time(path)
        exposure_times.append(time)
        print(f"{path} -> exposure time: {time}s")
    
    # transform LDR_images into np array
    LDR_images = np.array(LDR_images)

    return LDR_images, exposure_times

def get_exposure_time(path):
    # get file's exif tags
    exif_tags = exifread.process_file(open(path, "rb"))

    #return exposure time
    return transform_exif_fraction_to_float(str(exif_tags["EXIF ExposureTime"]))

def transform_exif_fraction_to_float(fraction):
    numbers = list(map(float, fraction.split("/")))

    if(len(numbers) == 1):
        return numbers[0]
    else:
        numbers[1] = 1<<(int(numbers[1])-1).bit_length()

    return numbers[0]/numbers[1]

def save_HDR_images(image, output):
    cv2.imwrite(output, image)