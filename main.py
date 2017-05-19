import cv2
import numpy as np
import os

from hdr import computeHDR


def main(image_files, output_path, exposure_times, resize=False):

    image_stack = [cv2.imread(name) for name in image_files]

    if resize:
        image_stack = [img[::4, ::4] for img in image_stack]

    log_exposure_times = np.log(exposure_times)
    hdr_image = computeHDR(image_stack, log_exposure_times, gamma=0.9)
    cv2.imwrite(output_path, hdr_image)



if __name__ == "__main__":
    """Generate an HDR image from the images in the input/sample directory """

    input_dir = "input/sample"
    output_dir = "output"
    filename = "output.png"

    src_contents = os.walk(input_dir)
    fnames = src_contents.next()[2]

    image_files = sorted([os.path.join(input_dir, name) for name in fnames])
    output_path = os.path.join(output_dir, filename)
    exposure_times = np.float64([1 / 160.0, 1 / 125.0, 1 / 80.0,
                                 1 / 60.0, 1 / 40.0, 1 / 15.0])

    main(image_files, output_path, exposure_times, resize=False)


