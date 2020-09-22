import os
import pandas as pd
from quaternions import quaternion_filtration
from frequencyDomain import habitual_filtration
import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Generate a csv file with habitual and quaternionic approach '
                                             'processing time to the 4 used filters.')
parser.add_argument('path_input', help='Path to input images directory.')
parser.add_argument('path_output', help='Path to output images directory.')


def generate_dataframe(args):
    path_input = args.path_input
    path_output = args.path_output


    filters = [
        {'name': 'ideal_filter', 'band': 'low_pass'},
        {'name': 'ideal_filter', 'band': 'high_pass'},
        {'name': 'butterworth_filter', 'band': 'low_pass'},
        {'name': 'butterworth_filter', 'band': 'high_pass'}
    ]

    dict_aux = dict()

    for filter in filters:
        directory_output = os.path.join(path_output, 'quaternion', f"{filter['name']}_{filter['band']}")
        if not os.path.exists(directory_output):
          os.makedirs(directory_output)
          os.makedirs(directory_output.replace('quaternion', 'habitual'))

        habitual_time, quaternion_time = calculate_processing_time(path_input,
                                                              directory_output,
                                                              filter)

        dict_aux[f"{filter['name']}_{filter['band']}"] = [habitual_time, quaternion_time]

    df = pd.DataFrame()
    for key in dict_aux.keys():
      df[key] = dict_aux[key]
    df.to_csv('time.csv', index=False)


def calculate_processing_time(dir_input, dir_output, image_filter):
    habitual_time_list = []
    quaternion_time_list = []
    for image_name in os.listdir(dir_input):
        if not image_name.endswith('.bmp'):
            continue
        image = cv2.imread(os.path.join(dir_input, image_name))

        habitual_image, habitual_time = habitual_filtration(image, image_filter)
        habitual_time_list.append(habitual_time)
        cv2.imwrite(os.path.join(dir_output.replace('quaternion', 'habitual'), image_name), habitual_image)

        quaternion_image, quaternion_time = quaternion_filtration(image, image_filter)
        quaternion_time_list.append(quaternion_time)
        cv2.imwrite(os.path.join(dir_output, image_name), quaternion_image)
        print(habitual_time, quaternion_time)
        assert np.equal(habitual_image, quaternion_image).all(), 'The resulting images should be the same'

    habitual_mean = sum(habitual_time_list)/len(habitual_time_list)
    quaternion_mean = sum(quaternion_time_list)/len(quaternion_time_list)

    return habitual_mean, quaternion_mean


if __name__ == '__main__':
    generate_dataframe(parser.parse_args())
