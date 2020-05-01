import os
import pandas as pd
from quaternions import quaternion_filtration
from frequencyDomain import habitual_filtration
import cv2
import argparse

# DataSet used: LIVE Image Quality Assessment Database
parser = argparse.ArgumentParser(description='Generate a csv file with metrics to habitual and quaternionic approach')
parser.add_argument('dir_original_images_path', help='Path to original images directory')
parser.add_argument('dirs_modified_images_path', help='Path to dir with modified images directories')


def generate_dataframe(args):
    dir_original_images_path = args.dir_original_images_path
    dirs_modified_images_path = args.dirs_modified_images_path

    df = pd.DataFrame(
        {'Problem': [], 'Quaternion_RMSE': [], 'Habitual_RMSE': [], 'Quaternion_SSIM': [], 'Habitual_SSIM': [],
         'Quaternion_Time': [], 'Habitual_Time': []})
    dirs = os.listdir(dirs_modified_images_path)

    filters = [
        {'name': 'ideal_filter', 'band': 'low_pass'},
        {},
        {},
        {},
        {},
        {}
    ]

    for dir in dirs:
        if not os.path.isdir(dir):
            continue
        quaternion_metric, habitual_metric = generate_metrics(dir_original_images_path,
                                                              os.path.join(dirs_modified_images_path, dir),
                                                              filters.pop(0))
        df.append([dir, quaternion_metric['rmse'], habitual_metric['rmse'],
                   quaternion_metric['ssim'], habitual_metric['ssim'],
                   quaternion_metric['time'], habitual_metric['time']])
    df.to_csv('metrics.csv', index=False)


def generate_metrics(dir_original_images, dir_modified_images, filter):
    images = os.listdir(dir_modified_images)
    images_relation = images.pop(
        images.index('info.txt'))  # a link of modified images with its respective original images
    with open(os.path.join(dir_modified_images, images_relation)) as f:
        lines = f.readlines()
    df_images_relation = pd.DataFrame(
        {'Original': [x.split()[0] for x in lines], 'Modified': [x.split()[1] for x in lines]})

    quaternion_metric_values = {'rmse': [], 'ssim': [], 'time': []}
    habitual_metric_values = {'rmse': [], 'ssim': [], 'time': []}

    for image_name in images:
        image = cv2.imread(os.path.join(dir_modified_images, image_name))
        original_image_path = os.path.join(dir_original_images,
                                           df_images_relation[df_images_relation['Modified'] == image_name]['Original'])
        original_image = cv2.imread(original_image_path)

        quaternion_filtered_image, quaternion_time = quaternion_filtration(image, filter)
        habitual_filtered_image, habitual_time = habitual_filtration(image, filter)

        quaternion_metric_values['rmse'].append(rmse(original_image, quaternion_filtered_image))
        quaternion_metric_values['ssim'].append(ssim(original_image, quaternion_filtered_image))
        quaternion_metric_values['time'].append(quaternion_time)
        habitual_metric_values['rmse'].append(rmse(original_image, habitual_filtered_image))
        habitual_metric_values['ssim'].append(ssim(original_image, habitual_filtered_image))
        habitual_metric_values['time'].append(habitual_time)

    quaternion_metric = {'rmse': sum(quaternion_metric_values['rmse']) / len(quaternion_metric_values['rmse']),
                         'ssim': sum(quaternion_metric_values['ssim']) / len(quaternion_metric_values['ssim']),
                         'time': sum(quaternion_metric_values['time']) / len(quaternion_metric_values['time'])}
    habitual_metric = {'rmse': sum(habitual_metric_values['rmse']) / len(habitual_metric_values['rmse']),
                       'ssim': sum(habitual_metric_values['ssim']) / len(habitual_metric_values['ssim']),
                       'time': sum(quaternion_metric_values['time']) / len(quaternion_metric_values['time'])}

    return quaternion_metric, habitual_metric


def rmse(image1, image2):
    pass


def ssim(image1, image2):
    pass


if __name__ == '__main__':
    generate_dataframe(parser.parse_args())
