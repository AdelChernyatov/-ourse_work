import os
from skimage import draw
import cv2
import numpy as np
import json


def saved_chest_annotations(annotations_path):
    lung_annotations_filename = []
    lung_images_filename = []
    annotations = os.listdir(str(annotations_path))
    for ann in annotations:
        with open(str(annotations_path) + '/' + ann, 'r') as json_file:
            data = json.load(json_file)
        if 'Lung' in data['annotations'][0]['name'] and 'Lung' in data['annotations'][1]['name']:
            lung_annotations_filename.append(ann)
            lung_images_filename.append(data['image']['original_filename'])

    print(f'Кол-во подходящих снимков - {len(lung_annotations_filename)} из {len(annotations)}')

    return lung_annotations_filename, lung_images_filename


def delete_extra_files(ann_path, img_path, lung_annotations_filename, lung_images_filename):
    """
    Delete files with inappropriate format
    """
    images_names = os.listdir(img_path)
    print(f'Начальное число изображений - {len(images_names)}')
    for name in images_names:
        if name not in lung_images_filename:
            full_path = os.path.join(img_path, name)
            os.remove(full_path)
        annotations_names = os.listdir(ann_path)
        for name in annotations_names:
            if name not in lung_annotations_filename:
                full_path = os.path.join(ann_path, name)
                os.remove(full_path)

        images_name = [img.split('.')[0] for img in lung_images_filename]
        # images_format = [img.split('.')[1] for img in lung_images_filename]
        ann_name = [ann.split('.')[0] for ann in lung_annotations_filename]
        extra_images = set(images_name) - set(ann_name)
        for name in extra_images:
            images = []
            if name in lung_images_filename:
                images.append(name)
            full_path = os.path.join(img_path, name)
            os.remove(full_path)

        print(f'Конечное число json файлов - {len(os.listdir(ann_path))}')


def save_masks(dataset, output_dir):
    mask_dir = output_dir / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)
    for num in range(len(dataset)):
        img, lungs, img_name = dataset[num]
        left_polygon, right_polygon = lungs['left_lung'], lungs['right_lung']
        left_mask = draw.polygon2mask(img.shape, left_polygon)
        right_mask = draw.polygon2mask(img.shape, right_polygon)
        combined_mask = left_mask + right_mask
        rotated_mask = np.rot90(np.fliplr(combined_mask), k=1)
        binary_image = rotated_mask.astype("uint8") * 255
        cv2.imwrite(f'{mask_dir}/_{img_name}', binary_image)
    return mask_dir
