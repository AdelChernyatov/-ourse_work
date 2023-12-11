import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_Polygon
from PIL import Image
from sklearn.model_selection import train_test_split


def create_contour_numpy_df(dataset, img_folder, mask_folder):
    np_contour_folder = mask_folder.parent / 'np_contours'
    np_contour_folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns=['image_path', 'mask_path', 'left_contour_path', 'right_contour_path'])
    for num, mask_name in zip(range(len(dataset)), os.listdir(mask_folder)):
        _, lungs, img_name = dataset[num]
        img_path = os.path.join(img_folder, img_name)
        mask_path = os.path.join(mask_folder, mask_name)
        left_contour_path, right_contour_path = save_contour(lungs, img_name, np_contour_folder)
        df.loc[num] = {'image_path': img_path, 'mask_path': mask_path, 'left_contour_path': left_contour_path,
                       'right_contour_path': right_contour_path}
    return df


def save_contour(lungs, img_name, np_contour_folder):
    left_contour, right_contour = np.array(lungs['left_lung']),  np.array(lungs['right_lung'])
    left_contour_name = str(os.path.splitext(img_name)[0]) + 'left.npy'
    right_contour_name = str(os.path.splitext(img_name)[0]) + 'right.npy'
    left_contour_path = os.path.join(np_contour_folder, left_contour_name)
    right_contour_path = os.path.join(np_contour_folder, right_contour_name)
    np.save(left_contour_path, left_contour)
    np.save(right_contour_path, right_contour)
    return left_contour_path, right_contour_path


def resize_contour(image, left_contour, right_contour, new_size=(576, 576)):
    scale_x = new_size[1] / int(image.size[1])
    scale_y = new_size[0] / int(image.size[0])
    left_contour = left_contour * np.array([scale_y, scale_x])
    right_contour = right_contour * np.array([scale_y, scale_x])
    return left_contour, right_contour


def create_contour(df_with_contour, generated_folder):
    print('Создание контуров')
    contours_folder = generated_folder / 'contours'
    contours_folder.mkdir(parents=True, exist_ok=True)
    for img_path, left_lung_path, right_lung_path in zip(df_with_contour['image_path'],
                                                         df_with_contour['left_contour_path'],
                                                         df_with_contour['right_contour_path']):
        left_contour = np.load(left_lung_path)
        right_contour = np.load(right_lung_path)
        img = Image.open(str(img_path)).convert('RGB')
        left_contour, right_contour = resize_contour(img, left_contour, right_contour)

        image_size = (576, 576)
        contour_image = np.zeros(image_size, dtype=np.uint8)  # empty image with black background

        plt.figure()
        ax = plt.gca()
        ax.set_axis_off()
        ax.imshow(contour_image, cmap='gray')

        left_polygon = plt_Polygon(left_contour, edgecolor='white', facecolor='none')
        ax.add_patch(left_polygon)

        right_polygon = plt_Polygon(right_contour, edgecolor='white', facecolor='none')
        ax.add_patch(right_polygon)

        contours_file_name = 'contour_' + os.path.basename(img_path)
        contours_path = os.path.join(contours_folder, contours_file_name)
        plt.savefig(contours_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
        plt.clf()
        plt.close()

    return contours_folder


def create_contour_path_df(img_folder, mask_folder, contour_folder):
    df_contour = pd.DataFrame(columns=['image_path', 'mask_path', 'contour_path'])
    for idx, (img_name, mask_name, contour_name) in enumerate(
            zip(os.listdir(img_folder), os.listdir(mask_folder), os.listdir(contour_folder))):
        img_path = os.path.join(img_folder, img_name)
        mask_path = os.path.join(mask_folder, mask_name)
        contour_path = os.path.join(contour_folder, contour_name)
        df_contour.loc[idx] = {'image_path': img_path, 'mask_path': mask_path, 'contour_path': contour_path, }
    return df_contour


def split_train_test_contour(df_contour, output_dir):
    df_folder = output_dir / 'dataframe'
    df_folder.mkdir(parents=True, exist_ok=True)

    images = df_contour['image_path']
    masks = df_contour['mask_path']
    contour = df_contour['contour_path']

    images_train, images_test, mask_train, mask_test, contour_train, contour_test = train_test_split(images, masks,
                                                                                                     contour, test_size=0.2,
                                                                                                     train_size=0.8,
                                                                                                     random_state=42)

    images_test, images_val, mask_test, mask_val, contour_test, contour_val = train_test_split(images_test, mask_test,
                                                                                               contour_test, test_size=0.5,
                                                                                               random_state=42)

    train_contour_df = pd.DataFrame({'image': images_train,
                                     'mask': mask_train,
                                     'contour': contour_train})
    train_contour_df.to_csv(df_folder / 'train_df.csv', index=False)

    test_contour_df = pd.DataFrame({'image': images_test,
                                    'mask': mask_test,
                                    'contour': contour_test})
    test_contour_df.to_csv(df_folder / 'test_df.csv', index=False)

    val_contour_df = pd.DataFrame({'image': images_val,
                                   'mask': mask_val,
                                   'contour': contour_val})
    val_contour_df.to_csv(df_folder / 'val_df.csv', index=False)

    print(f'counts:\n train_df - {len(images_train)},\n test_df {len(images_test)},\n val_df : {len(images_val)}')