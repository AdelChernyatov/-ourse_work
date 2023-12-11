import argparse
from pathlib import Path
from modules.process import delete_extra_files, saved_chest_annotations, save_masks
from modules.process_conturs import create_contour_numpy_df, create_contour, create_contour_path_df, split_train_test_contour
from modules.dataset import GetDataSet


def process(args) -> None:
    """
    Full dataset processing
    """
    ann_filename, img_filenames = saved_chest_annotations(args.annotations_path)
    delete_extra_files(args.annotations_path, args.img_path, ann_filename, img_filenames)
    dataset = GetDataSet(args.img_path, args.annotations_path)
    output_dir = args.img_path.parent / 'exp_generated'
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = save_masks(dataset, output_dir=output_dir)
    df_np_contour = create_contour_numpy_df(dataset, args.img_path, mask_dir)
    contour_dir = create_contour(df_np_contour, output_dir)
    df_contours = create_contour_path_df(args.img_path, mask_dir, contour_dir)
    split_train_test_contour(df_contours, output_dir)


def get_args():
    parser = argparse.ArgumentParser(
        description="Chest Segmentation args."
    )
    parser.add_argument(
        '-ann', '--annotations-path',
        type=Path, required=False,
        default="D:/Course_work/data/exp_annotations"
    )
    parser.add_argument(
        '-img', '--img-path',
        type=Path, required=False,
        default='D:/Course_work/data/exp_images'
    )
    return parser.parse_args()


if __name__ == '__main__':
    process(get_args())