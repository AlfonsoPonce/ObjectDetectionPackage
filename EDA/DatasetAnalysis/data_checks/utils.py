from pathlib import Path
from PIL import Image, ImageChops
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSION_LIST = ['.png', '.jpg', '.jpeg']
LABEL_EXTENSION_LIST = ['.xml', '.json', '.txt']

def check_label_extensions(labels_dir: Path) -> dict:
    '''
    Checks label extensions inside labels folder
    :param labels_dir (Path): labels folder
    :return (dict): Number of files per available extension (available extensions: xml (PASCAL VOC), json(COCO), txt(YOLO))
    '''
    extension_dict = {'.xml': 0, '.json': 0, '.txt': 0}
    for extension in LABEL_EXTENSION_LIST:
        extension_dict[extension] = len(list(labels_dir.glob(f'*{extension}')))

    return extension_dict


def check_images_extensions(images_dir: Path) -> dict:
    '''
    Checks images extensions inside images folder
    :param images_dir (Path): images folder
    :return (dict): Number of files per available extension (available extensions: png, jpg, jpeg)
    '''
    extension_dict = {'.png': 0, '.jpg': 0, '.jpeg': 0}
    for extension in IMAGE_EXTENSION_LIST:
        extension_dict[extension] = len(list(images_dir.glob(f'*{extension}')))

    return extension_dict

def check_corrupted_images(images_dir: Path) -> list:
    '''
    Checks the number of corrupted images inside images folder
    :param images_dir (Path): images folder
    :return (list): list of corrupted images
    '''
    corrupted_file_list = []
    for extension in IMAGE_EXTENSION_LIST:
        for image_file in images_dir.glob(f'*{extension}'):
            try:
                im = Image.open(str(image_file))
                im.verify()  # I perform also verify, don't know if he sees other types o defects
                im.close()  # reload is necessary in my case
                im = Image.open(str(image_file))
                im.transpose(Image.FLIP_LEFT_RIGHT)
                im.close()
            except:
                corrupted_file_list.append(image_file.name)

    return corrupted_file_list

def check_unpaired_entities(images_dir: Path, labels_dir: Path) -> dict:
    '''
    Checks if all images have their labels and viceversa. Images and labels MUST be named the same (e.g.:1.jpg --> 1.xml)
    :param root_dir (Path): Root directory of the data
    :return (dict): Images without labels and labels without images.
    '''

    unpaired_entities_dict = {'images_wo_labels': [], 'labels_wo_images': []}


    image_list = []
    label_list = []

    for image_extension in IMAGE_EXTENSION_LIST:
        image_list.extend(list(images_dir.glob(f'*{image_extension}')))
        for label_extension in LABEL_EXTENSION_LIST:
            label_list.extend(list(labels_dir.glob(f'*{label_extension}')))

    image_set = set([file.stem for file in image_list])
    labels_set = set([file.stem for file in label_list])

    unpaired_entities_dict['images_wo_labels'] = image_set.difference_update(labels_set)
    unpaired_entities_dict['labels_wo_images'] = labels_set.difference_update(image_set)

    return unpaired_entities_dict

def check_duplicated_images(images_dir: Path):
    '''
    Gets duplicated images in a dataset
    :param images_dir (Path): images folder
    :return (list): List of pairs whose pairs contain duplicated images.
    '''
    duplicate_images_list = []

    for extension in IMAGE_EXTENSION_LIST:
        img_files = list(images_dir.glob(f'*{extension}'))

        for img_idx in tqdm(range(len(img_files))):
            img = Image.open(str(img_files[img_idx])).convert('RGB')
            img_idx_to_remove = img_idx + 1
            while img_idx_to_remove < len(img_files):
                img_to_remove = Image.open(str(img_files[img_idx_to_remove])).convert('RGB')
                diff = ImageChops.difference(img, img_to_remove)

                if not diff.getbbox(): #If not different images
                    duplicate_images_list.append((img_files[img_idx].name, img_files[img_idx_to_remove].name))

                img_idx_to_remove += 1

    return duplicate_images_list

def get_images_shapes(images_dir: Path) -> set:
    '''
    Get all image shapes in dataset
    :param images_dir (Path):  images folder
    :return (set): All shapes in dataset
    '''

    shape_list = []
    for extension in IMAGE_EXTENSION_LIST:
        image_list = list(images_dir.glob(f'*{extension}'))
        shape_list.extend([np.array(Image.open(str(image))).shape for image in image_list])

    return set(shape_list)


if __name__ == '__main__':
    print()