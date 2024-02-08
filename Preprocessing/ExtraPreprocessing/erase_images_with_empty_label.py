from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

def remove_images_with_empty_labels(image_directory: Path, labels_directory: Path):
    '''
    Function that removes images whose associated label file does not contain any label. Besides, the label file is also erased.
    IMAGES AND LABELS MUST BE NAMED THE SAME.
    :param image_directory: path where images are stored.
    :param labels_directory: path where labels are stored.
    :return:
    '''

    labels_list = list(labels_directory.glob('*'))


    for label_file in tqdm(labels_list, desc='Removing empty labels...'):
        if label_file.suffix == '.xml':
            try:
                tree = ET.parse(label_file)
            except BaseException:
                xmlp = ET.XMLParser(encoding='utf-8')
                tree = ET.parse(label_file, parser=xmlp)
            root = tree.getroot()

            if len(root.findall('object')) == 0:
                label_file.unlink()
                image_directory.joinpath(f'{label_file.stem}.jpg').unlink(missing_ok=True)
                image_directory.joinpath(f'{label_file.stem}.png').unlink(missing_ok=True)
