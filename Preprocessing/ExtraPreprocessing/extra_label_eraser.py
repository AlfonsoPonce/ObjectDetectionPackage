from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

def erase_extra_labels(labels_folder_path: Path, labels_to_remain_list: list) -> None:
    '''
    Function to remove unused labels.
    :param labels_folder: Path to folder where labels are stored
    :param labels_to_remain_list: list of labels that will be kept. Other labels different than this will be removed.
    :return:
    '''
    label_list = list(labels_folder_path.glob('*'))
    for file in tqdm(label_list, desc='Processing labels'):
        if file.suffix == '.xml':
            try:
                tree = ET.parse(file)
            except BaseException:
                xmlp = ET.XMLParser(encoding='utf-8')
                tree = ET.parse(file, parser=xmlp)
            root = tree.getroot()

            for obj in root.findall('object'):
                label = obj.find("name").text

                if label not in labels_to_remain_list:
                    root.remove(obj)
            tree.write(str(file))