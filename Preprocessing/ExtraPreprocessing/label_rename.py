from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

def label_rename(labels_directory: Path, current_labels_list: list, target_labels_list: list) -> None:
    '''
    Function that renames current labels to new ones. current_labels_list[i] matches target_labels_list[i].
    :param labels_directory: Path to labels folder.
    :param current_labels_list: list of current labels to be renamed.
    :param target_labels_list: list of new labels.
    :return:
    '''

    labels_list = list(labels_directory.glob('*'))

    for file in tqdm(labels_list, desc='Renaming in process...'):
        if file.suffix == '.xml':
            try:
                tree = ET.parse(file)
            except BaseException:
                xmlp = ET.XMLParser(encoding='utf-8')
                tree = ET.parse(file, parser=xmlp)
            root = tree.getroot()

            for obj in root.findall('object'):
                curr_label = obj.find('name').text

                if curr_label in current_labels_list:
                    obj.find('name').text = target_labels_list[current_labels_list.index(curr_label)]

            tree.write(str(file))
