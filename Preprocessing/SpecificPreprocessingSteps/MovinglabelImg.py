from pathlib import Path

source_dir = Path('C:\\Users\\fonso\\Pictures\\WarshipCam2')

img_dir = Path('../input/Dataset_BarcosV3/train/images')
ann_dir = Path('../input/Dataset_BarcosV3/train/pascal_labels')

xml_files = source_dir.glob('*.xml')

for ann_file in xml_files:
    img_file = Path(str(ann_file).replace('xml', 'jpg'))

    ann_file.rename(ann_dir.joinpath(ann_file.name))
    img_file.rename(img_dir.joinpath(img_file.name))