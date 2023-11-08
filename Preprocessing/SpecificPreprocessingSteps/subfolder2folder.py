import os.path
from pathlib import Path

root_path = Path('C:\\Users\\fonso\\Pictures\\WarshipCam2')


jpg_directory = root_path.rglob('*.jpg')
xml_directory = root_path.rglob('*.xml')
for f in jpg_directory:
    if not os.path.exists(str(root_path.joinpath(f.name))):
        f.rename(str(root_path.joinpath(f.name)))

for f in xml_directory:
    if not os.path.exists(str(root_path.joinpath(f.name))):
        f.rename(str(root_path.joinpath(f.name)))