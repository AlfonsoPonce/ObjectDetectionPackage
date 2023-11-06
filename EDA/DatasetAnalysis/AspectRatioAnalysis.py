import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


image_folder = Path(f"../input/Torch_Ansys_Dataset/train/images")

list_aspect_ratio = []

for file in image_folder.glob(pattern='*.png'):
    im = cv2.imread(str(file))
    width, height, channels = im.shape
    aspect_ratio = width / height
    list_aspect_ratio.append(aspect_ratio)


sns.histplot(list_aspect_ratio)
plt.show()