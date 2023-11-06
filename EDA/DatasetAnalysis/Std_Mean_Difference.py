from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
Average = image.load_img('Average.png', target_size=(1024,1280),
                               color_mode='rgb')
STD = image.load_img('STD.png', target_size=(1024,1280),
                               color_mode='rgb')

Average_arr = image.img_to_array(Average)
STD_arr = image.img_to_array(STD)

difference = np.abs(Average_arr - STD_arr).astype('uint8')

out=Image.fromarray(difference)
out.save("Difference.png")
out.show()