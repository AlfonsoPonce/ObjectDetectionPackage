import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image

images_route = f"../input/Torch_Ansys_Dataset_v2/valid/images/"
normal_imgs = [fn for fn in os.listdir(images_route) if fn.endswith('.png')]

def img2np(path, list_of_filename, size = (1280, 1024)):
    # iterating through each file
    for fn in list_of_filename:
        fp = path + fn
        current_image = image.load_img(fp, target_size = size,
                                       color_mode = 'rgb')
        # covert image to a matrix
        img_ts = image.img_to_array(current_image)
        # turn that into a vector / 1D array
        img_ts = [img_ts.ravel()]
        try:
            # concatenate different images
            full_mat = np.concatenate((full_mat, img_ts))
        except UnboundLocalError:
            # if not assigned yet, assign one
            full_mat = img_ts
    return full_mat

def find_std_img(full_mat, title, size = (1024, 1280,3)):
    # calculate the average
    mean_img = np.std(full_mat, axis = 0)
    # reshape it back to a matrix
    mean_img = mean_img.reshape(size)
    mean_img = mean_img.astype('uint8')
    out = Image.fromarray(mean_img)
    out.save("STD.png")
    out.show()
    #plt.imshow(mean_img, vmin=0, vmax=255)
    #plt.title(f'Average {title}')
    #plt.axis('off')
    #plt.show()
    return mean_img

# run it on our folders
normal_images = img2np(images_route, normal_imgs)
std = find_std_img(normal_images, 'STD')

