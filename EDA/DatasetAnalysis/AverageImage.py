import os, numpy, PIL
from PIL import Image

images_route = f"../input/Torch_Ansys_Dataset_v2/train/images/"
# Access all PNG files in directory
allfiles=os.listdir(images_route)
imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"]]

# Assuming all images are the same size, get dimensions of first image
w,h=Image.open(images_route+imlist[0]).size
N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((h,w,4),numpy.float)

# Build up average pixel intensities, casting each image as an array of floats
for im in imlist:
    imarr=numpy.array(Image.open(images_route+im),dtype=numpy.float)
    arr=arr+imarr/N

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image
out=Image.fromarray(arr)
out.save("Average.png")
out.show()