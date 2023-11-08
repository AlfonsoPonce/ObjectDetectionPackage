import cv2
vidcap = cv2.VideoCapture('C:\\Users\\fonso\\Documents\\TFG\\Version2\\ObjectDetectionPackage\\video5.mp4')
success,image = vidcap.read()
count = 19184
while success:
  cv2.imwrite("C:\\Users\\fonso\\Documents\\TFG\\Version2\\ObjectDetectionPackage\\input\\Autoetiquetado\\video5_Images\\%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1