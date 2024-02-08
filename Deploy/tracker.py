
import torch
from PIL import Image
from pathlib import Path
import numpy as np

from sort import *
import cv2
import time
import sys
sys.path.append('../Modeling/')
from Training import custom_utils
from Model_Zoo.Zoo import Zoo

video_input = Path('C:\\Users\\fonso\\Documents\\Data\\corte_1.mp4')
output_path = Path('C:\\Users\\fonso\\Documents\\model_video_output\\tracker')
model_name = 'fasterrcnn_resnet50'
checkpoint = 'C:\\Users\\fonso\\Documents\\ObjectDetectionPackage\\model_output_3\\BEST_FasterRCNN.pth'
class_list = ['ball']
resize = (1280, 720)
detection_threshold = 0.9
tracker = 'csrt'

model_repo = Zoo(len(class_list))

model = model_repo.get_model(model_name, True)

checkpoint = torch.load(str(Path(checkpoint)), map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])


# Create inference result dir if not present.
if not output_path.exists():
    output_path.mkdir(exist_ok=True, parents=True)

class_list.insert(0, '__bg__')
num_classes = len(class_list)
# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(num_classes, 3))

model.to('cuda').eval()

cap = cv2.VideoCapture(str(video_input))

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create VideoWriter object
out = cv2.VideoWriter(str(output_path.with_suffix('.mp4')),
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))

frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second



first_detection = False
sort_max_age = 5
sort_min_hits = 2
sort_iou_thresh = 0.2
mot_tracker =Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)

# read until end of video
while (cap.isOpened()):
    print(frame_count)
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        if resize is not None:
            image = cv2.resize(image, (resize[0], resize[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get predictions for the current frame
            outputs = model(image.to('cuda'))
        end_time = time.time()

        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold].astype(float)





            if len(boxes) > 0:
                track_bbs_ids = mot_tracker.update(np.array([list(np.append(box, score))for box, score in zip(boxes, scores)])) #NEW
            else:
                track_bbs_ids = mot_tracker.update(np.empty((0,5)))
            print(track_bbs_ids)

            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [class_list[i]
                            for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]

                color = COLORS[class_list.index(class_name)]
                frame = custom_utils.draw_boxes(frame, box, color, resize)
                frame = custom_utils.put_class_text(
                    frame, box, class_name,
                    color, resize
                )
        cv2.putText(frame, f"{fps:.1f} FPS",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    2, lineType=cv2.LINE_AA)

        # cv2.imshow('image', frame)
        # frame_show = Image.fromarray(np.uint8(frame*255))
        # frame_show.show()
        out.write(frame)
        # press `q` to exit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
# cv2.destroyAllWindows()

# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")