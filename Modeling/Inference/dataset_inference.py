'''
Module that implements model inference in a set of images

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from pascal_voc_writer import Writer
import time
from tqdm import tqdm
from Modeling.Training import custom_utils
import shutil
import math

import sys
sys.path.append('../../')

from Preprocessing.Tiling.image_bbox_slicer.slicer import Slicer
from Preprocessing.Tiling.image_bbox_slicer.helpers import calc_columns_rows
def dataset_inference(
        model,
        device: str,
        image_input_dir: Path,
        label_input_dir: Path,
        output_dir: Path,
        class_list: list,
        detection_threshold: float,
        resize: tuple) -> None:
    '''
    Model inference over a set of datasets.

    :param model: torch detection model
    :param device: device where the model is set
    :param image_input_dir: directory to fetch images
    :param label_input_dir: directory to fetch labels. If None is passed, then no metrics will be computed.
    :param output_dir: directory where results will be output
    :param class_list: list of classes to detect
    :param detection_threshold: threshold that sets minimum confidence to an object to be detected
    :param resize: tuple that represents image resize
    :return:
    '''

    # Create inference result dir if not present.
    output_dir.joinpath('images').mkdir(exist_ok=True, parents=True)
    output_dir.joinpath('labels').mkdir(exist_ok=True, parents=True)

    class_list.insert(0, '__bg__')

    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(class_list), 3))

    model.to(device).eval()

    TILING = True
    num_tiles = 4

    # directory where all the images are present
    # DIR_TEST = args['input']
    test_images = []
    if image_input_dir.exists():
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(list(image_input_dir.glob(file_type)))
    else:
        test_images.append(str(image_input_dir))
    #print(f"Test instances: {len(test_images)}")

    # ANN_DIR = args['labels']
    if label_input_dir != None:
        test_ann = []
        if label_input_dir.exists():
            annotation_file_types = ['*.xml', '*.txt', '*.json']
            for file_type in annotation_file_types:
                test_ann.extend(list(label_input_dir.glob(file_type)))
        else:
            test_ann.append(str(label_input_dir))

    # to count the total number of frames iterated through
    frame_count = 0
    # to keep adding the frames' FPS
    total_fps = 0
    images_list = range(len(test_images))

    for i in tqdm(images_list, desc='Computing predictions...'):

        # get the image file name for saving output later on
        image_name = test_images[i].stem
        image = Image.open(str(test_images[i]))
        orig_width = image.width
        orig_height = image.height
        orig_image = image.copy()
        if resize:
            image = image.resize(resize)
        start_time = time.time()
        if TILING:
            outputs = tiled_image_prediction(model, device, image, num_tiles)
        else:
            outputs = full_image_prediction(model, device, image)
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

        # if len(outputs[0]['boxes']) != 0:

        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()

        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)



        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [class_list[j]
                        for j in outputs[0]['labels'].cpu().numpy()]

        writer = Writer(str(output_dir.joinpath(
            "images", f"{image_name}.jpg")), orig_width, orig_height)
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[class_list.index(class_name)]
            orig_image = custom_utils.draw_boxes(
                np.array(orig_image), box, color, resize)
            orig_image = custom_utils.put_class_text(
                orig_image, box, class_name,
                color, resize
            )

            #orig_image = Image.fromarray(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            orig_image = Image.fromarray(orig_image)
            orig_image.save(
                str(output_dir.joinpath("images", f"{image_name}.jpg")))

            for x in range(len(boxes)):
                xmin = int((boxes[x][0] / resize[0]) * orig_width)
                ymin = int((boxes[x][1] / resize[1]) * orig_height)
                xmax = int((boxes[x][2] / resize[0]) * orig_width)
                ymax = int((boxes[x][3] / resize[1]) * orig_height)

                writer.addObject(pred_classes[x], xmin, ymin, xmax, ymax)

        writer.save(str(output_dir.joinpath("labels", f"{image_name}.xml")))

        total, used, free = shutil.disk_usage("/")
        #if free // (2**30) < 2:
        #    print("running out of memory...")
        #    break
        #print(f"Image {image_name} done...")
        #print('-' * 50)


    print('TEST PREDICTIONS COMPLETE')
    # cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


    '''
    ##Obteniendo las métricas de la inferencia##
    with_metric = True
    if with_metric:
        print("adios")
        valid_dataset = datasets.create_valid_dataset()
        valid_loader = datasets.create_valid_loader(valid_dataset, NUM_WORKERS)
        with open(f"./inference_outputs/{root_inf_dir}/log.txt", "w") as f:
            with contextlib.redirect_stdout(f):
                evaluate(model, valid_loader, device=DEVICE)

        log.append(f"Metric saved in ./inference_outputs/{root_inf_dir}/log.txt")
    '''




def full_image_prediction(model, device: str, full_image):
    image = np.array(full_image).astype('float64')
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype('float64')
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # image = image[None,:,:]
    # add batch dimension
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image.to(device))

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    return outputs









def tiled_image_prediction(model,
                            device: str,
                            full_image,
                            num_tiles):

    tiler = Slicer()
    tile_list, tiles_coords = tiler.slice_single_image(None, None, num_tiles, full_image)

    output_list = []
    for tile_idx in range(len(tile_list)):
        image = np.array(tile_list[tile_idx]).astype('float64')
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype('float64')
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # image = image[None,:,:]
        # add batch dimension
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            outputs = model(image.to(device))


        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        for i in range(len(outputs[0]['boxes'])):
            outputs[0]['boxes'][i][0] = outputs[0]['boxes'][i][0] + tiles_coords[tile_idx][0] #xmin
            outputs[0]['boxes'][i][1] = outputs[0]['boxes'][i][1] + tiles_coords[tile_idx][1] #ymin
            outputs[0]['boxes'][i][2] = outputs[0]['boxes'][i][2] + tiles_coords[tile_idx][0] #xmax
            outputs[0]['boxes'][i][3] = outputs[0]['boxes'][i][3] + tiles_coords[tile_idx][1] #ymax


        output_list.append(outputs)


    #Merge outputs from tiles to form the final output
    final_result = []

    for output in output_list:
        final_result.extend(output)

    merged_dict = {'boxes': torch.empty(0),
                   'labels': torch.empty(0, dtype=torch.long),
                   'scores': torch.empty(0)}

    for d in final_result:
        merged_dict['boxes'] = torch.cat([merged_dict['boxes'], d['boxes']])
        merged_dict['labels'] = torch.cat([merged_dict['labels'], d['labels']])
        merged_dict['scores'] = torch.cat([merged_dict['scores'], d['scores']])

    return [merged_dict]






if __name__ == '__main__':

    tiler = Slicer()

    tiler.slice_single_image(None, None, 4, np.ones(10))