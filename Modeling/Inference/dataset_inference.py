'''
Module that implements model inference in a set of images

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from pascal_voc_writer import Writer
import time
from Modeling.Training import custom_utils


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
    :param label_input_dir: directory to fetch labels
    :param output_dir: directory where results will be output
    :param class_list: list of classes to detect
    :param detection_threshold: threshold that sets minimum confidence to an object to be detected
    :param resize: tuple that represents image resize
    :return:
    '''
    # Create inference result dir if not present.
    output_dir.joinpath('images').mkdir(exist_ok=True, parents=True)
    output_dir.joinpath('labels').mkdir(exist_ok=True, parents=True)

    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(class_list), 3))

    model.to(device).eval()

    # directory where all the images are present
    # DIR_TEST = args['input']
    test_images = []
    if image_input_dir.exists():
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(list(image_input_dir.glob(file_type)))
    else:
        test_images.append(str(image_input_dir))
    print(f"Test instances: {len(test_images)}")

    # ANN_DIR = args['labels']
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

    for i in range(len(test_images)):
        # get the image file name for saving output later on
        image_name = test_images[i].stem
        image = Image.open(str(test_images))
        orig_width = image.width
        orig_height = image.height
        orig_image = image.copy()
        # BGR to RGB

        if resize:
            image = image.resize(resize)

        image = np.array(image)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # image = image[None,:,:]
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(device))
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
                orig_image, box, color, resize)
            orig_image = custom_utils.put_class_text(
                orig_image, box, class_name,
                color, resize
            )

            orig_image = Image.fromarray(np.uint8(orig_image * 255))
            orig_image.save(
                str(output_dir.joinpath("images", f"{image_name}.jpg")))

            for x in range(len(boxes)):
                xmin = int((boxes[x][0] / resize) * orig_width)
                ymin = int((boxes[x][1] / resize) * orig_height)
                xmax = int((boxes[x][2] / resize) * orig_width)
                ymax = int((boxes[x][3] / resize) * orig_height)

                writer.addObject(pred_classes[x], xmin, ymin, xmax, ymax)

        writer.save(str(output_dir.joinpath("labels", f"{image_name}.xml")))
        print(f"Image {image_name} done...")
        print('-' * 50)

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
