'''
This module is used to sample images from videos
Author: Alfonso Ponce Navarro
Date: 03/01/2024
'''

import cv2
from pathlib import Path
from tqdm import tqdm
import random
import logging


def video_sampling(
        input_folder_path: Path,
        output_folder_path: Path,
        ratio_discard: float) -> None:
    '''
    Function that samples images randomly from a folder of videos.

    :param input_folder_path: Path to the folder containing all videos.
    :param output_folder_path: Path to the folder where results will be produced
    :param ratio_discard: Ratio used to discard images.
    '''

    try:
        assert input_folder_path.exists()
    except AssertionError as err:
        logging.error(err)
        raise err

    sample_count = 1
    video_list = list(input_folder_path.glob('*.mp4'))
    for video_file in tqdm(video_list, desc='Processing Videos...'):
        vidcap = cv2.VideoCapture(str(video_file))

        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_discarded_frames = int(total_frames * (ratio_discard / 100))
        discarded_frame_list = random.sample(
            range(total_frames), total_discarded_frames)
        frame_list = range(total_frames)
        for i in tqdm(frame_list, desc='Frame Sampling...'):
            ret, frame = vidcap.read()
            if not ret:
                break
            if i not in discarded_frame_list:
                cv2.imwrite(str(output_folder_path.joinpath(
                    f'{str(sample_count)}.jpg')), frame)
                sample_count += 1
