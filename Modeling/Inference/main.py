'''
Main module that implements inference functionality

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''

from Modeling.Model_Zoo.Zoo import Zoo
import torch
from Modeling.Inference.video_inference import local_video_inference
from pathlib import Path
def run(args):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_repo = Zoo(len(args['classes'])+2)
    model = model_repo.get_model(args['model_name'])
    checkpoint = torch.load(args['model_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    local_video_inference(Path(args['video_input']),
                          Path(args['output_path']),
                          model,
                          args['classes'],
                          args['detection_threshold'],
                          device,
                          args['resize'])



if __name__ == '__main__':
    args = {}
    args['resize'] = (1920, 1080, 3)
    args['classes'] = ['player', 'ball', 'referee', 'goalkeeper']
    args['video_input'] = '../../Data/videos/corte_1.mp4'
    args['output_path'] = './video_output/video_1.mp4'
    args['model_name'] = 'fasterrcnn_resnet50'
    args['model_checkpoint'] = '../output/BEST_FasterRCNN.pth'
    args['detection_threshold'] = 0.7

    run(args)