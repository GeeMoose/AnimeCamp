
from ultralytics import FastSAM, SAM
from ultralytics.models.fastsam import FastSAMPrompt

import torch

def sam_segment_object(image_path, output_dir):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_ckpt = '../weights/FastSAM-x.pt'
    model = FastSAM(sam_ckpt)
    # Load a model
    results = model(image_path,device=device,retina_masks=True,imgsz=1024,conf=0.4,iou=0.9)
    prompt_process = FastSAMPrompt(image_path, results, device=device)
    # 发型检测器
    ann = prompt_process.text_prompt(text='hair on the person\'s head')
    prompt_process.plot(annotations=ann, output=output_dir)
    
    return ann

