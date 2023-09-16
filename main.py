import numpy as np
import os
import yaml

from PIL import Image

from mask_processor.object_detector import sam_segment_object

CONFIG_FILE_PATH = "config.yml"

def read_config(file_path: str):
    with open(file_path,"r") as config_file:
        try:
            config = yaml.safe_load(config_file)
            return config['database']['input_dir'], config['database']['output_dir']
        except yaml.YAMLError as exc:
            print(exc)
            return None,None

def read_stored_assets(image_dir: str):
    # read all file from same image_dir
    filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f != '.DS_Store']
    return filenames



if __name__ == "__main__" :
    image_dir, output_dir = read_config(CONFIG_FILE_PATH)
    filenames = read_stored_assets(image_dir)
    for file in filenames:
        masks = sam_segment_object(image_dir,output_dir)
        # generate mask image
        if masks is not None:
            masks = masks.astype(np.uint8) * 255
        im = Image.fromarray(masks[0])
        im.save(output_dir + f"mask_{file}")