import json
import yaml
import re
import os
import logging
import hashlib
import random
import torch
import numpy as np
import torchvision.transforms as torch_transforms

from torchvision.transforms.functional import InterpolationMode

def clean_hash_prompt(prompt):
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt)
    hash_obj = hashlib.new("sha256")
    hash_obj.update(prompt.encode('utf-8'))
    cleaned_prompt = str(hash_obj.hexdigest())[:64]
    return cleaned_prompt

def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        
def read_dir_filepath(dir):
    filepaths = []
    for filename in os.listdir(dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepaths.append( os.path.join(dir, filename))
    return filepaths
                
def load_yaml(path):
	with open(path, 'r') as file:  
		cfg = yaml.safe_load(file) 
	return cfg

def load_data(path):
    prompt_list = []
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    for i in lines:
        prompt_list.append(i)
    return prompt_list

def get_dictionary(len_subword, en):
    if en == False:
        f = open('./data/vocab.json')
        data_json = json.load(f)
        prompt_list = []
        for key, value in data_json.items():
            if len(key) < len_subword:
                new_key = re.sub(u"([^\u0041-\u005a\u0061-\u007a])", "", key)
                if new_key != "":
                    prompt_list.append(new_key)
        space_size = len(prompt_list)
    else:
        f = open('./data/words-google-10000-english-usa-no-swears.json')
        data_json = json.load(f)
        prompt_list = list(data_json)
        space_size = len(prompt_list)

    return prompt_list, space_size

# a function  to create and save logs in the log files
def log(path, file):
    """[Create a log file to record the experiment's logs]
    
    Arguments:
        path {string} -- path to the directory
        file {string} -- file name
    
    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    # console_logging_format = "%(levelname)s %(message)s"
    # file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
    console_logging_format = "%(message)s"
    file_logging_format = "%(message)s"
    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_unlearnDiff_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform