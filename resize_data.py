from PIL import Image
import numpy as np
from os import path, listdir
from tqdm import tqdm

INPUT = 'train'
OUTPUT = 'train_resized'

TARGET_SIZE = (128, 128)

for file in tqdm(listdir(INPUT)):
    img = Image.open(path.join(INPUT, file))
    img = img.resize(TARGET_SIZE).convert('RGB')
    np.save(path.join(OUTPUT, file[:-4]+'.npy'), np.asarray(img) / 127.5 - 1)
