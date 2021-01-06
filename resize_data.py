from PIL import Image
import numpy as np
from os import path, listdir
import glob

INPUT = 'train'
OUTPUT = 'train_resized'

TARGET_SIZE = (256, 256)

for i, file in enumerate(glob.glob(path.join(INPUT, '8_*'))):
    """
    img = Image.open(path.join(INPUT, file))
    img = (np.array(img.resize(TARGET_SIZE), np.uint8) / 127.5 - 1).astype(np.float32)
    if len(img.shape) != 3:
        img = np.repeat(np.expand_dims(img, -1), 3, axis=-1)
    np.save(path.join(OUTPUT, str(i+1)), img)
    """
    img = Image.open(file)
    img = img.resize(TARGET_SIZE).convert('RGB')
    img.save(path.join(OUTPUT, str(i+1)+'.png'))