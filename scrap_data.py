import requests
from os import path

FOLDER = 'train'

for artist in range(1, 51):
    paint = 1
    while True:
        URL = f'http://artchallenge.me/painters/{artist}/{paint}.jpg'
        r = requests.get(URL)
        if r.status_code == 200:
            with open(path.join(FOLDER, f'{artist}_{paint}.png'), 'wb') as file:
                file.write(r.content)
        else:
            break
        paint += 1