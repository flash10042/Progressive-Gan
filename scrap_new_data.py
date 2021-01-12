import requests
import bs4 as bs
from PIL import Image
from os import path
from tqdm import tqdm

FOLDER = 'train'
BASE_URL = 'https://www.wikiart.org'

authors_r = requests.get('https://www.wikiart.org/en/artists-by-genre/abstract/text-list')
authors_soup = bs.BeautifulSoup(authors_r.text, 'lxml')

for i, a in enumerate(tqdm(authors_soup.find('div', class_='masonry-text-view masonry-text-view-all').find_all('a'))):

    url = a.get('href')
    paints_r = requests.get(BASE_URL+url+'/all-works/text-list')
    paints_soup = bs.BeautifulSoup(paints_r.text, 'lxml')

    for j, paint_li in enumerate(paints_soup.find_all('li', class_='painting-list-text-row')):
        try:
            paint_url = paint_li.findChildren('a', recursive=False)[0].get('href')
            paint_r = requests.get(BASE_URL+paint_url)
            paint_soup = bs.BeautifulSoup(paint_r.text, 'lxml')
            
            paint_src = paint_soup.find('img', itemprop='image').get('src')
            
            with open(path.join(FOLDER, f'{i}_{j}.png'), 'wb') as file:
                file.write(requests.get(paint_src).content)
        except:
            continue