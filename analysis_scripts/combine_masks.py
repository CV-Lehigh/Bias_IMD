import numpy as np
from tqdm import tqdm
import cv2 as cv
import os


def load_image(path, img_name):
    for filename in sorted(os.listdir(path)):
        if(img_name[:len(img_name)-3] in filename):
            path = os.path.join(path,filename)
            image = cv.imread(path,0).astype(np.float64)
            return (filename,image)

def mask_add_and_combine(u2net_image, r3net_image):
    u2net_half = np.divide(u2net_image,2)
    r3net_half = np.divide(r3net_image,2)
    combined_networks = np.add(u2net_half, r3net_half)
    combined_networks[combined_networks > 255] = 255
    return combined_networks 

def generate_combination(sal_dir, dataset):
    for image in os.listdir(sal_dir):
        u2net_file, u2net_image = load_image(f'./inference_outputs/{dataset}/u2net/', image)
        r3net_file, r3net_image = load_image(f'./inference_outputs/{dataset}/r3net/', image)
        if(np.shape(u2net_image) == np.shape(r3net_image)):
            new_mask = mask_add_and_combine(u2net_image, r3net_image)
            cv.imwrite(f'./inference_outputs/{dataset}/combine_maps/{r3net_file}', new_mask)
        else:
            print(f"not same size: {r3net_file} & {u2net_file}")


all_dataset = ['korus', 'korus_size_up', 'korus_SE', 'MFC18_small', 'MFC18_resize', 'MFC18_SE', 'IMD2020_formatted', 'IMD2020_resize', 'IMD2020_SE']
for dataset in all_dataset:
    print(f'generating combination maps: {dataset}')
    generate_combination(f'./inference_outputs/{dataset}/u2net/', dataset)