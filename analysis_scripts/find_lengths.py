import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import random

sal_metric_name = "mean_recall"
thresh_metric_name = "and_or_threshold"

for data in ['korus','MFC18_small', 'imd2020', 'imd2020_resize', 'MFC18_resize', 'korus_size_up', 'korus_SE', 'MFC18_SE','imd2020_SE']:
    for partition in ['under_two', 'two_to_four', 'four_to_six', 'six_to_eight', 'above_eight']:
        values = np.load(f'./ICIP_2024/sal_groups/{data}_{sal_metric_name}_{thresh_metric_name}_{partition}.npy', allow_pickle = True)
        length = len(values)
        print(f'in dataset: {data} for partition: {partition} -- {length}')
