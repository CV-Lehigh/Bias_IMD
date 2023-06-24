import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import random

sal_metric_name = "mean_recall"
thresh_metric_name = "and_or_threshold"

for data in ['imd2020', 'MFC18_small', 'korus']:
    for partition in ['under_two', 'two_to_four', 'four_to_six', 'six_to_eight', 'above_eight']:
        values = np.load(f'./data_splits/5_split/{data}_{sal_metric_name}_{partition}.npy', allow_pickle = True)
        picker = random.randrange(len(values))
        print(f'in dataset: {data} for partition: {partition} -- {values[picker]}')
