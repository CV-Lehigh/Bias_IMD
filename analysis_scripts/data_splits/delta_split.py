import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt



def load_data_split(path):
    split_values = np.load(path, allow_pickle=True)
    return split_values

def movement_table(OG, new):
    OG_split = []
    updated_split = []
    move_table = []
    for split in range(5):
        for image in OG[split]:
            OG_split.append(split)
            for new_split in range(5):
                if(image[:len(image)-3]+'jpg' in new[new_split]):
                    move_table.append([image, (new_split-split), new_split, split])
                    updated_split.append(new_split)
    return move_table, OG_split, updated_split


def move_table_stats(table):
    total_movement = 0.0
    for values in table:
        total_movement += values[1]
        print(values)
    avg_movement = (total_movement/len(table))
    print(avg_movement)

dataset = ['korus', 'imd2020', 'MFC18']

for data in dataset:
    ext = 'resize'
    if(data == 'korus'):
        ext = 'size_up'
        
    original_all = []
    new_all = []
    for split in ['under_two', 'two_to_four', 'four_to_six', 'six_to_eight', 'above_eight']:
        original_all.append(load_data_split(f'5_split/{data}_{ext}_mean_recall_{split}.npy'))
        new_all.append(load_data_split(f'5_split/{data}_SE_mean_recall_{split}.npy'))


    move_table, OG_split, new_split = movement_table(original_all, new_all)
    print(OG_split)
    print(new_split)
    move_table_stats(move_table)
    confusion_matrix = metrics.confusion_matrix(OG_split, new_split)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['<.2','.2-.4','.4-.6','.6-.8','>.8'])
    cm_display.plot()
    plt.xlabel(f'{data} Saliency Enhanced Split')
    plt.ylabel(f'{data} Split')
    plt.title(f'{data} Saliency Movement')
    plt.show()
    plt.savefig(f'./movement_matrix_SE_{data}.png')
    plt.clf()