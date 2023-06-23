import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt



def load_data_split(path):
    split_values = np.load(path, allow_pickle=True)
    return split_values

def movement_table(OG, new):
    move_table = []
    for image in OG:
        for new_split in new:
                if(image[0][:len(image[0])-3]+'jpg' == new_split[0]):
                    move_table.append([image[0], (float(new_split[1])-float(image[1])), new_split[1], image[1]])
                    continue
    return move_table


def move_table_stats(table):
    total_movement = 0.0
    value_list = []
    for values in table:
        value_list.append(values[1])
        total_movement += values[1]
    avg_movement = (total_movement/len(table))
    print(avg_movement)
    return value_list



original_mean_recall = load_data_split(f'combine_maps_mean_recall_korus_size_up.npy')
new_mean_recall = load_data_split(f'combine_maps_mean_recall_korus_SE.npy')


# move_table, OG_split, new_split = movement_table(original_all, new_all)print(new_mean_recall)
delta_recall = movement_table(original_mean_recall, new_mean_recall)
for item in delta_recall:
    print(item)
dp = move_table_stats(delta_recall)
plt.boxplot(dp)
plt.savefig('score_delta.jpg')
