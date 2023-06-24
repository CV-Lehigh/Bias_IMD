import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_numpy(path):
    values = np.load(path, allow_pickle=True)
    return values

manip_metric_name = "ROC"
sal_metric_name = "mean_recall"
split_metric_name = "and_or_threshold"


def partition_mean_auc(partition_list, manip_score_list):
    partition_aucs = []
    for filename in partition_list:
            index = np.where(manip_score_list[:,0] == filename)[0][0]
            partition_aucs.append(float(manip_score_list[index][1]))
    return partition_aucs

def all_partitions_auc(dataset):
    mAUC = []
    sec_length = []
    partition = ['under_two', 'two_to_four', 'four_to_six', 'six_to_eight', 'above_eight']
    manip_scores = load_numpy(f'manip_size/GT_{dataset}.npy')
    for p in partition:
        section = load_numpy(f'data_splits/5_split/{dataset}_{sal_metric_name}_{p}.npy')
        length = len(section)
        mAUC.append(partition_mean_auc(section, manip_scores))
        sec_length.append(length)
    return (mAUC,sec_length)

all_data = ['korus']
for dataset in all_data:
    GT_size, h_len = all_partitions_auc(dataset)


    vals = [h_len[0], h_len[1], h_len[2], h_len[3], h_len[4]]

    _data = {'<.2': GT_size[0], '.2-.4': GT_size[1], '.4-.6': GT_size[2], '.6-.8': GT_size[3], '>.8': GT_size[4]}
    _df = pd.concat([pd.DataFrame(v, columns=[k]) for k, v in _data.items()], axis=1)
    print(_df)
        
    
    # Multiple bar chart
    g_1 = 0 #length of group 1
    g_2 = 0 #length of group 2
    g_3 = 0 #length of group 3
    g_4 = 0 #length of group 4
    g_5 = 0 #length of group 5
    plt.boxplot(x = [_df['<.2'][:g_1], _df['.2-.4'][:g_2], _df['.4-.6'][:g_3], _df['.6-.8'][:g_4], _df['>.8'][:g_5]])
        
    plt.xlabel(f"Manipulation saliency Using Combined map {sal_metric_name}")
    plt.ylabel(f"% Manipulated")
    plt.xticks([1, 2, 3, 4,5], ['<.2', '.2-.4', '.4-.6', '.6-.8', '>.8'])
    plt.title(f"% of manipulated split by saliency of manipulation")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"manip_size/{dataset}_manip_size.png")
    plt.clf()
