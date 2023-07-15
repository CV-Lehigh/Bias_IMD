import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice
from matplotlib import colors as mcolors

import pandas as pd

def load_numpy(path):
    values = np.load(path, allow_pickle=True)
    return values

manip_metric_name = "ROC"
sal_metric_name = "mean_recall"
sal_metric_display = "Mean Recall"
split_metric_name = "and_or_threshold"


def partition_mean_auc(partition_list, manip_score_list):
    partition_aucs = []
    count = 0
    for filename in partition_list:
        try:
            index = np.where(manip_score_list[:,0] == filename[:len(filename)-4]+'.jpg')[0][0]
            partition_aucs.append(float(manip_score_list[index][1]))

        except: 
            try:
                index = np.where(manip_score_list[:,0] == filename[:len(filename)-4]+'.PNG')[0][0]
                partition_aucs.append(float(manip_score_list[index][1]))
            except:
                try:
                    index = np.where(manip_score_list[:,0] == filename[:len(filename)-4]+'.png')[0][0]
                    partition_aucs.append(float(manip_score_list[index][1]))
                except:
                    count += 1
    # print(f'failed: {count}')
    return (np.mean(partition_aucs)*100), count

def all_partitions_auc(network, dataset):
    split_type = dataset
    if(dataset == 'korus_SE'):
        split_type = 'korus_size_up'
    elif(dataset == 'imd2020_SE'):
        split_type = 'imd2020_resize'
    elif(dataset == 'MFC18_SE'):
        split_type = 'MFC18_resize'

    mAUC = []
    sec_length = []
    partition = ['under_two', 'two_to_four', 'four_to_six', 'six_to_eight', 'above_eight']
    manip_scores = load_numpy(f'./WIFS_ready/manipulation_scores/{network}_{manip_metric_name}_{dataset}.npy')
    for p in partition:
        section = load_numpy(f'./WIFS_ready/sal_groups/{split_type}_{sal_metric_name}_{split_metric_name}_{p}.npy')
        length = len(section)
        mean, failed = partition_mean_auc(section, manip_scores)
        mAUC.append(mean)
        sec_length.append((length-failed))
    return (mAUC,sec_length)


# all_data = ['imd2020','imd2020_resize','imd2020_SE', 'MFC18_small', 'MFC18_resize', 'MFC18_SE']

#To include human results with a korus dataset un-comment:
all_data = ['korus','MFC18_small', 'imd2020', 'korus_size_up', 'MFC18_resize', 'imd2020_resize', 'korus_SE', 'MFC18_SE', 'imd2020_SE']
for dataset in all_data:
    dataset_display = dataset
    if(dataset == 'korus_SE'):
        dataset_display = 'Saliency Enhanced'
    if(dataset == 'korus'):
        dataset_display = 'Realistic Tampering'
    if(dataset == 'korus_size_up'):
        dataset_display = 'Resized'


    if(dataset == 'MFC18_small'):
        dataset_display = 'MFC18'
    if(dataset == 'MFC18_SE'):
        dataset_display = 'MFC18 Saliency Enhanced'
    if(dataset == 'MFC18_resize'):
        dataset_display = 'MFC18 Resized'


    if(dataset == 'imd2020'):
        dataset_display = 'IMD2020'
    if(dataset == 'imd2020_SE'):
        dataset_display = 'IMD2020 Saliency Enhanced'
    if(dataset == 'imd2020_resize'):
        dataset_display = 'IMD2020 Resized'

    plt.figure(figsize=(8,10))
    PSCC, p_len = all_partitions_auc('pscc', dataset)
    OSN, o_len = all_partitions_auc('osn', dataset)
    ManTraNet, m_len = all_partitions_auc('mantranet', dataset)
    BusterNet, b_len = all_partitions_auc('busternet', dataset)
    #1. To include human results with a korus dataset un-comment:
    
    if(dataset == 'korus_size_up'):
        human_study, h_len = all_partitions_auc('human_manip', 'korus') 
    elif(dataset == 'korus' or dataset == 'korus_SE'):
        human_study, h_len = all_partitions_auc('human_manip', dataset) 
    

    idx_0 = f'<.2 ({p_len[0]})'
    idx_1 = f'.2-.4 ({p_len[1]})'
    idx_2 = f'.4-.6 ({p_len[2]})'
    idx_3 = f'.6-.8 ({p_len[3]})'
    idx_4 = f'>.8 ({p_len[4]})'

    #2. To include human results with a korus dataset comment out next two lines:
    _data = {'PSCC-Net': PSCC, 'OSN': OSN, 'ManTra-Net':ManTraNet, 'BusterNet': BusterNet}
    _df = pd.DataFrame(_data,columns=['BusterNet', 'PSCC-Net', 'ManTra-Net', 'OSN'], index = [idx_0, idx_1, idx_2, idx_3, idx_4])

    amt = 20
    my_colors = list(islice(cycle([	(131/255.0,178/255.0,208/255.0,1.0), (226/255.0, 116/255.0, 41/255.0,1.0),	(149/255.0,218/255.0,182/255.0,1.0),(220/255.0,133/255.0,128/255.0,1.0)]), None, len(_df)))
    ax = _df.plot.bar(align='center', width = .6, color = my_colors)
    
    #3. To include human results with a korus dataset un-comment:
    if('korus' in dataset):
        _data = {'BusterNet': BusterNet,'PSCC-Net': PSCC, 'ManTra-Net':ManTraNet, 'OSN': OSN, 'Human Pred.': human_study}
        _df = pd.DataFrame(_data,columns=['BusterNet','PSCC-Net', 'ManTra-Net','OSN', 'Human Pred.'], index = [idx_0, idx_1, idx_2, idx_3, idx_4])
        my_colors = list(islice(cycle([	(131/255.0,178/255.0,208/255.0,1.0), (226/255.0, 116/255.0, 41/255.0,1.0),	(149/255.0,218/255.0,182/255.0,1.0),(220/255.0,133/255.0,128/255.0,1.0), 'tab:gray']), None, len(_df)))
        ax = _df.plot.bar(align='center', width = .6, color = my_colors)
        amt +=5

    # Multiple bar chart
    #.18
    

    if('korus' in dataset):
        # ax.patches[20].set_width(.18)
        # ax.patches[20].set_x(.22)
        ax.patches[20].set_hatch('//')
        ax.patches[20].set_fill(False)
        # ax.patches[21].set_width(.18)
        # ax.patches[21].set_x(1.22)
        ax.patches[21].set_hatch('//')
        ax.patches[21].set_fill(False)
        # ax.patches[22].set_width(.18)
        # ax.patches[22].set_x(2.22)
        ax.patches[22].set_hatch('//')
        ax.patches[22].set_fill(False)
        # ax.patches[23].set_width(.18)
        # ax.patches[23].set_x(3.22)
        ax.patches[23].set_hatch('//')
        ax.patches[23].set_fill(False)
        # ax.patches[24].set_width(.18)
        # ax.patches[24].set_x(4.22)
        ax.patches[24].set_hatch('//')
        ax.patches[24].set_fill(False)


    #4. To include human results with a korus dataset change 20 to 25:
    for p in range(amt):
        val = 10
        ax.annotate(f'{ax.patches[p].get_height():2.0f}', (ax.patches[p].get_x(), ax.patches[p].get_height() * 1.015), fontsize = val, rotation=45)
        
    # Display the plot
    ax.set_xticklabels([idx_0, idx_1, idx_2, idx_3, idx_4], rotation=0, ha='center',fontsize=12)
    plt.xlabel(f"Saliency Of Manipulation Using {sal_metric_display}", labelpad=12, fontsize=12)
    plt.ylabel(f"Average AuROC", labelpad=12, fontsize =12)
    plt.yticks(fontsize=12)
    plt.ylim(0,110)
    plt.title(f"{dataset_display}")
    plt.tight_layout()
    plt.legend(loc='lower left', title='Networks')
    plt.show()
    plt.savefig(f"./WIFS_ready/performance/{dataset}_{sal_metric_name}_{split_metric_name}_{manip_metric_name}.png")
    print(f"./WIFS_ready/performance/{dataset}_{sal_metric_name}_{split_metric_name}_{manip_metric_name}.png")
    plt.clf()
