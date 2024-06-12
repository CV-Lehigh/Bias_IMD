import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice
from matplotlib import colors as mcolors
plt.switch_backend('agg')


import pandas as pd

def load_numpy(path):
    values = np.load(path, allow_pickle=True)
    return values

manip_metric_name = "ROC"
sal_metric_name = "mean_recall"
sal_metric_display = "Mean Recall"
split_metric_name = "and_or_threshold"


def partition_mean_auc(partition_list, manip_score_list, ro = True):
    partition_aucs = []
    count = 0
    for filename in partition_list:

        if (ro == True and (filename == 'DPP0122.PNG' or filename == 'DSC06659.PNG')): #removing outliers
            continue
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

def all_partitions_auc(network, dataset,  ro = False, use_human=False):
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
    manip_scores = load_numpy(f'./ICIP_2024/manipulation_scores/{network}_{manip_metric_name}_{dataset}.npy')
    for p in partition:
        section = load_numpy(f'./ICIP_2024/sal_groups/{split_type}_{sal_metric_name}_{split_metric_name}_{p}.npy') if not use_human else load_numpy(f'./ICIP_2024/sal_groups/{split_type}_{sal_metric_name}_human_{p}.npy') 
        length = len(section)
        mean, failed = partition_mean_auc(section, manip_scores,  ro = False)
        mAUC.append(mean)
        sec_length.append((length-failed))
    return (mAUC,sec_length)


# all_data = ['imd2020','imd2020_resize','imd2020_SE', 'MFC18_small', 'MFC18_resize', 'MFC18_SE']
all_sets = [['korus_size_up', 'korus_SE'], ['korus', 'MFC18_small', 'imd2020'], ['imd2020_resize', 'imd2020_SE'], ['korus']]
for j in range(len(all_sets)):
    fig, axes = plt.subplots(1,2, sharey=True, figsize=(30, 10))
    if j == 1:
        fig, axes = plt.subplots(1,3, sharey=True, figsize=(45, 10))
    if j == 3:
        fig, axes = plt.subplots(1,1, sharey=True, figsize=(15, 10))
    idx = 0

    prefix = ''
    if j==0:
        prefix = 'korus_SE_'
    if j == 2:
        prefix = 'imd2020_SE_'
    if j==3:
        prefix = 'korus_human_'
    #To include human results with a korus dataset un-comment:
    for dataset in all_sets[j]:
        dataset_display = dataset
        if(dataset == 'korus_SE'):
            dataset_display = 'Saliency Enhanced'
        if(dataset == 'korus'):
            dataset_display = 'Realistic Tampering'
        if(dataset == 'korus_size_up'):
            dataset_display = 'Original Saliency'



        if(dataset == 'MFC18_small'):
            dataset_display = 'MFC18'
        if(dataset == 'MFC18_SE'):
            dataset_display = 'MFC18 Saliency Enhanced'
        if(dataset == 'MFC18_resize'):
            dataset_display = 'Original Saliency'



        if(dataset == 'imd2020'):
            dataset_display = 'IMD2020'
        if(dataset == 'imd2020_SE'):
            dataset_display = 'IMD2020 Saliency Enhanced'
        if(dataset == 'imd2020_resize'):
            dataset_display = 'Original Saliency'

        PSCC, p_len = all_partitions_auc('pscc', dataset)
        OSN, o_len = all_partitions_auc('osn', dataset)
        ManTraNet, m_len = all_partitions_auc('mantranet', dataset)
        BusterNet, b_len = all_partitions_auc('busternet', dataset)
        #1. To include human results with a korus dataset un-comment:
        
        if(dataset == 'korus_size_up'):
            human_study, h_len = all_partitions_auc('redo_human_manip', 'korus') 
        if(dataset == 'korus'):
            human_study, h_len = all_partitions_auc('redo_human_manip', dataset, ro = False)
            if(j==3):
                human_study, h_len = all_partitions_auc('redo_human_manip', dataset, ro = False, use_human=True)
        if(dataset == 'korus_SE'):
            human_study, h_len = all_partitions_auc('redo_human_manip', dataset) 
        

        idx_0 = f'<.2 ({p_len[0]})'
        idx_1 = f'.2-.4 ({p_len[1]})'
        idx_2 = f'.4-.6 ({p_len[2]})'
        idx_3 = f'.6-.8 ({p_len[3]})'
        idx_4 = f'>.8 ({p_len[4]})'
        if  j == 0:
            idx_4 = f'>.8 (10)'

        if j == 3:
            idx_0 = f'<.2 ({h_len[0]})'
            idx_1 = f'.2-.4 ({h_len[1]})'
            idx_2 = f'.4-.6 ({h_len[2]})'
            idx_3 = f'.6-.8 ({h_len[3]})'
            idx_4 = f'>.8 ({h_len[4]})'

        #2. To include human results with a korus dataset comment out next two lines:
        amt = 20
        if('korus' not in dataset):
            _data = {'PSCC-Net': PSCC, 'OSN': OSN, 'ManTra-Net':ManTraNet, 'BusterNet': BusterNet}
            _df = pd.DataFrame(_data,columns=['BusterNet', 'PSCC-Net', 'ManTra-Net', 'OSN'], index = [idx_0, idx_1, idx_2, idx_3, idx_4])

            amt = 20
            my_colors = list(islice(cycle([	(131/255.0,178/255.0,208/255.0,1.0), (226/255.0, 116/255.0, 41/255.0,1.0),	(149/255.0,218/255.0,182/255.0,1.0),(220/255.0,133/255.0,128/255.0,1.0)]), None, len(_df)))
            _df.plot.bar(align='center', width = .78, color = my_colors, ax=axes[idx])
        
        #3. To include human results with a korus dataset un-comment:
        if('korus' in dataset):
            _data = {'BusterNet': BusterNet,'PSCC-Net': PSCC, 'ManTra-Net':ManTraNet, 'OSN': OSN, 'Human Pred.': human_study}
            _df = pd.DataFrame(_data,columns=['BusterNet','PSCC-Net', 'ManTra-Net','OSN', 'Human Pred.'], index = [idx_0, idx_1, idx_2, idx_3, idx_4])
            my_colors = list(islice(cycle([	(131/255.0,178/255.0,208/255.0,1.0), (226/255.0, 116/255.0, 41/255.0,1.0),	(149/255.0,218/255.0,182/255.0,1.0),(220/255.0,133/255.0,128/255.0,1.0), 'tab:gray']), None, len(_df)))
            if(j!=3):
                _df.plot.bar(align='center', width = .74, color = my_colors, ax=axes[idx])
            amt +=5
            if j == 3:
                _data = {'Human Pred.': human_study}
                _df = pd.DataFrame(_data,columns=['Human Pred.'], index = [idx_0, idx_1, idx_2, idx_3, idx_4])
                my_colors = list(islice(cycle([	(131/255.0,178/255.0,208/255.0,1.0), (226/255.0, 116/255.0, 41/255.0,1.0),	(149/255.0,218/255.0,182/255.0,1.0),(220/255.0,133/255.0,128/255.0,1.0), 'tab:gray']), None, len(_df)))
                _df.plot.bar(align='center', width = .74, color = my_colors, ax=axes)
                amt = 5

        # Multiple bar chart
        #.18
        if('korus' in dataset and j != 3):
            # ax.patches[20].set_width(.18)
            # ax.patches[20].set_x(.22)
            axes[idx].patches[20].set_hatch('//')
            axes[idx].patches[20].set_fill(False)
            # ax.patches[21].set_width(.18)
            # ax.patches[21].set_x(1.22)
            axes[idx].patches[21].set_hatch('//')
            axes[idx].patches[21].set_fill(False)
            # ax.patches[22].set_width(.18)
            # ax.patches[22].set_x(2.22)
            axes[idx].patches[22].set_hatch('//')
            axes[idx].patches[22].set_fill(False)
            # ax.patches[23].set_width(.18)
            # ax.patches[23].set_x(3.22)
            axes[idx].patches[23].set_hatch('//')
            axes[idx].patches[23].set_fill(False)
            # ax.patches[24].set_width(.18)
            # ax.patches[24].set_x(4.22)
            axes[idx].patches[24].set_hatch('//')
            axes[idx].patches[24].set_fill(False)


        #4. To include human results with a korus dataset change 20 to 25:
        for p in range(amt):
            val = 25
            if j != 3:
                axes[idx].annotate(f'{axes[idx].patches[p].get_height():2.0f}', (axes[idx].patches[p].get_x(), axes[idx].patches[p].get_height() * 1.015), fontsize = val, rotation=0)
            else:
                axes.annotate(f'{axes.patches[p].get_height():2.0f}', (axes.patches[p].get_x(), axes.patches[p].get_height() * 1.015), fontsize = val, rotation=0)
            
        # Display the plot
        if j!=3:
            axes[idx].set_xticklabels([idx_0, idx_1, idx_2, idx_3, idx_4], rotation=0, ha='center',fontsize=33)
            axes[idx].set_title(f"{dataset_display}", fontsize=35)
            axes[idx].set_yticklabels(([0,20,40,60,80,100]), rotation=0, fontsize=33)
            axes[idx].legend(loc='lower left', fontsize=25)
            idx +=1
        else:
            axes.set_xticklabels([idx_0, idx_1, idx_2, idx_3, idx_4], rotation=0, ha='center',fontsize=33)
            axes.set_title(f"{dataset_display}", fontsize=35)
            axes.set_yticklabels(([0,20,40,60,80,100]), rotation=0, fontsize=33)
            axes.legend(loc='lower left', fontsize=25)
            idx +=1

    fig.text(0.5, 0.02, f"Saliency Of Manipulation [{sal_metric_display}]", fontsize=30, ha='center')
    fig.text(0.005, 0.5, f"Detection Performance [Avg. AuROC]", fontsize =30, va='center', rotation='vertical')
    # plt.yticks(fontsize=60)
    plt.ylim(0,105)
    plt.tight_layout(pad=4.8)
    plt.subplots_adjust(wspace=0.01)
    fig.show()
    fig.savefig(f"./ICIP_2024/performance/{prefix}performance.pdf", dpi=2000)
    print(f"./ICIP_2024/performance/{prefix}performance.pdf")
    plt.clf()
