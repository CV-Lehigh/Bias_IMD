import matplotlib.pyplot as plt
import numpy as np
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
                    print(f'failed: {filename}')
    return np.mean(partition_aucs)

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
    manip_scores = load_numpy(f'scores/{network}_{manip_metric_name}_{dataset}.npy')
    for p in partition:
        section = load_numpy(f'data_splits/5_split/{split_type}_{sal_metric_name}_{split_metric_name}_{p}.npy')
        length = len(section)
        mAUC.append(partition_mean_auc(section, manip_scores))
        sec_length.append(length)
    return (mAUC,sec_length)


all_data = ['imd2020','imd2020_resize','imd2020_SE', 'MFC18_small', 'MFC18_resize', 'MFC18_SE']

#To include human results with a korus dataset un-comment:
'''all_data = ['korus', 'korus_size_up', korus_SE']'''
for dataset in all_data:
    dataset_display = dataset
    if(dataset == 'korus_SE'):
        dataset_display = 'Korus Saliency Enhanced'
    if(dataset == 'korus'):
        dataset_display = 'Korus'
    if(dataset == 'korus_size_up'):
        dataset_display = 'Korus Resized'


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

    plt.figure(figsize=(6,6))
    PSCC, p_len = all_partitions_auc('pscc', dataset)
    OSN, o_len = all_partitions_auc('osn', dataset)
    ManTraNet, m_len = all_partitions_auc('mantranet', dataset)
    BusterNet, b_len = all_partitions_auc('busternet', dataset)
    #1. To include human results with a korus dataset un-comment:
    ''' 
    if(dataset == 'korus_size_up'):
        human_study, h_len = all_partitions_auc('human_study_manip', 'korus') 
    else:
        human_study, h_len = all_partitions_auc('human_study_manip', dataset) 
    '''


    #2. To include human results with a korus dataset comment out next two lines:
    _data = {'PSCC': PSCC, 'OSN': OSN, 'ManTraNet':ManTraNet, 'BusterNet': BusterNet}
    _df = pd.DataFrame(_data,columns=['PSCC', 'OSN', 'ManTraNet', 'BusterNet'], index = ['<.2', '.2-.4', '4.-.6', '.6-.8', '>.8'])

    #3. To include human results with a korus dataset un-comment:
    '''
    _data = {'PSCC': PSCC, 'OSN': OSN, 'ManTraNet':ManTraNet, 'BusterNet': BusterNet, 'Human Pred.': human_study}
    _df = pd.DataFrame(_data,columns=['PSCC', 'OSN', 'ManTraNet', 'BusterNet', 'Human Pred.'], index = ['<.2', '.2-.4', '4.-.6', '.6-.8', '>.8'])
    '''

    # Multiple bar chart
    ax = _df.plot.bar()

    #4. To include human results with a korus dataset change 20 to 25:
    for p in range(20):
        ax.annotate(f'{ax.patches[p].get_height():.2f}', (ax.patches[p].get_x() -.07, ax.patches[p].get_height() * 1.005), fontsize = 10)
        
    # Display the plot
    plt.xlabel(f"Saliency Of Manipulation Using {sal_metric_display}")
    plt.ylabel(f"Average Area Under {manip_metric_name}")
    plt.ylim(0.0, 1.05)
    plt.title(f"{dataset_display} Manipulation Detection")
    plt.tight_layout()
    plt.legend(loc='lower left', title='Networks')
    plt.show()
    plt.savefig(f"mAUC_wrt_saliency_graphs/{dataset}_{sal_metric_name}_{split_metric_name}_{manip_metric_name}.png")
    print(f"mAUC_wrt_saliency_graphs/{dataset}_{sal_metric_name}_{split_metric_name}_{manip_metric_name}.png")
    plt.clf()