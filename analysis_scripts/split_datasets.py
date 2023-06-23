import matplotlib.pyplot as plt
import numpy as np
import os.path
import os
import cv2 as cv

#decide which saliency group each image falls into
def and_or_threshold(scores):
    under_two = []
    two_to_four = []
    four_to_six = []
    six_to_eight = []
    above_eight = []
    for i in range(len(scores)): #listing through all scores
        if(float(scores[i][1]) <= .20):
            under_two.append(scores[i][0])
        elif(float(scores[i][1])  > .20 and float(scores[i][1]) <= .40):
            two_to_four.append(scores[i][0])
        elif(float(scores[i][1])  > .40 and float(scores[i][1]) <= .60):
            four_to_six.append(scores[i][0])
        elif(float(scores[i][1])  > .60 and float(scores[i][1]) <= .80):
            six_to_eight.append(scores[i][0])
        elif(float(scores[i][1])  > .80):
            above_eight.append(scores[i][0])

    return (under_two,two_to_four,four_to_six,six_to_eight,above_eight) #return lists

def load_numpy(path):
    values = np.load(path, allow_pickle=True)
    return values


sal_metric_name = "mean_recall"
sal_metric_display = "mean recall"
split_metric_name = "and_or_threshold"

#all datasets to run
datasets = ['korus', 'korus_size_up','korus_SE','MFC18_small','MFC18_resize', 'MFC18_SE', 'imd2020', 'imd2020_resize', 'imd2020_SE']
for i in datasets:

    #name formatting for graphs
    dataset_display = i
    if(i == 'korus_SE'):
        dataset_display = 'Korus Saliency Enhanced'
    if(i == 'korus'):
        dataset_display = 'Korus'
    if(i == 'korus_size_up'):
        dataset_display = 'Korus Resized'


    if(i == 'MFC18_small'):
        dataset_display = 'MFC18'
    if(i == 'MFC18_SE'):
        dataset_display = 'MFC18 Saliency Enhanced'
    if(i == 'MFC18_resize'):
        dataset_display = 'MFC18 Resized'

    if(i == 'imd2020'):
        dataset_display = 'IMD2020'
    if(i == 'imd2020_SE'):
        dataset_display = 'IMD2020 Saliency Enhanced'
    if(i == 'imd2020_resize'):
        dataset_display = 'IMD2020 Resized'

    #load scores
    scores = load_numpy(f'scores/combine_maps_{sal_metric_name}_{i}.npy')

    #human split -- only on korus dataset
    if(i == 'korus' or i == 'korus_SE'):
        sal_scores = load_numpy(f'scores/human_study_sal_{sal_metric_name}_{i}.npy')
        under_two,two_to_four,four_to_six,six_to_eight, above_eight = and_or_threshold(sal_scores)
        np.save(f'data_splits/5_split/{i}_{sal_metric_name}_human_under_two', under_two)
        np.save(f'data_splits/5_split/{i}_{sal_metric_name}_human_two_to_four', two_to_four)
        np.save(f'data_splits/5_split/{i}_{sal_metric_name}_human_four_to_six', four_to_six)
        np.save(f'data_splits/5_split/{i}_{sal_metric_name}_human_six_to_eight', six_to_eight)
        np.save(f'data_splits/5_split/{i}_{sal_metric_name}_human_above_eight', above_eight)

        #plot graph -- for human map split
        max_len_h = np.max([len(under_two),len(two_to_four),len(four_to_six), len(six_to_eight), len(above_eight)])
        plt.bar(['<.2', '.2-.4','.4-.6','.6-.8', '>.8'],[len(under_two),len(two_to_four),len(four_to_six), len(six_to_eight), len(above_eight)])
        plt.xlabel("Saliency Of Manipulation Using Mean Recall")
        plt.ylabel("Number Of Images")
        graph_max_h =int(max_len_h + (max_len_h*.05))
        plt.ylim(0,graph_max_h)
        plt.title(f"Distribution Of {dataset_display}")
        plt.show()
        plt.savefig(f'data_splits/5_split/histogram/splits_graph_{i}_{sal_metric_name}_human.png')
        print(f'data_splits/5_split/histogram/splits_graph_{i}_{sal_metric_name}_human.png')
        plt.clf()


    #combined map split
    under_two,two_to_four,four_to_six,six_to_eight,above_eight = and_or_threshold(scores)
    np.save(f'data_splits/5_split/{i}_{sal_metric_name}_{split_metric_name}_under_two', under_two)
    np.save(f'data_splits/5_split/{i}_{sal_metric_name}_{split_metric_name}_two_to_four', two_to_four)
    np.save(f'data_splits/5_split/{i}_{sal_metric_name}_{split_metric_name}_four_to_six', four_to_six)
    np.save(f'data_splits/5_split/{i}_{sal_metric_name}_{split_metric_name}_six_to_eight', six_to_eight)
    np.save(f'data_splits/5_split/{i}_{sal_metric_name}_{split_metric_name}_above_eight', above_eight)

    #plot graph -- for combined map split
    plt.bar(['<.2', '.2-.4','.4-.6','.6-.8', '>.8'],[len(under_two),len(two_to_four),len(four_to_six), len(six_to_eight), len(above_eight)])
    max_len = np.max([len(under_two),len(two_to_four),len(four_to_six), len(six_to_eight), len(above_eight)])
    plt.xlabel("Saliency Of Manipulation Using Mean Recall")
    plt.ylabel("Number Of Images")
    graph_max =int(max_len + (max_len*.05))
    plt.ylim(0,graph_max)
    plt.title(f"Distribution Of {dataset_display}")
    plt.show()
    plt.savefig(f'data_splits/5_split/histogram/splits_graph_{i}_{sal_metric_name}.png')
    print(f'data_splits/5_split/histogram/splits_graph_{i}_{sal_metric_name}.png')
    plt.clf()