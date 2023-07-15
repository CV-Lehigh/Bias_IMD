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
sums = []
lens = []

datasets = ['korus', 'MFC18_small', 'imd2020']
for i in datasets:

    #name formatting for graphs
    dataset_display = i
    if(i == 'korus_SE'):
        dataset_display = 'Realistic Tampering Saliency Enhanced'
    if(i == 'korus'):
        dataset_display = 'Realistic Tampering'
    if(i == 'korus_size_up'):
        dataset_display = 'Realistic Tampering Resized'

    #load scores
    scores = load_numpy(f'./WIFS_ready/saliency_scores/combine_maps_{sal_metric_name}_{i}.npy')

    #human split -- only on korus dataset
    if(i == 'korus' or i == 'korus_SE'):
        sal_scores = load_numpy(f'./WIFS_ready/saliency_scores/human_sal_{sal_metric_name}_{i}.npy')
        under_two,two_to_four,four_to_six,six_to_eight, above_eight = and_or_threshold(sal_scores)
        np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_human_under_two', under_two)
        np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_human_two_to_four', two_to_four)
        np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_human_four_to_six', four_to_six)
        np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_human_six_to_eight', six_to_eight)
        np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_human_above_eight', above_eight)

        #plot graph -- for human map split
        sum = (len(under_two)+len(two_to_four)+len(four_to_six)+len(six_to_eight)+len(above_eight))
        plt.bar(['<.2', '.2-.4','.4-.6','.6-.8', '>.8'],[len(under_two)/sum,len(two_to_four)/sum,len(four_to_six)/sum, len(six_to_eight)/sum, len(above_eight)/sum],label = 'Realistic Tampering (130)', color = (131/255.0,178/255.0,208/255.0,1.0))
        plt.xlabel("Saliency Of Manipulation Using Mean Recall")
        plt.ylabel("Proportion Of Images")
        plt.ylim(0,.35)
        plt.title(f"Distribution Of {dataset_display}")
        plt.show()
        plt.savefig(f'./WIFS_ready/sal_groups/histogram/splits_graph_{i}_{sal_metric_name}_human.png')
        print(f'./WIFS_ready/sal_groups/histogram/splits_graph_{i}_{sal_metric_name}_human.png')
        plt.clf()


    #combined map split
    under_two,two_to_four,four_to_six,six_to_eight,above_eight = and_or_threshold(scores)
    np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_{split_metric_name}_under_two', under_two)
    np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_{split_metric_name}_two_to_four', two_to_four)
    np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_{split_metric_name}_four_to_six', four_to_six)
    np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_{split_metric_name}_six_to_eight', six_to_eight)
    np.save(f'./WIFS_ready/sal_groups/{i}_{sal_metric_name}_{split_metric_name}_above_eight', above_eight)

    #plot graph -- for combined map split
    
    sum = (len(under_two)+len(two_to_four)+len(four_to_six)+len(six_to_eight)+len(above_eight))
    print(len(under_two))
    print(len(two_to_four))
    print(len(four_to_six))
    print(len(six_to_eight))
    print(len(above_eight))
    lens.append([len(under_two)/sum,len(two_to_four)/sum,len(four_to_six)/sum, len(six_to_eight)/sum, len(above_eight)/sum])

X = ['<.2', '.2-.4','.4-.6','.6-.8', '>.8']

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, lens[0], 0.2, label = 'Realistic Tampering (130)', color = (131/255.0,178/255.0,208/255.0,1.0))
plt.bar(X_axis, lens[1], 0.2, label = 'MFC18 (1127)', color = (226/255.0, 116/255.0, 41/255.0,1.0) )
plt.bar(X_axis + 0.2, lens[2], 0.2, label = 'IMD2020 (2007)', color = (149/255.0,218/255.0,182/255.0,1.0))

plt.xticks(X_axis, X)
    
plt.xlabel("Saliency Of Manipulation Using Mean Recall")
plt.ylabel("Proportion Of Images")
plt.ylim(0,.55)

plt.title(f"Distribution of Datasets")
plt.legend()
plt.show()
plt.savefig(f'./WIFS_ready/sal_groups/histogram/splits_graph.png')
print(f'./WIFS_ready/sal_groups/histogram/splits_graph.png')
plt.clf()
