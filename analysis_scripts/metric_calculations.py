import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import os.path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from PIL import Image

'''
Function: Load all images from path inputted
Input: Path to folder of images
Output: (filename, image content)
'''
def load_images_from_path(path):
    images = []
    for filename in tqdm(os.listdir(path)):
            lower_filename = filename.lower() #Not used -- nessar in previous version

            full_name = os.path.join(path,filename) #full path name
            image = cv.imread(full_name, 0).astype(np.float64) #load in with doubles (needed for some masks betwwen 0-1)
            images.append((filename,image)) #add to list
    return images #return full values

'''
def load_images_from_path_coco(path):
    images = []
    for filename in tqdm(os.listdir(path)):
            lower_filename = filename.lower()
            full_name = os.path.join(path,filename)
            image = cv.imread(full_name, 0).astype(np.int16)
            images.append((filename,image))
    return images

def load_image_coco(path, img_name, part_1, part_2):
    #needs to be changed per dataset
    sub_section = img_name.split('_')
    for filename in sorted(os.listdir(path)):
        file_parts = filename.split("_")
        if (sub_section[1] == file_parts[part_1]) and (sub_section[2][:len(sub_section[2])-4] == file_parts[part_2]):
        # if(img_name[:len(img_name)-3] in filename):
            path = os.path.join(path,filename)
            image_attr = Image.open(path).convert('L')
            image = np.array(image_attr).astype(np.float64)
            # image = cv.imread(path,0).astype(np.float64)
            return (filename,image)
'''

'''
Function: search for image in list and load Specific image
Input: path to image, image name
Output: (image name, image content)
'''
def load_image(path, img_name):
    for filename in sorted(os.listdir(path)):
        if(img_name[:len(img_name)-3] in filename):
            path = os.path.join(path,filename)
            image = cv.imread(path,0).astype(np.float64)
            return (filename,image)

'''
Function: Get recall of an image at multiple thresholds 0,.2,.4,.6,.8,1.0
Input: ground truth mask (0-1), saliency map (0-1)
Output: array of recall scores
'''
def std_recall(groundtruth, sal_mask):
    recall = []
    for thresh in range(0,10,2):
        _thresh = thresh/10.0
        TP = 0.0
        FN = 0.0
        ret, sal_thresh = cv.threshold(sal_mask, _thresh, 1, cv.THRESH_BINARY) #theshold mask
        for i in range(len(groundtruth)): #loop thorugh image
            if (groundtruth[i] == 1):
                if(sal_thresh[i] == 1): #True positive
                    TP += 1
                elif(sal_thresh[i] == 0): #False Negative
                    FN += 1
        thresh_recall = (TP/(TP+FN)) #recall for that threshold
        recall.append(thresh_recall) #add to list
    return recall

'''
Function: get average recall value
Input: Input: ground truth mask (0-1), saliency map (0-1)
Output: avg recall
'''
def mean_recall(groundtruth, sal_mask):
    recall = std_recall(groundtruth,sal_mask)
    avg_recall = np.mean(recall)
    return avg_recall

'''
Function: get the area under a PR-Curve 
WARNING -- UNUSED
'''
def PR_AUC(groundtruth, mask):
    precision, recall, thresholds = precision_recall_curve(groundtruth, mask)
    return auc(recall, precision)

'''
Function: Return entire datasets list of either saliency metric (i.e mean recall) or manipulation metric (i.e AUC ROC)
Input List of masks, folder to saliency masks, evaluation function to be called
'''
def AUC_all(masks, folder, manip_or_sal_func):
    any_AUC = []
    for images in tqdm(masks):
        img_name = images[0] #get image name
        sal_img = load_image(folder, img_name) #load saliency mask

        #binarize groundtruth from 0-1
        ret, thresh1 = cv.threshold(images[1],1,255,cv.THRESH_BINARY) 
        thresh1[thresh1 > 1] = 1 #set all manipulated pixels of mask to 1

        #check  images are the same shape
        if(np.shape(images[1]) == np.shape(sal_img[1])):
            #run evaluation runction
            AUC = manip_or_sal_func(thresh1.flatten(), (sal_img[1]/255.0).flatten())
            any_AUC.append([img_name, AUC])
        else:
            #Do reshaping if nessary (usually very small re-shaping)
            print('Reshape Required')
            new_img = cv.resize(thresh1, np.shape(sal_img[1])[:2], interpolation= cv.INTER_LINEAR)
            AUC = manip_or_sal_func(new_img.flatten(), (sal_img[1]/255.0).flatten())
            any_AUC.append([img_name, AUC])
    return any_AUC


# def AUC_COCO(masks, folder, manip_or_sal_func,part_1,part_2):
#     any_AUC = []
#     for images in tqdm(masks):
#         img_name = images[0]
#         sal_img = load_image_coco(folder, img_name,part_1,part_2)

#         if(np.shape(images[1]) == np.shape(sal_img[1])):
#             AUC = manip_or_sal_func(images[1].flatten(), (sal_img[1]/255.0).flatten())
#             any_AUC.append([sal_img[0], AUC])
#         else:
#             new_img = cv.resize(images[1], np.shape(sal_img[1])[:2], interpolation= cv.INTER_LINEAR)
#             AUC = manip_or_sal_func(new_img.flatten(), (sal_img[1]/255.0).flatten())
#             any_AUC.append([sal_img[0], AUC])
#     return any_AUC


'''
Function: Get the percentage of an image that was manipulated
Input: [nx[hxw]] list of binary groundtruth masks
'''
def get_manip_size(masks):
    all_percentage = []
    for images in tqdm(masks):
        manip_count = 0.0
        img_name = images[0]
        height, width = images[1].shape[:2]

        ret, thresh1 = cv.threshold(images[1],1,255,cv.THRESH_BINARY)
        thresh1[thresh1 > 1] = 1

        thresh_flatten = thresh1.flatten()
        for i in range(len(thresh_flatten)):
            if (thresh_flatten[i] == 1):
                manip_count +=1
        manip_percent = manip_count/(height*width)
        all_percentage.append([img_name, manip_percent])
    return all_percentage


#KManipulation and saliency metrics
sal_metric = mean_recall
sal_metric_name = "mean_recall"

manip_metric = roc_auc_score
manip_metric_name = "ROC"

#datasets to be run on 
dataset = ['korus', 'korus_resize', 'korus_SE', 'MFC18_formatted_single_small', 'MFC18_resize', 'MFC18_SE', 'IMD2020_formatted', 'IMD2020_resize', 'IMD2020_SE']
for data in dataset:
    #this takes a while
    print(f'dataset: {data}')

    #Decide output names
    output_name = data
    if('IMD2020_formatted' in data):
        output_name = 'imd2020'
        print(output_name)

    if('IMD2020_SE' in data):
        output_name = 'imd2020_SE'
        print(output_name)
    
    if('IMD2020_resize' in data):
        output_name = 'imd2020_resize'
        print(output_name)

    elif('MFC18_formatted_single_small' in data):
        output_name = 'MFC18_small'
        print(output_name)

    #load mask 
    mask_list = data
    if(data == 'korus_size_up'):
        mask_list = 'korus_SE'
    if(data == 'IMD2020_resize'):
        mask_list = 'IMD2020_SE'
    if(data == 'MFC18_resize'):
        mask_list = 'MFC18_SE'
        
    mask = load_images_from_path(f'{mask_list}/masks/')

    '''
    These lines below can be done to find the size of each manipulation in the dataset 
    '''
    # manip_size = get_manip_size(mask)
    # np.save(f'saliency_analysis/manip_size/GT_{output_name}', manip_size)

    #analysis networks -- if you want to save time you can remove (u2net and r3net)
        #Note this step does not do inferencing just calculates the mean recall and area under ROC
    sal_networks = [ 'u2net', 'r3net', 'combine_maps']
    manip_networks = ['pscc', 'mantranet', 'osn', 'busternet']

    # get saliency scores
    for sal_network in sal_networks:
        print(f'\t saliency network: {sal_network} with {sal_metric_name}')
        sal_score = AUC_all(mask, f'inference_outputs/{data}/{sal_network}/', sal_metric)
        np.save(f'analysis_scripts/scores/{sal_network}_{sal_metric_name}_{output_name}', sal_score)

    # get human study scores
    if(data == 'korus' or data == 'korus_SE'):
        print(f'\t saliency network: Human saliency with {sal_metric_name}')
        human_sal = AUC_all(mask,f'inference_outputs/{data}/Human_study/saliency_prediction/', sal_metric)
        np.save(f'analysis_scripts/scores/human_study_sal_{sal_metric_name}_{output_name}', human_sal)

        print(f'\t manipulation network: human manipulation with {manip_metric_name}')
        human_manip = AUC_all(mask, f'inference_outputs/{data}/Human_study/manipulated_prediction/', manip_metric)
        np.save(f'analysis_scripts/scores/human_study_manip_{manip_metric_name}_{output_name}', human_manip)

    #get manpulation scores
    for manip_network in manip_networks:
        print(f'\t manipulation network: {manip_network} with {manip_metric_name}')
        manip_score = AUC_all(mask, f'Manipulation_masks/{data}/{manip_network}_predict_masks/', manip_metric)
        np.save(f'analysis_scripts/scores/{manip_network}_{manip_metric_name}_{output_name}', manip_score)