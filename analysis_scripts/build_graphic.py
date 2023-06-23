import cv2
import numpy as np

def load_numpy(path):
    values = np.load(path, allow_pickle=True)
    return values

def load_numpy_manip(path):
    values = np.load(path, allow_pickle=True)
    formatted_values = []
    scores = []
    for item in values:
        formatted_values.append([item[0], float(item[1])])
        scores.append(float(item[1]))
    return formatted_values, scores

def build_result_image(orginal_image, manip_path, groundtruth_path, filename, score, sal_score):
    # read image
    img = cv2.imread(orginal_image)

    # create cyan image
    blue = np.full_like(img,(255,0,0))
    red = np.full_like(img,(0,0,255))
    addition = np.full_like(img,(255,0,255))


    # add cyan to img and save as new image
    blend = 0.5
    img_blue = cv2.addWeighted(img, blend, blue, 1-blend, 0)
    img_red = cv2.addWeighted(img, blend, red, 1-blend, 0)
    img_add = cv2.addWeighted(img, blend, addition, 1-blend, 0)

    # create black image for mask base
    mask = np.zeros_like(img)
    v_mask = np.zeros_like(img)
    comb_mask = np.zeros_like(img)
    # define rectangle for region where want image colorized with cyan
    manip_values = cv2.imread(manip_path,0)
    GT_values = cv2.imread(groundtruth_path,0)


    ret, manip_thresh = cv2.threshold(manip_values,1,255,cv2.THRESH_BINARY)
    ret1, GT_thresh = cv2.threshold(GT_values, 1, 255, cv2.THRESH_BINARY) 
    comb_thresh = cv2.bitwise_and(manip_thresh, GT_thresh)

    mask[manip_thresh > 1] = 255
    v_mask[GT_thresh > 1] = 255
    comb_mask[comb_thresh > 1] = 255

    # combine img and img_cyan using mask
    result = np.where(mask==255, img_red, img)
    result = np.where(v_mask==255, img_blue, result)
    result = np.where(comb_mask==255, img_add, result)
    result = cv2.putText(result, 'Human manipulation mask', (25, 30), cv2.FONT_HERSHEY_PLAIN, 2.5, (255,0,0), 2, cv2.LINE_AA)
    result = cv2.putText(result, 'Human saliency mask', (25, 70), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,255), 2, cv2.LINE_AA)

    # save results
    cv2.imwrite(f'graphics/manip_size/{filename}_human_mean_recall_saliency_map_{score:.4f}_{sal_score:.4f}.jpg', result)

def find_ten_manip_pred(list_of_images, scores, sal_scores):
    count = 0
    for image_full in list_of_images:
        image = image_full[:len(image_full)-4]
        build_result_image(f'../korus/tampered/{image}.TIF', f'../inference_outputs/korus/Human_study/saliency_prediction/{image}.PNG', f'../korus/masks/{image}.PNG', image, scores[count], sal_scores[count])
        count+=1


manip, scores = load_numpy_manip('manip_size/GT_korus.npy')
sal, sal_scores = load_numpy_manip('scores/human_study_sal_mean_recall_korus.npy')

#where all 10/10 best predictions are high saliency -- only 5/10 lowest prediction are high saliency

k = 10
k_smallest = np.argsort(sal_scores)[:]
k_largest = np.argsort(sal_scores)[len(scores)-k:]
list_imgs = []
manip_scores = []
sal_scores = []
for val in k_smallest:
        list_imgs.append(manip[val][0])
        sal_scores.append(sal[val][1])
        manip_scores.append(manip[val][1]) 

find_ten_manip_pred(list_imgs, manip_scores, sal_scores)