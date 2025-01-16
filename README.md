# Exploring_bias_in_IMD
Exploring Saliency Bias in Manipulation Detection has been accepted to **ICIP 2024!!**: https://ieeexplore.ieee.org/document/10648063
# reproducing the results
## 1. Set up and downloading data --
1. create a conda environment
``` conda env create -f environment.yml ```
2. Download the saliency maps and manipulation predictions (ICIP_2024.zip) from the Gdrive link and place in the 'Exploring_bias_in_IMD/' folder

&emsp; GDrive link: https://drive.google.com/file/d/1CH2L1HTGmxUGSgqzkUfBARjujdkrpInG/view?usp=drive_link

3. Then unzip the folder ``` unzip ICIP_2024.zip ```
   
File structure should look similar to this: 

### Directory Structure  
<pre>Exploring_bias_in_IMD 
 ┬  
 ├ [DIR] analysis_scripts   
     ┬  
     ├ [DIR] manip_size
     ├ boxplot.py
     ├ combine_maps.py  
     ├ metric_calculations.py 
     ├ find_length.py
     ├ split_datasets.py
     └ visualization.py  
 ├ [DIR] ICIP_2024
     ┬   
     ├ [DIR] manipulation_predictions
     ├ [DIR] manipulation_scores
     ├ [DIR] mask
     ├ [DIR] performance
     ├ [DIR] sal_groups
     ├ [DIR] saliency_predictions
     └ [DIR] saliency_scores  
 ├ README.md
 └ environment.yml
</pre>

### (optional) regenerate Human Study maps
This is not nessary because it is already in the folder 'saliency_predictions/korus/Human_study' & 'saliency_predictions/korus_SE/Human_study' but if you would like to you can regenerate the saliency and manipulation prediction masks directly from the human study data. (raw data can be found in user_data_0.txt and SE_data.txt)
1. Navigate to 'Exploring_bias_in_IMD/saliency_prediction/korus/Human_study/'
2. Run ```python generate_hs_masks.py```
3. Navigate to 'Exploring_bias_in_IMD/saliency_prediction/korus_SE/Human_study/'
4. Run ```python generate_hs_masks.py```

### (optional) regenerate Combine Map
This is not nessary because it is already in the file but if you would like to you can regenerate the combination for the u2net and r3net (Note: you will have to change 2 lines of code when constructing MFC18_resize and imd2020_resize)
1. Navigate to 'Exploring_bias_in_IMD/'
2. Run ```python analysis_scripts/combine_maps.py```

## 2. Generating scores
Firstly, we have to get the mean recall and ROC AUC scores for each image
1. Navigate to the 'Exploring_bias_in_IMD/' folder
2. Run ```python analysis_scipts/metric_calculations.py```

## 3. Split dataset
Now based on the mean recall of the combine maps we need to split the dataset into 5 saliency groups
1. Navigate to the 'Exploring_bias_in_IMD/' folder
2. Run ```python analysis_scipts/split_datasets.py ```

(you should be able to see a histogram for each of the datasets and how many images fall into each saliency group in ICIP_2024/sal_groups/histogram)

## 3. Performance Evaluation
Now based on the saliency groups you can get the average area under ROC for several IMD networks
1. Navigate to the 'Exploring_bias_in_IMD/' folder
2. Run ```python analysis_scipts/visualization.py ```

(you should be able to see a bar graph for each of the datasets and each model performed on each saliency group in ICIP_2024/performance/)

# Supplimental Scripts
Manipulation Size Box and Whisker Plot -- Will display a box and whisker plot of the size of manipulation in each saliency group

&emsp; Run ```python analysis_scipts/boxplot.py```

Raw human study data & the code to generate manipulation and prediction masks is available here: 

&emsp; GDrive link: https://drive.google.com/drive/folders/1RE7mUL6I0XNu3mzz3-wWrQ3v47-vdQGA?usp=sharing
