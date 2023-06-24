# Exploring_bias_in_IMD

# reproducing the results
## 1. Gathering data --
Gdrive link: https://drive.google.com/drive/folders/1F1fjItFjeBShbu0F1bu5mnMhtYo1Ayf9?usp=sharing
1. Download the saliency predictions from the Gdrive link and place in the 'Exploring_bias_in_IMD/' folder -- ***inference_outputs***
2. Download the Manipulation predictions from the Gdrive link and place in the 'Exploring_bias_in_IMD/' folder -- ***Manipulation_masks***
   
File structure should look a little like: 

Exploring_bias_in_IMD/

&emsp; all other folders/

&emsp; analysis_scripts/

&emsp; inference_outputs/

&emsp; Manipulation_masks/

### (optional) regenerate Combine Map
This is not nessary because it is already in the filebut if you would like to you can regenerate the combination for the u2net and r3net
1. Navigate to 'Exploring_bias_in_IMD/'
2. Run ```python analysis_scripts/combine_maps.py```

## 2. Generating scores
Firstly, we have to get the mean recall and ROC AUC scores for each image
1. Navigate to the 'Exploring_bias_in_IMD/' folder
2. Run ```python analysis_scipts/metric_calculations.py```

## 3. Split dataset
Now based on the mean recall of the combine maps we need to split the dataset into 5 saliency groups
1. Navigate to the 'Exploring_bias_in_IMD/analysis_scipts/' folder
2. Run ```python split_datasets.py ```

(you should be able to see a histogram for each of the datasets and how many images fall into each saliency group)

## 3. Performance Evaluation
Now based on the saliency groups you can get the average area under ROC for several IMD networks
1. Navigate to the 'Exploring_bias_in_IMD/analysis_scipts/' folder
2. Run ```python visualization.py ```

(you should be able to see a bar graph for each of the datasets and each model performed on each saliency group)

## Supplimental Scripts
delta_split.py -- Will tell you how many images moved from one saliency group to another after saliency enhancement
Manipulation Size graphs -- 
&emsp; 1. Navigate to the 'Exploring_bias_in_IMD/analysis_scipts/' folder
&emsp; 2. Run ```python find lengths.py```
&emsp; 3. 

