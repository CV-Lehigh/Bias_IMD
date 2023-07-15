# Exploring_bias_in_IMD

# reproducing the results
## 1. Set up and downloading data --
1. create a conda environment
``` conda env create -f environment.yml ```
2. Download the saliency maps and manipulation predictions from the Gdrive link and place in the 'Exploring_bias_in_IMD/' folder
&emsp; GDrive link:
3. Then unzip the folder ``` unzip WIFS_ready.zip ```
   
File structure should look a little like: 

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
     ├ split_datasets.p
     └ visualisation.py  
 ├ [DIR] WIFS_ready
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

(you should be able to see a histogram for each of the datasets and how many images fall into each saliency group in WIFS_ready/sal_groups/histogram)

## 3. Performance Evaluation
Now based on the saliency groups you can get the average area under ROC for several IMD networks
1. Navigate to the 'Exploring_bias_in_IMD/' folder
2. Run ```python analysis_scipts/visualization.py ```

(you should be able to see a bar graph for each of the datasets and each model performed on each saliency group in WIFS_ready/performance/)

# Supplimental Scripts
Manipulation Size Box and Whisker Plot -- Will display a boxz and whisker plot of the size of manipulation in each saliency group

&emsp; 1. Navigate to the 'Exploring_bias_in_IMD/analysis_scipts/' folder

&emsp; 2. Run ```python find_lengths.py```

&emsp; 3. Place length of partitions of desired dataset in variables g1-5 in boxplot.py file and change name of dataset in all_data ('imd2020', 'MFC18_small', 'korus')

&emsp; 4. Run ```python boxplot.py```
