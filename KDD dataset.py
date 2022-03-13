#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import csv

from matrix_profile.MP import *
from var_and_z_score.variance_and_z_score import *
from rolling_min_max.rolling import rolling_min_max

file_dir = "../data-sets/KDD-Cup/data"
data_files = os.listdir(file_dir)
data_files.sort()
data_files = data_files[1:]


# In[2]:


count = 1

with open('./submission.csv', 'w') as f:
    writer = csv.writer(f)
        
    for file in data_files:
        print("Dataset #", count)
        sol_cand = []

        #matrix profile algorithm
        confidence, outlier = MP(file)
        sol_cand.append((confidence,outlier))

        #variance and z-score algorithm
        confidence, outlier = variance_and_z(file)
        sol_cand.append((confidence,outlier))

        #rolling minimum maximum algorithm
        confidence, outlier = rolling_min_max(file)
        sol_cand.append((confidence,outlier))


        
        #outlier with highest confidence
        final_outlier = max(sol_cand)[1]
        final_confidence = max(sol_cand)[0]

        print("Outlier: ", final_outlier)
        
        #write to output file 
        #include header
        if count == 1:
            header = ["No.", "Location of Anomaly"]
            writer.writerow(header)

        content = [count,final_outlier]
        writer.writerow(content)

        count+=1


# In[ ]:




