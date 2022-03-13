#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from var_and_z_score.zScore import z_outlier
from var_and_z_score.variance import variance_detection
from var_and_z_score.confidence import get_confidence

import numpy as np

def variance_and_z(file):
    cand_sol = []
    
    split = int(file.split("_")[-1].split(".")[0])
    data_list = np.loadtxt('../data-sets/KDD-Cup/data/' +file)
    
    
    #print("+++++++++Z Score+++++++++")
    outlier_z,confidence_z = z_outlier(data_list, split)
    #print("+++++++++Variance Score+++++++++")
    outlier_v, confidence_v = variance_detection(data_list,split)
    
    #z-score is conservative. less weight is put on the z-score.
    confidence_z = confidence_z*0.7
    
    cand_sol.append((confidence_z,outlier_z))
    cand_sol.append((confidence_v,outlier_v))

    outlier = max(cand_sol)[1]
    confidence = max(cand_sol)[0]

    
    return confidence, outlier

