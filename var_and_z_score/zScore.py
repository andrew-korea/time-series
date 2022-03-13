import scipy.stats as stats
from var_and_z_score.confidence import get_confidence


def z_outlier(data_list, split):
    

    #we find the two highest z-scores
    #the second highest z-score must be at least 300 indeces apart from the first highest z-score to avoid overlap
    
    zdata = stats.zscore(data_list)
    zdata = abs(zdata)
    zdata = zdata.tolist()
    max1_idx = zdata.index(max(zdata[split:]))
    zdata[max1_idx] = 0
    max2_idx = zdata.index(max(zdata[split:]))
    while max2_idx < max1_idx + 300 and max2_idx > max1_idx - 300:
        zdata[max2_idx] = 0
        max2_idx = zdata.index(max(zdata[split:]))
        
       
    #we reinitialize zdata since it was tampered with above
    zdata = stats.zscore(data_list)
    zdata = abs(zdata)
    
    
    return get_confidence(zdata, max1_idx, max2_idx)




