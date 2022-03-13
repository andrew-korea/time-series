#a_idx is max index
#b_idx is second max index

def get_confidence(zdata, a_idx, b_idx):
    
    outlier = a_idx
    
    #the difference between the highest and the second highest anomaly scores indicates how confident the scoring algorithm is
    #the more different the two maximum anomaly scores are, the more likely it is an anomaly
    confidence = (zdata[a_idx]-zdata[b_idx])/zdata[a_idx]
    
    
    return outlier, confidence

