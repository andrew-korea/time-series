# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:17:24 2021

@author: emily
 
Remark:

"""
import numpy as np
import pandas as pd

def rolling_min_max(file):
    ### Determine File Split Number
    splitnumber = file.replace(file[len(file)-4:len(file)],"")
    splitnumber = splitnumber.replace(file[0:16],"")
    splitnumber = int(splitnumber)
    
    
    file = "../data-sets/KDD-Cup/data/" + file
    
    flag = 0
    with open(file , "r") as filecheck:
        first_line = filecheck.readline()
        number_of_spaces = first_line.count("  ")
        if number_of_spaces == 1:
            # Separator is "\n"
            flag = 1
        else:
            # Separator is " "
            flag = 2    
    if flag == 1:
        data = pd.read_csv(file , sep="\n", header=None)
    if flag == 2:
        data = pd.read_csv(file , sep="  ", header=None, engine='python')
        extract_series = data.loc[0,:] # Extract the data into a Series first
        data = pd.DataFrame(extract_series) # # Convert into a DataFrame again

    
    ### Extract Data
    # Turn Dataframe to Series
    data_training = data.loc[0:(splitnumber-1),0]
    data_test = data.loc[splitnumber: len(data),0]
    
    ### Turn Dataframe to List
    # for the y values
    data_training_y = data_training.tolist()
    data_test_y = data_test.tolist()
    # for the x values
    data_training_x = data_training.index.values.tolist()
    data_test_x = data_test.index.values.tolist()
    
    ######## Training Stage ######## 
    ### Training Data to find Max and Min
    # Set the Window Length
    window_length = 100 
    

    data_range_max = [] # Record the list of Max and Min for each Window
    data_range_min = [] # Record the list of Max and Min for each Window
    data_raw_index = [] # Record the raw index
    data_window_index = [] # Record the window index (Actual x-axis)
    
    data_range_mean = [] # Record the list of Mean for each Window
    
    start_index = 0 # Initiate the index to 0 
    end_index = len(data_training_y)-1 # The Last Index Value
    for row in range(0, round(len(data_training_y)/window_length)):
        ### End the For Loop when the Next Window exceeds the Last Index 
        # - Otherwise proceed to determine the Max, Min and the Mean for each Window
        next_index = start_index + window_length
        if next_index > end_index:
            break
        else:
            ### Determine the Max, Min and the Mean for each Window
            range_index = (start_index, next_index) # raw index range within Window
            current_window_y =  data_training_y[range_index[0]: range_index[1]] # Y value within Window
            current_window_x =  data_training_x[range_index[0]: range_index[1]] # X value within WIndow
            window_index = (current_window_x[0], current_window_x[-1]) # actual dataset index within Window (Actual x-axis)
    
            # Determine the Max and Min Peak in the Window
            in_window_max = np.max(current_window_y) # Max peak in the window
            in_window_min = np.min(current_window_y) # Min peak in the window
     
            # Determine the Mean in the Window 
            in_window_mean = np.mean(current_window_y) # Mean in the window
            data_range_mean.append(in_window_mean)
            
            # Store the Max Peak, Min Peak and Mean determined for each Window into a list
            data_range_max.append(in_window_max) # Store
            data_range_min.append(in_window_min) # Store
            data_raw_index.append(range_index) # Store
            data_window_index.append(window_index) # Store
            
            start_index = next_index # Next Window Range
    
    
    ### Store the Max Peak, Min Peak and Mean for every Window Range into a DataFrame
    data_all = {"raw_index":data_raw_index,
                "window_index":data_window_index,
                "window's max":data_range_max,
                "window's min": data_range_min}
    train_data_df = pd.DataFrame(data_all)
    
    
    ### Determine the Mean and Standard Deviation of the Max Peak and Min Peak from the Training Data
    # - Max Peak Mean and Standard Deviation 
    window_max_std = np.std(data_range_max)
    window_max_mean = np.mean(data_range_max)
    # - Min Peak Mean and Standard Deviation 
    window_min_std = np.std(data_range_min)
    window_min_mean = np.mean(data_range_min)

    ### Determine the Mean of the Mean found from each Window Range 
    window_mean_mean = np.mean(data_range_mean)
    
    
    
    ######## Testing Stage ######## 
    # "reiterate_count" variable is to track the iteration for adjusting the number of Standard Deviation parameter
    # - The Number of Standard Deviation away from the Mean is adjusted for every iteration
    # - When an Anomaly is found, end the While Loop
    # - After adjusting the Number of Standard Deviation, if no Anomaly is found, end the While Loop
    reiterate_count = 0 
    anomaly_search_flag = True
    # The Number of Standard Deviation from the Mean of the Max Peak is initially set to 4
    std_factor_max = 4 
    # The Number of Standard Deviation from the Mean of the Min Peak is initially set to 4
    std_factor_min = 4  

    while anomaly_search_flag:
        ################
        data_range_max_test = [] # Store the Max Peak found in each Window Range from the Test Dataset
        data_range_min_test = [] # Store the Min Peak found in each Window Range from the Test Dataset
        data_raw_index_test = [] # Record the raw index
        data_window_index_test = [] # Record the window index (Actual x-axis) from the Test Dataset
        data_window_flag = [] # Record if any Anomaly was found in this Window Range
        data_anomaly_type = [] # Record the Anomaly Type found in this Window Range
        
        ### Record the upper limit and lower limit for the Max and Min Peak for each Window Range
        # - This is determine using the Standard Deviation and the Mean (from the Training Data) for the Max Peak
        max_upper_list = []
        max_lower_list = []
        # - This is determine using the Standard Deviation and the Mean (from the Training Data) for the Min Peak
        min_upper_list = []
        min_lower_list = []
        
        ### Start to Determine the Anomaly Window by Window
        start_index_test = 0 
        end_index_test = len(data_test_x) 
        record_anomaly_range = []   
        for row in range(0, round(len(data_test_y)/window_length)):
            
            next_index_test = start_index_test + window_length

            ### End the For Loop when the Next Window exceeds the Last Index 
            if next_index_test > end_index_test: # Window Period exceeded Dataset TimeFrame
                break
            else:
                ### Determine the Max Peak and Min Peak for each Window from the Testing Dataset
                range_index_test = (start_index_test, next_index_test) # index of window 
                current_window_y_test =  data_test_y[range_index_test[0]: range_index_test[1]] # Y value within Window
                current_window_x_test =  data_test_x[range_index_test[0]: range_index_test[1]] # X value within Window
                window_index_test = (current_window_x_test[0], current_window_x_test[-1]) # actual dataset index within Window

                # Determine the Max and Min Peak in the Window
                in_window_max = np.max(current_window_y_test) # Max peak in the window
                in_window_min = np.min(current_window_y_test) # Min peak in the window
                
                # Store the Max Peak and Min Peak determined for each Window into a list
                data_range_max_test.append(in_window_max) # Store
                data_range_min_test.append(in_window_min) # Store
                data_raw_index_test.append(range_index_test) # Store
                data_window_index_test.append(window_index_test) # Store
                
                start_index_test = next_index_test # Next Window Range
                
                
                ### Compare the Max Peak and Min Peak found in the Testing Dataset Window with the Standard Deviation and Mean from the Training Dataset
                # For the Max Peak
                # - if the Max Peak found in the window exceeds the Max Upper Limit, it is considered a possible Anomaly
                # - Max Upper Limit calculated by "window_max_mean + window_max_std*std_factor_max "
                # For the Min Peak
                # - If the Min Peak found in the window exceeds the Min Lower Limit, it is considered a possible Anomaly 
                # - Min Lower Limit calculated by "window_min_mean - window_min_std*std_factor_min "
                # Conclusion, When the Peaks found deviates beyond the Standard Deviation from the Mean, it is considered an Anomaly
                max_upper = window_max_mean + window_max_std*std_factor_max
                max_lower = window_max_mean - window_max_std*std_factor_max
                
                min_upper = window_min_mean + window_min_std*std_factor_min
                min_lower = window_min_mean - window_min_std*std_factor_min
                
                # Store the Upper Limit and Lower Limit for Max and Min Peak
                max_upper_list.append(max_upper)
                max_lower_list.append(max_lower)
                min_upper_list.append(min_upper)
                min_lower_list.append(min_lower)
                
                ### Record when an Anomaly was found in a Window Range 
                # If Min Peak exceeds the Min Lower Limit
                if in_window_min < min_lower: 
                    record_anomaly_range.append(window_index_test) # Record the Window Range index where the Anomaly is in
                    data_window_flag.append("Anomaly")
                    data_anomaly_type.append('Min')
                # If Max Peak exceeds the Max Upper Limit
                elif in_window_max > max_upper:
                    record_anomaly_range.append(window_index_test)  # Record the Window Range index where the Anomaly is in
                    data_window_flag.append("Anomaly")
                    data_anomaly_type.append('Max')
                # Otherwise no anomaly was found, proceed
                else:
                    data_window_flag.append("Normal")
                    data_anomaly_type.append('None')
        
        ### When a Final Anomaly was found end the While Loop 
        # - When an Anomaly was found, end the while loop
        # - Otherwise the Number of Standard Deviation parameter (std_factor_max and std_factor_min) is adjusted
        if len(record_anomaly_range) == 1:
            anomaly_search_flag = False
        # Number of Standard Deviation Parameter (std_factor_max and std_factor_min) is adjusted for the next iteration
        else:
            ### Number of Standard Deviation Parameter for the Max Upper Limit is adjusted by 0.5 for every iteration
            # The Number of Standard Deviation Parameter for the Min Lower Limit is Constant
            if reiterate_count < 4: # The Max Upper Limit is adjusted 4 times only 
                std_factor_max -= 0.5
                std_factor_min = 5
                reiterate_count += 1
                anomaly_search_flag = True
            ### Number of Standard Deviation Parameter for the Min Lower Limit is adjusted by 0.5 for every iteration
            # The Number of Standard Deviation Parameter for the Max Upper Limit is Constant
            if 4 <= reiterate_count < 8: # The Min Lower Limit is adjusted 4 times only 
                std_factor_max = 5
                std_factor_min -= 0.5
                reiterate_count += 1
                anomaly_search_flag = True
            ### After Adjusting the Max Upper Limit and Min Lower Limit 4 times, the While Loop will be ended
            # - regardless if an Anomaly was found or not.
            if reiterate_count == 8:
                if len(record_anomaly_range) == 1:
                    record_anomaly_range = record_anomaly_range # return the Window index where the Anomaly is in
                    anomaly_search_flag = False
                else:
                    record_anomaly_range = []
                    anomaly_search_flag = False 
        ################
        # print(record_anomaly_range)
    
    ### Record all the Data that was found into a DataFrame
    data_all = {"raw_index":data_raw_index_test,
                "window_index":data_window_index_test,
                "Max Lower":max_lower_list,
                "Window's max":data_range_max_test,
                "Max Upper":max_upper_list,
                "Min Lower":min_lower_list, 
                "Window's min": data_range_min_test,
                "Min Upper": min_upper_list,
                "Flag":data_window_flag,
                "Type": data_anomaly_type}
    test_data_df = pd.DataFrame(data_all)

    
    ### Print the Anomaly Information
    # - Confidence Score are determined using the following formula
    # - The Difference of the Height of the Anomaly Peak (Max or Min) against the Mean of the Peak (Max or Min) from the Training Dataset ... 
    # - .. is normalized by the Difference of the Height of the Anomaly Peak (Max or Min) against the Mean of the Mean from the Training Dataset
    if len(record_anomaly_range) !=0:
        ## Determine the Window Index where the Anomaly was found
        index_on_dataframe = test_data_df.index[test_data_df['window_index'] == record_anomaly_range[0]].tolist()
        ## Extract all the Anomaly Information into a Series
        results_anamoly = test_data_df.iloc[index_on_dataframe,:].squeeze()
        # Extract the Anomaly Type
        type_anomaly = results_anamoly['Type']
        # Extract the Middle Point of that Anomaly
        middle_point_range = round((results_anamoly["window_index"][0] + results_anamoly["window_index"][1])/2)
        ### Calculate the Confidence Score Based on the Type of Anomaly (Max or Min)
        if type_anomaly == 'Max':
            max_anomaly = results_anamoly["Window's max"]
            max_upper = results_anamoly["Max Upper"]
            score = abs(1 - ((abs(window_max_mean - window_mean_mean))/(abs(max_anomaly - window_mean_mean))))
        if type_anomaly == 'Min':
            min_anomaly = results_anamoly["Window's min"]
            min_lower = results_anamoly["Min Lower"]
            score = abs(1 - ((abs(window_min_mean - window_mean_mean))/(abs(min_anomaly - window_mean_mean))))          
    else:
        ### output 0 confidence score
        score = 0
        middle_point_range = 0
        pass
    # return condition
    if score >0.6:
        return score, middle_point_range
    else:
        return 0, 0



    
    
    


