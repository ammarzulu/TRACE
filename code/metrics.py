import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from training import threshold_anomalies
from sklearn.metrics import roc_curve, auc 

# Anomaly labels are generated under the following assumptions
# If an event has been manually labeled, there is no delay in reporting.
# If a crash has been reported, there is at most a 15 minute delay in reporting, so any prediction 15 minutes before is correct.
# The end of an event cannot be accurately determined. Therefore, data 15 minutes after an event is still anomalous. 
# The next 1:45 cannot be accurately determined to be nominal or anomalous.
# These assumptions allow us to get a grasp of our detection accuracy, but clearly are very conservative and lead to true positive rate being less meaningful.
def generate_anomaly_labels(test_data, kept_indices):
    unix_times = np.unique(test_data['unix_time'])
    test_data = test_data[test_data['unix_time'].isin(unix_times[kept_indices])]
    human_label_times = np.unique(test_data[test_data['human_label']==1]['unix_time'])
    for human_label_time in human_label_times:
        test_data.loc[(test_data['unix_time'] - human_label_time <= 1800) & (test_data['unix_time'] - human_label_time >= 0), 'anomaly'] = 1

    crash_label_times = np.unique(test_data[test_data['crash_record']==1]['unix_time'])
    for crash_label_time in crash_label_times:
        test_data.loc[(test_data['unix_time'] - crash_label_time <= 900) & (test_data['unix_time'] - crash_label_time >= -900), 'anomaly'] = 1

    incident_times = np.unique(test_data[(test_data['human_label']==1) | (test_data['crash_record']==1)]['unix_time'])
    for incident_time in incident_times:
        test_data.loc[(test_data['unix_time'] - incident_time <= 6300) & (test_data['unix_time'] - incident_time >= 900), 'anomaly'] = -1
    
    test_data.fillna(0, inplace=True)

    return test_data['anomaly'].to_numpy()

def crash_detection_delay_loc(anomaly_pred, crash_reported, sr=0.5):
    time_anomalies = np.any(anomaly_pred==1, axis=1)
    spac_pred = np.any(anomaly_pred.reshape(-1, 49, 4)==1, axis=2)
    space_anomalies = np.any(crash_reported.reshape(-1, 49, 4)==1, axis=2)
    delay = []
    detects = []
    localizations = []
    
    reported_indices = np.unique(np.where(crash_reported == 1)[0])
    for i in reported_indices:
        detected = False
        for t in range(int(i-(15/sr)), int(i+(15/sr))):
            if t >= len(time_anomalies):
                detected = False
                break
            if time_anomalies[t] == 1:
                delay.append(t-i)
                anom_indices=np.where(space_anomalies[i]==1)[0]
                pred_indices=np.where(spac_pred[t]==1)[0]
                distances = np.abs(anom_indices[:, None] - pred_indices[None, :])

                # Find the mean distance
                
                localizations.append(distances.flatten())
                
                
                
                detected = True
                break
            
        detects.append(detected)
    all_distances = np.concatenate(localizations) if localizations else np.array([])

    
    return delay, detects, all_distances

def discrete_fp_delays(thresh, test_errors, anomaly_labels, crash_reported, sr=0.5):
    thresholds = find_thresholds(thresh, test_errors, anomaly_labels) #thresholds[:,0] is the fpr

    fprs = [1, 2.5, 5, 10, 20]
    new_thresholds = []
    for fpr in fprs:
        new_thresholds.append(find_percent(thresholds, fpr))

    anomaly_instances = []
    for t in new_thresholds:
        anomaly_instances.append(threshold_anomalies(thresh+t, test_errors))

    results=[]
    for i, fpr in enumerate(fprs):
        delay, found = crash_detection_delay(anomaly_instances[i], crash_reported, sr=sr)
        mu = np.mean(delay) * sr
        std = np.std(delay) * sr
        miss_percent = 1-(np.sum(found) / len(found))
        
        results.append([fpr, mu, std, miss_percent])
        print(f'FPR {fpr}% gives mean delay of {mu} +/- {std} while missing {miss_percent}%.')
    results_df = pd.DataFrame(results, columns=['FPR (%)', 'Mean Delay', 'Std Delay', 'Miss Percentage'])
    
    return results_df


def discrete_fp_delays_loc(thresh, test_errors, anomaly_labels, crash_reported, sr=0.5):
    thresholds = find_thresholds(thresh, test_errors, anomaly_labels) #thresholds[:,0] is the fpr

    fprs = [1, 2.5, 5, 10, 20]
    new_thresholds = []
    for fpr in fprs:
        new_thresholds.append(find_percent(thresholds, fpr))

    anomaly_instances = []
    for t in new_thresholds:
        anomaly_instances.append(threshold_anomalies(thresh+t, test_errors))
        
    results=[]
    for i, fpr in enumerate(fprs):
        delay, found, distances = crash_detection_delay_loc(anomaly_instances[i], crash_reported, sr=0.5)
        mu = np.mean(delay) * 0.5
        std = np.std(delay) * 0.5
        mu_dist = np.mean(distances)
        std_dist = np.std(distances)
        miss_percent = 1-(np.sum(found) / len(found))
        
        results.append([fpr, mu, std, miss_percent, mu_dist, std_dist])
        print(f'FPR {fpr}% gives mean delay of {mu} +/- {std} with mean. distance of {mu_dist*0.35} +/- {std_dist*0.35} while missing {miss_percent}%.')
        # break
    results_df = pd.DataFrame(results, columns=['FPR (%)', 'Mean Delay', 'Std Delay', 'Miss Percentage', 'Mean Distance', 'Std Distance'])


    
    return results_df

def discrete_fp_delays_nll(thresh, test_errors, anomaly_labels, crash_reported, sr=0.5):
    thresholds = find_thresholds_nll(thresh, test_errors, anomaly_labels) #thresholds[:,0] is the fpr

    fprs = [1, 2.5, 5, 10, 20]
    new_thresholds = []
    for fpr in fprs:
        new_thresholds.append(find_percent(thresholds, fpr))

    anomaly_instances = []
    for t in new_thresholds:
        anomaly_instances.append(threshold_anomalies(t, test_errors))

    results=[]
    for i, fpr in enumerate(fprs):
        delay, found = crash_detection_delay(anomaly_instances[i], crash_reported, sr=sr)
        mu = np.mean(delay) * sr
        std = np.std(delay) * sr
        miss_percent = 1-(np.sum(found) / len(found))
        results.append([fpr, mu, std, miss_percent])
        print(f'FPR {fpr}% gives mean delay of {mu} +/- {std} while missing {miss_percent}%.')
    
    results_df = pd.DataFrame(results, columns=['FPR (%)', 'Mean Delay', 'Std Delay', 'Miss Percentage'])
    
    return results_df

def discrete_fp_delays_loc_nll(thresh, test_errors, anomaly_labels, crash_reported, sr=0.5):
    thresholds = find_thresholds_nll(thresh, test_errors, anomaly_labels) #thresholds[:,0] is the fpr
    print(thresholds)
    fprs = [1, 2.5, 5, 10, 20]
    new_thresholds = []
    for fpr in fprs:
        new_thresholds.append(find_percent(thresholds, fpr))

    anomaly_instances = []
    for t in new_thresholds:
        anomaly_instances.append(threshold_anomalies(t, test_errors))

    results=[]
    for i, fpr in enumerate(fprs):
        delay, found, distances = crash_detection_delay_loc(anomaly_instances[i], crash_reported, sr=0.5)
        mu = np.mean(delay) * 0.5
        std = np.std(delay) * 0.5
        mu_dist = np.mean(distances)
        std_dist = np.std(distances)
        miss_percent = 1-(np.sum(found) / len(found))
        
        results.append([fpr, mu, std, miss_percent, mu_dist, std_dist])
        print(f'FPR {fpr}% gives mean delay of {mu} +/- {std} with mean. distance of {mu_dist*0.35} +/- {std_dist*0.35} while missing {miss_percent}%.')
        # break
    results_df = pd.DataFrame(results, columns=['FPR (%)', 'Mean Delay', 'Std Delay', 'Miss Percentage', 'Mean Distance', 'Std Distance'])

    
    return results_df

def find_percent(thresholds, percent):
    percent = percent / 100
    thresholds = np.array(thresholds)
    index_closest = thresholds.shape[0] - 1 - np.argmin(np.abs(thresholds[:,0][::-1] - percent))
    print(f"Found FPR of {thresholds[index_closest][0]} for {percent}")
    return thresholds[index_closest][1]




def find_thresholds_nll( thresh,nll_scores, anomaly_labels, changed_sr=False):
    results = []
    changed_sr = False
    
    # Define the threshold range based on NLL scores and whether the sampling rate has changed
    if changed_sr:
        # If the sampling rate has changed, use a broader threshold range
        thresh_range = np.linspace(np.min(nll_scores), np.max(nll_scores), 1000)
    else:
        # Use a narrower range based on percentiles (e.g., 5th to 95th percentile)
        thresh_range = [np.percentile(np.sort(nll_scores.flatten()),i) for i in np.arange(0, 100, 0.01)]
    
    # Loop over the threshold range and calculate TPR/FPR for each threshold
    for i in tqdm(thresh_range):
        # Apply threshold to NLL scores
        anomaly_pred = threshold_anomalies(i, nll_scores)
        
        # Calculate true positive rate (TPR) and false positive rate (FPR)
        tpr, fpr = calculate_tp_fp(anomaly_pred, anomaly_labels)
        # Store the false positive rate (FPR) and the corresponding threshold offset
        results.append([fpr, i])
        
    return results

def find_delays_nll(thresh, nll_scores, anomaly_labels, crash_reported, changed_sr=False, sr=0.5):
    results = []
    # Find thresholds based on NLL scores
    thresholds = np.array(find_thresholds_nll(thresh, nll_scores, anomaly_labels, changed_sr))
   

    
    # Find indices where false positive rate (FPR) is 1 and 0
    all_fp_indices = np.where(thresholds[:, 0] == 1)[0]
    all_fp_index = all_fp_indices[-1] if len(all_fp_indices) > 0 else None

    no_fp_indices = np.where(thresholds[:, 0] == 0)[0]
    no_fp_index = no_fp_indices[-1] if len(no_fp_indices) > 0 else None
    nll_percentiles = np.percentile(nll_scores, np.linspace(0, 100, 99))
    

    if no_fp_index is None:
        # Handle case when no false positives are found
        print("No threshold found with zero false positives.")
        no_fp_index= np.argmin(thresholds[:, 0])
        # no_fp_indices = 
        # raise ValueError() # You can adjust this logic to fit your use case
        
    if all_fp_index is None:
        # Handle case when all false positives are found
        print("No threshold found with all false positives.")
        all_fp_index = np.argmax(thresholds[:, 0])
        # raise ValueError() # You can adjust this logic to fit your use case
    
    # Apply the threshold corresponding to no false positives and compute delays
    anomaly_pred = threshold_anomalies(thresholds[no_fp_index, 1], nll_scores)
    delays, detects = crash_detection_delay(anomaly_pred, crash_reported, sr=sr) 
    results.append([0, np.mean(delays), np.std(delays), np.sum(detects) / 12])
    
    # Loop through different thresholds based on val_range
    for i in tqdm(nll_percentiles):
        # Find the closest threshold to the desired false positive rate
        offset_index = np.argmin(np.abs(thresholds[:, 0] - i))
        offset = thresholds[offset_index, 1]
        # Apply the threshold to NLL scores
        anomaly_pred = threshold_anomalies(offset, nll_scores)
        delays, detects = crash_detection_delay(anomaly_pred, crash_reported, sr=sr)
        
        if np.sum(detects) == 0:
            delays = [15 / sr]
        
        results.append([thresholds[offset_index, 0], np.mean(delays), np.std(delays), np.sum(detects) / 12])
        
    # Apply the threshold corresponding to all false positives and compute delays
    anomaly_pred = threshold_anomalies( thresholds[all_fp_index, 1], nll_scores)
    delays, detects = crash_detection_delay(anomaly_pred, crash_reported, sr=sr)
    results.append([1, np.mean(delays), np.std(delays), np.sum(detects) / 12])

    return results,thresholds


def find_delays(thresh, errors, anomaly_labels, crash_reported, changed_sr=False, sr=0.5):
    results = []
    thresholds = np.array(find_thresholds(thresh, errors, anomaly_labels, changed_sr))
    all_fp_indices = np.where(thresholds[:,0] == 1)[0]
    all_fp_index = all_fp_indices[-1] if len(all_fp_indices) > 0 else None

    no_fp_indices = np.where(thresholds[:,0] == 0)[0]
    no_fp_index = no_fp_indices[-1] if len(no_fp_indices) > 0 else None
    val_range = np.linspace(0.01, 0.99, 98)

    if no_fp_index is None:
        no_fp_index = np.argmin(thresholds[:,0])
        # do something here
    if all_fp_index is None:
        all_fp_index = np.argmax(thresholds[:,0])
        # do something here
    
    
    anomaly_pred = threshold_anomalies(thresh+thresholds[no_fp_index,1], errors)
    delays, detects = crash_detection_delay(anomaly_pred, crash_reported, sr=sr) 
    results.append([0, np.mean(delays), np.std(delays), np.sum(detects)/12])
    


    for i in tqdm(val_range):
        # offset_index = np.abs(thresholds[:,0] - i).argmin()
        offset_index = thresholds.shape[0] - 1 - np.argmin(np.abs(thresholds[:,0][::-1] - i))
        offset = thresholds[offset_index,1]
        anomaly_pred = threshold_anomalies(thresh+offset, errors)
        delays, detects = crash_detection_delay(anomaly_pred, crash_reported, sr=sr) 
        if np.sum(detects) == 0:
            delays = [15/sr]
        results.append([thresholds[offset_index,0], np.mean(delays), np.std(delays), np.sum(detects)/12])
        
    anomaly_pred = threshold_anomalies(thresh+thresholds[all_fp_index,1], errors)
    delays, detects = crash_detection_delay(anomaly_pred, crash_reported, sr=sr) 
    results.append([1, np.mean(delays), np.std(delays), np.sum(detects)/12])

    return results

def calculate_tp_fp(anomaly_pred, anomaly_labels):
    tps = 0
    fps = 0
    tns = 0
    num_anom = 0
    anomaly_pred = anomaly_pred.flatten()
    num_nodes = 196
    for i in range(0, len(anomaly_labels), num_nodes):
        predictions = anomaly_pred[i:i+num_nodes]
        predicted_anomaly = np.any(predictions==1)
        
        true_vals = anomaly_labels[i:i+num_nodes]
        actually_anomaly = np.all(true_vals==1)
        if np.any(true_vals==-1):
            continue 
        if predicted_anomaly and actually_anomaly:
            tps += 1
        elif predicted_anomaly and not actually_anomaly:
            fps += 1
            
        if actually_anomaly:
            num_anom += 1

        if not actually_anomaly and not predicted_anomaly:
            tns += 1
    
    tpr = tps / num_anom
    fpr = fps / (fps + tns)
    return tpr, fpr

def calculate_tp_fp_num(anomaly_pred, anomaly_labels):
    tps = 0
    fps = 0
    tns = 0
    num_anom = 0
    anomaly_pred = anomaly_pred.flatten()
    num_nodes = 196
    for i in range(0, len(anomaly_labels), num_nodes):
        predictions = anomaly_pred[i:i+num_nodes]
        predicted_anomaly = np.any(predictions==1)
        
        true_vals = anomaly_labels[i:i+num_nodes]
        actually_anomaly = np.all(true_vals==1)
        if np.any(true_vals==-1):
            continue 
        if predicted_anomaly and actually_anomaly:
            tps += 1
        elif predicted_anomaly and not actually_anomaly:
            fps += 1
            
        if actually_anomaly:
            num_anom += 1

        if not actually_anomaly and not predicted_anomaly:
            tns += 1
    

    return tps, fps, tns,num_anom

def find_thresholds(thresh, errors, anomaly_labels, changed_sr=False):
    results = []
    # When changing the sampling rate, the range of thresholds needs to be larger. Keep changed_sr False to reproduce results of main paper
    if changed_sr:
        thresh_range = np.linspace(-0.5, 1.5, 1000)
    else:
        thresh_range = np.linspace(-0.1, 0.2, 1000)
    for i in tqdm(thresh_range):
        anomaly_pred = threshold_anomalies(thresh+i, errors)
        tpr, fpr = calculate_tp_fp(anomaly_pred, anomaly_labels)
        results.append([fpr, i])
        
    return results

def crash_detection_delay(anomaly_pred, crash_reported, sr=0.5):
    time_anomalies = np.any(anomaly_pred==1, axis=1)
    delay = []
    detects = []
    
    reported_indices = np.where(crash_reported == 1)[0]
    for i in reported_indices:
        detected = False
        for t in range(int(i-(15/sr)), int(i+(15/sr))):
            if t >= len(time_anomalies):
                detected = False
                break
            if time_anomalies[t] == 1:
                delay.append(t-i)
                detected = True
                break
            
        detects.append(detected)
    
    return delay, detects

def calculate_accuracy(anomaly_pred, anomaly_labels):
    correct = []
    anomaly_pred = anomaly_pred.flatten()
    num_nodes = 196
    for i in range(0, len(anomaly_labels), num_nodes):
        predictions = anomaly_pred[i:i+num_nodes]
        predicted_anomaly = np.any(predictions==1)
        
        true_vals = anomaly_labels[i:i+num_nodes]
        actually_anomaly = np.any(true_vals==1)
        if np.any(true_vals==-1):
            continue 
        if predicted_anomaly == actually_anomaly:
            correct.append(1)
        else:
            correct.append(0)
    
    correct = np.array(correct)
    return np.sum(correct) / len(correct)

def n_in_a_row(anomalies, n):
    result = np.zeros_like(anomalies)
    for j in range(anomalies.shape[1]):
        counter = 0
        for i in range(anomalies.shape[0]):
            if anomalies[i,j] == 1:
                counter += 1
            else:
                counter = 0
            
            if counter >= n:
                result[i,j] = 1
                
    return result

def calculate_auc(test_errors, anomaly_labels):
    def anomaly_score(errors):
        return np.max(errors, axis=1)
    
    def remove_unknowable(score, anomaly_labels):
        time_anomalies = anomaly_labels.reshape(-1,196)[:,0] # they are all the same
        known = time_anomalies != -1
        return score[known], time_anomalies[known]
    
    score = anomaly_score(test_errors)
    score, time_labels = remove_unknowable(score, anomaly_labels)
    fpr, tpr, thresholds = roc_curve(time_labels, score)
    roc_auc = auc(fpr, tpr)
    return roc_auc

from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

def calculate_metrics(test_errors, anomaly_labels, thresholds):
    """
    Calculate Precision, Recall, and F1-score using predefined thresholds for each node.

    :param test_errors: Error values in shape (B, N) where B is batch size and N is the number of nodes
    :param anomaly_labels: Anomaly labels in shape (B*N,)
    :param thresholds: Thresholds for each node in shape (N,)
    
    :return: Precision, Recall, F1-score, and thresholds
    """
    def anomaly_score(errors, thresholds):
        # Apply thresholds per node to obtain anomaly scores
        return (errors > thresholds).astype(int)
    
    def remove_unknowable(score, anomaly_labels):
        # Assuming the anomaly_labels has -1 for unknowable samples, remove them
        time_anomalies = anomaly_labels.reshape(-1, 196)[:, 0]  # Assuming 196 nodes per time step
        known = time_anomalies != -1
        return score[known], time_anomalies[known]
    
    # Calculate anomaly scores
    predicted_scores = anomaly_score(test_errors, thresholds)
    
    # Flatten predictions and anomaly labels
    predicted_scores = predicted_scores.flatten()
    anomaly_labels = anomaly_labels.flatten()

    # Filter out unknowable samples
    predicted_scores, time_labels = remove_unknowable(predicted_scores, anomaly_labels)
    
    # Calculate Precision, Recall, F1-score
    precision = precision_score(time_labels, predicted_scores, zero_division=1)
    recall = recall_score(time_labels, predicted_scores, zero_division=1)
    f1 = f1_score(time_labels, predicted_scores, zero_division=1)
    
    return precision, recall, f1

def find_delays_nll(thresh, errors, anomaly_labels, crash_reported, changed_sr=False, sr=0.5):
    results = []
    thresholds = np.array(find_thresholds_nll(thresh, errors, anomaly_labels, changed_sr))
    all_fp_indices = np.where(thresholds[:,0] == 1)[0]
    all_fp_index = all_fp_indices[-1] if len(all_fp_indices) > 0 else None

    no_fp_indices = np.where(thresholds[:,0] == 0)[0]
    no_fp_index = no_fp_indices[-1] if len(no_fp_indices) > 0 else None
    val_range = np.linspace(0.01, 0.99, 98)

    if no_fp_index is None:
        no_fp_index = np.argmin(thresholds[:,0])
        # do something here
    if all_fp_index is None:
        all_fp_index = np.argmax(thresholds[:,0])
        # do something here
    
    
    anomaly_pred = threshold_anomalies(thresholds[no_fp_index,1], errors)
    delays, detects = crash_detection_delay(anomaly_pred, crash_reported, sr=sr) 
    results.append([0, np.mean(delays), np.std(delays), np.sum(detects)/12])
    


    for i in tqdm(val_range):
        # offset_index = np.abs(thresholds[:,0] - i).argmin()
        offset_index = thresholds.shape[0] - 1 - np.argmin(np.abs(thresholds[:,0][::-1] - i))
        offset = thresholds[offset_index,1]
        anomaly_pred = threshold_anomalies(thresh+offset, errors)
        delays, detects = crash_detection_delay(anomaly_pred, crash_reported, sr=sr) 
        if np.sum(detects) == 0:
            delays = [15/sr]
        results.append([thresholds[offset_index,0], np.mean(delays), np.std(delays), np.sum(detects)/12])
        
    anomaly_pred = threshold_anomalies(thresholds[all_fp_index,1], errors)
    delays, detects = crash_detection_delay(anomaly_pred, crash_reported, sr=sr) 
    results.append([1, np.mean(delays), np.std(delays), np.sum(detects)/12])

    return results
