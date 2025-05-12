import torch
import numpy as np
import matplotlib.pyplot as plt

from models import GATAE, RelationalSTAE, SpatioTemporalAutoencoder, GATSpatioTemporalAutoencoder, GraphAE, TransformerAutoencoder, MLPAutoencoder, GraphTransformerNormalizingFlow, FixedGraphTransformerNormalizingFlow, TRACE_GRU, TRACE_LSTM, TRACE_GCN, TRACE_Transformer
from parameters import GATAEParameters, RSTAEParameters, STAEParameters, TrainingParameters, GATSTAEParameters, GraphAEParameters, TransformerAEParameters, MLPAEParameters, GraphTransformerNormalizingFlowParameters
from datautils import get_morning_data, milemarkers
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import Data as PyGData
import pandas as pd

import random

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, input, target):
        # Calculate squared errors
        errors = (input - target) ** 2

        # Apply weights to squared errors
        weighted_errors = errors * self.weights

        # Calculate the mean over all dimensions
        loss = torch.mean(weighted_errors)

        return loss
    
def save_model(model, name):
    torch.save(model.state_dict(), f'./saved_models/{name}.pth')

def load_model(modelclass, parameters, name):
    
    if modelclass == FixedGraphTransformerNormalizingFlow or modelclass ==TRACE_LSTM or modelclass ==TRACE_GRU or modelclass ==TRACE_GCN  or modelclass ==TRACE_Transformer:
        model = modelclass(**parameters)
    else:
        model = modelclass(parameters)
    checkpoint = torch.load(f'./saved_models/{name}.pth')
    model.load_state_dict(checkpoint)

    return model

def train_stae(staeparams: STAEParameters, trainingparams: TrainingParameters, training_data: PyGData, mse_weights: list = [1,1,1], verbose=False, full_data=True):
    ae = SpatioTemporalAutoencoder(staeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            if full_data:
                graph_sequence = graph_sequence[0]
            optimizer.zero_grad()
            xhat = ae(graph_sequence) # encode and decode the sequence

            loss = weighted_mse(xhat, graph_sequence[-1].x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_rstae(rstaeparams: RSTAEParameters, trainingparams: TrainingParameters, training_data, mse_weights: list = [1,1,1], verbose=False):
    ae = RelationalSTAE(rstaeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for rgraph in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(rgraph[0]) # encode and decode the sequence

            loss = weighted_mse(xhat, rgraph[1].x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_gatae(gataeparams: GATAEParameters, trainingparams: TrainingParameters, training_data, mse_weights: list = [1,1,1], verbose=False):
    ae = GATAE(gataeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for rgraph in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(rgraph[0]) # encode and decode the sequence

            loss = weighted_mse(xhat, rgraph[1].x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_transformerae(params: TransformerAEParameters, trainingparams: TrainingParameters, training_data, mse_weights: list = [1,1,1], verbose=False):
    ae = TransformerAutoencoder(params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for sequence, current in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(sequence) # encode and decode the sequence

            loss = weighted_mse(xhat, current)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_mlpae(params: MLPAEParameters, trainingparams: TrainingParameters, training_data, mse_weights: list = [1,1,1], verbose=False):
    ae = MLPAutoencoder(params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for sequence, current in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(sequence) # encode and decode the sequence

            loss = weighted_mse(xhat, current)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_gcnae(gcnaeparams: GraphAEParameters, trainingparams: TrainingParameters, training_data, mse_weights: list = [1,1,1], verbose=False):
    ae = GraphAE(gcnaeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for rgraph in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(rgraph) # encode and decode the sequence

            loss = weighted_mse(xhat, rgraph.x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_gatstae(staeparams: GATSTAEParameters, trainingparams: TrainingParameters, training_data: PyGData, mse_weights: list = [1,1,1], verbose=False):
    ae = GATSpatioTemporalAutoencoder(staeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(graph_sequence)

            loss = weighted_mse(xhat, graph_sequence[-1].x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

# Computes weighted reconstruction error
def compute_weighted_error(error, weights):
    error *= np.array(weights) 
    error = np.mean(error, axis=1)

    return error

# Determines anomaly threshold
def compute_anomaly_threshold(training_data, model, weights, method='max'):
    model.eval()
    errors = []

    for graph_sequence in tqdm(training_data):
        xhat = model(graph_sequence[1])
        error = (xhat - graph_sequence[-1].x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        errors.append(weighted)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()
    
def compute_anomaly_threshold_rstae(training_data, model, weights, method='max'):
    model.eval()
    errors = []

    for graph_sequence in tqdm(training_data):
        xhat = model(graph_sequence[0])
        error = (xhat - graph_sequence[1].x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        errors.append(weighted)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()

def compute_anomlay_threshold_gtnf(data, model, device, method='mean'):
    model.eval()
    errors = []


    all_node_log_probs = []  # For storing node-level anomaly scores
    all_graph_log_probs = []  # For storing graph-level anomaly scores



    # First pass: Collect node-level and graph-level scores for dynamic thresholding
    with torch.no_grad():
        for batch in data:
            feature_matrices = batch.x.to(device)  # Shape: [B, W, N, F]
            adj_matrices = batch.adj.to(device)    # Shape: [B, W, N, N]
            true_labels = batch.y.to(device)       # Shape: [B, W, N, 1]

            # Extract features for anomaly detection
            anomaly_scores = model(feature_matrices, adj_matrices)  # Shape: [B, W, N]
            anomaly_scores= torch.mean(anomaly_scores,dim=1)

            # Flatten the anomaly scores for node-level thresholding
            anomaly_scores_flat = anomaly_scores.cpu().numpy()  # Shape: [B * N]
            all_node_log_probs.extend(anomaly_scores_flat)

            # take mean the predicted probabilities across all nodes in each graph for graph-level thresholding
            graph_level_scores = anomaly_scores.mean(dim=1)  # Shape: [B]
            all_graph_log_probs.extend(graph_level_scores.view(-1).cpu().numpy())  # Shape: [B]

        # Compute dynamic thresholds based on percentiles
        node_dynamic_threshold = np.mean(all_node_log_probs,axis=0) + 3*np.std(all_node_log_probs,axis=0)
        graph_dynamic_threshold = np.mean(all_graph_log_probs) + 3*np.std(all_graph_log_probs)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(all_node_log_probs, axis=0), np.max(all_graph_log_probs, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return node_dynamic_threshold, graph_dynamic_threshold
    else:
        raise NotImplementedError()
    
    
def compute_anomaly_threshold_transformerae(training_data, model, weights, method='max'):
    model.eval()
    errors = []

    for sequence, current in tqdm(training_data):
        xhat = model(sequence)
        error = (xhat - current) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        errors.append(weighted)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()
    
def compute_anomaly_threshold_mlpae(training_data, model, weights, method='max'):
    model.eval()
    errors = []

    for sequence, current in tqdm(training_data):
        xhat = model(sequence).squeeze()
        error = (xhat - current) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        errors.append(weighted)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()
    
def compute_anomaly_threshold_gcnae(training_data, model, weights, method='max'):
    model.eval()
    errors = []

    for graph_sequence in tqdm(training_data):
        xhat = model(graph_sequence)
        error = (xhat - graph_sequence.x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        errors.append(weighted)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()
    
def train_gtnf(params, trainingparams, training_data, verbose=False):
    model = GraphTransformerNormalizingFlow(params)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=trainingparams.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        model.train()
        for batch_idx, batch in enumerate(training_data):
            feature_matrices = batch.x.to(params.device)   # Shape: [B, W, N, F]
            adj_matrices = batch.adj.to(params.device)     # Shape: [B, W, N, N]
            true_labels = batch.y.to(params.device)        # Shape: [B, W, N, 1]
            B=feature_matrices.size(0)
            W=feature_matrices.size(1)
            N=feature_matrices.size(2)

            # Extract features for anomaly detection
            anomaly_scores = model(feature_matrices, adj_matrices)  # Shape: [B, W, N]
            true_labels=(torch.mean(true_labels.view(B,W,N),dim=1)>0).float()
            anomaly_scores= torch.mean(anomaly_scores,dim=1)
            loss = criterion(anomaly_scores, true_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return model, losses
    
# Determine which nodes are anomalies based on threshold
def threshold_anomalies(thresh, errors):
    anomalies = []
    
    for error_t in errors:
        anomalies.append((error_t > thresh).astype(int))

    return np.array(anomalies)
    
# Evaluate a model on the test set
def test_model(test_sequence, weights, model, verbose=False):
    model.eval()
    errors = []
    recons_speeds = []
    speeds = []

    for graph_sequence in tqdm(test_sequence, disable=not verbose):
        xhat = model(graph_sequence)
        error = (xhat - graph_sequence[-1].x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        recons_speeds.append(xhat.detach().numpy()[:,1])
        speeds.append(graph_sequence[-1].x[:,1].detach().numpy())
        errors.append(weighted)

    errors = np.array(errors)
    recons_speeds = np.array(recons_speeds)
    speeds = np.array(speeds)

    return errors, recons_speeds, speeds

def test_gtnf(test_data, model, device, node_dynamic_threshold, graph_dynamic_threshold, verbose=False):

    all_node_true_labels = []
    all_node_pred_labels = []
    all_node_log_probs = []  # For storing node-level anomaly scores
    
    with torch.no_grad():
                for batch_idx, batch in enumerate(test_data):
                    feature_matrices = batch.x.to(device)   # Shape: [B, W, N, F]
                    adj_matrices = batch.adj.to(device)     # Shape: [B, W, N, N]
                    true_labels = batch.y.to(device)        # Shape: [B, W, N, 1]
                    B=feature_matrices.size(0)
                    W=feature_matrices.size(1)
                    N=feature_matrices.size(2)

                    # Extract features for anomaly detection
                    anomaly_scores = model(feature_matrices, adj_matrices)  # Shape: [B, W, N]
                    true_labels=(torch.mean(true_labels.view(B,W,N),dim=1)>0).float()
                    anomaly_scores= torch.mean(anomaly_scores,dim=1)
                    all_node_log_probs.extend(anomaly_scores.cpu().numpy())

                    

                    # ---- Node-Level Anomaly Detection ----
                    # Flatten node-level true labels and predicted scores
                    true_labels_flat = true_labels.cpu().numpy()  # Shape: [B * N]
                    node_level_pred_labels = (anomaly_scores.cpu() > torch.tensor(node_dynamic_threshold).unsqueeze(0)).float().cpu().numpy()  # Shape: [B, N]

                    # Store node-level true and predicted labels for evaluation
                    all_node_true_labels.extend(true_labels_flat)
                    all_node_pred_labels.extend(node_level_pred_labels)

                
    return  all_node_log_probs,
        

def test_rstae(test_sequence, weights, model, verbose=False):
    model.eval()
    errors = []
    recons_speeds = []
    speeds = []

    for graph_sequence in tqdm(test_sequence, disable=not verbose):
        xhat = model(graph_sequence[0])
        error = (xhat - graph_sequence[1].x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        recons_speeds.append(xhat.detach().numpy()[:,1])
        speeds.append(graph_sequence[1].x[:,1].detach().numpy())
        errors.append(weighted)

    errors = np.array(errors)
    recons_speeds = np.array(recons_speeds)
    speeds = np.array(speeds)

    return errors, recons_speeds, speeds

def test_transformerae(test_sequence, weights, model, verbose=False):
    model.eval()
    errors = []
    recons_speeds = []
    speeds = []

    for sequence, current in tqdm(test_sequence, disable=not verbose):
        xhat = model(sequence)
        error = (xhat - current) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        recons_speeds.append(xhat.detach().numpy()[:,1])
        speeds.append(current[:,1].detach().numpy())
        errors.append(weighted)

    errors = np.array(errors)
    recons_speeds = np.array(recons_speeds)
    speeds = np.array(speeds)

    return errors, recons_speeds, speeds

def test_mlpae(test_sequence, weights, model, verbose=False):
    model.eval()
    errors = []
    recons_speeds = []
    speeds = []

    for sequence, current in tqdm(test_sequence, disable=not verbose):
        xhat = model(sequence).squeeze()
        error = (xhat - current) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        recons_speeds.append(xhat.detach().numpy())
        speeds.append(current[:,1].detach().numpy())
        errors.append(weighted)

    errors = np.array(errors)
    recons_speeds = np.array(recons_speeds)
    speeds = np.array(speeds)

    return errors, recons_speeds, speeds

def test_gcnae(test_sequence, weights, model, verbose=False):
    model.eval()
    errors = []
    recons_speeds = []
    speeds = []

    for graph_sequence in tqdm(test_sequence, disable=not verbose):
        xhat = model(graph_sequence)
        error = (xhat - graph_sequence.x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        recons_speeds.append(xhat.detach().numpy()[:,1])
        speeds.append(graph_sequence.x[:,1].detach().numpy())
        errors.append(weighted)

    errors = np.array(errors)
    recons_speeds = np.array(recons_speeds)
    speeds = np.array(speeds)

    return errors, recons_speeds, speeds

# Create a df to save results
def build_result_df(anomalies, lanes, mms, times, speeds, recons_speeds):
    results = pd.DataFrame({
        'Anomaly': anomalies,
        'Lane': lanes,
        'Milemarker': mms,
        'Time Index': times,
        'Speed': speeds,
        'Reconstructed Speed': recons_speeds
    })

    return results

# Convert index to time
def time_convert(time_indices: np.ndarray):
    return time_indices * 30

# Convert node indices to a list of the lane numbers for those nodes
def node_lane_convert(node_indices: np.ndarray):
    lane_num = node_indices % 4 + 1
    return lane_num

# Convert node indices to their corresponding milemarkers
def node_milemarker_convert(node_indices: np.ndarray, milemarkers: np.ndarray):
    milemarker_num = milemarkers[node_indices // 4]
    return milemarker_num

# Put all results into the df
def fill_result_df(anomalies, speeds, recons_speeds, timesteps):
    mms = node_milemarker_convert(np.array(range(anomalies.shape[1])), milemarkers)
    mms = np.tile(mms, anomalies.shape[0])

    lanes = node_lane_convert(np.array(range(anomalies.shape[1])))
    lanes = np.tile(lanes, anomalies.shape[0])

    times = np.array(range(anomalies.shape[0])) + timesteps - 1
    times = times.repeat(anomalies.shape[1])

    df = build_result_df(anomalies.flatten(), lanes, mms, times, speeds.flatten(), recons_speeds.flatten())

    return df

def train_f_gtnf(params, trainingparams, training_data, verbose=False):
    ae = FixedGraphTransformerNormalizingFlow(**params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)


    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            
            graph_sequence = graph_sequence[0]
            
            xhat = ae(graph_sequence) # encode and decode the sequence
            
            loss= torch.mean((torch.mean(xhat,dim=1))) # calculate the loss
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_TRACE_GRU(params, trainingparams, training_data, verbose=False):
    ae = TRACE_GRU(**params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)


    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            
            graph_sequence = graph_sequence[0]
            
            xhat = ae(graph_sequence) # encode and decode the sequence
            
            loss= torch.mean((torch.mean(xhat,dim=1))) # calculate the loss
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_TRACE_TCC(params, trainingparams, training_data, verbose=False):
    ae = TRACE_connected_components(**params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)


    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            
            graph_sequence = graph_sequence[0]
            
            xhat = ae(graph_sequence) # encode and decode the sequence
            
            loss= torch.mean((torch.mean(xhat,dim=1))) # calculate the loss
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_TRACE_TG(params, trainingparams, training_data, verbose=False):
    ae = TRACE_TG(**params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)


    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            
            graph_sequence = graph_sequence[0]
            
            xhat = ae(graph_sequence) # encode and decode the sequence
            
            loss= torch.mean((torch.mean(xhat,dim=1))) # calculate the loss
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)

            losses.append(loss.item())
            
        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses



def train_TRACE_LSTM(params, trainingparams, training_data, verbose=False):
    ae = TRACE_LSTM(**params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)


    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            
            graph_sequence = graph_sequence[0]
            
            xhat = ae(graph_sequence) # encode and decode the sequence
            
            loss= torch.mean((torch.mean(xhat,dim=1))) # calculate the loss
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_TRACE_GCN(params, trainingparams, training_data, verbose=False):
    ae = TRACE_GCN(**params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)


    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            
            graph_sequence = graph_sequence[0]
            
            xhat = ae(graph_sequence) # encode and decode the sequence
            
            loss= torch.mean((torch.mean(xhat,dim=1))) # calculate the loss
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_TRACE_Transformer(params, trainingparams, training_data, verbose=False):
    ae = TRACE_Transformer(**params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)


    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            
            graph_sequence = graph_sequence[0]
            
            xhat = ae(graph_sequence) # encode and decode the sequence
            
            loss= torch.mean((torch.mean(xhat,dim=1))) # calculate the loss
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def test_f_gtnf(test_sequence, model, verbose=False):
    model.eval()
    nll = []


    for graph_sequence in tqdm(test_sequence, disable=not verbose):
        
        xhat = model(graph_sequence[0])

        xhat=torch.mean(xhat,dim=1)
        nll.extend(xhat.detach().numpy())

    nll = np.array(nll)


    return nll

def compute_threshold_f_gtnf(data, model, method='max'):
    errors=test_f_gtnf(data, model)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()

from copy import deepcopy
def train_TRACE(params, trainingparams, training_data,validation_data, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = TRACE(**params).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=trainingparams.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    losses = []

    # training_data is now assumed to be a list of sequences,
    # where each sequence is [Data, Data, ..., Data] (length W).
    # We will batch these sequences.
    batch_size = trainingparams.batch_size
    

    best_val_loss = np.inf
    iter=0
    best_model = None
    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        random.shuffle(training_data)
        for i in range(0, len(training_data), batch_size):
            
            batch_sequences = training_data[i:i+batch_size]
            batch_sequences = [[data.to(device) for data in seq] for seq in batch_sequences]
            # If the last batch is smaller than batch_size, you can either skip or handle it.
            
            
            # Forward pass
            ae.train()
            xhat = ae(batch_sequences)
            
            # loss calculation
            # xhat: [B, W, N], take the mean for instance
            loss = torch.mean(torch.mean(xhat, dim=1))  # similar to original
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            del batch_sequences, xhat, loss
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')
        
        ae.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(validation_data), batch_size):
                batch_sequences = validation_data[i:i+batch_size]
                batch_sequences = [[data.to(device) for data in seq] for seq in batch_sequences]
                
                xhat = ae(batch_sequences)
                xhat = torch.mean(xhat, dim=1)
                val_loss = torch.mean(torch.mean(xhat, dim=1))
                val_losses.append(val_loss.item())
                # Cleanup
                del batch_sequences, xhat
                torch.cuda.empty_cache()
                
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)
            print(f'Validation loss: {val_loss}')
            if val_loss < best_val_loss:
                iter=0
                best_val_loss = val_loss
                best_model = deepcopy(ae.state_dict())
                
            else:
                iter+=1
                if iter>trainingparams.patience:
                    print(f'Early stopping at epoch {epoch_num}')
                    break
                
                
        

    return best_model, losses, best_val_loss

# def test_TRACE(test_sequence, model, verbose=False):
#     model.eval()
#     nll = []


#     for graph_sequence in tqdm(test_sequence, disable=not verbose):

#         xhat = model(graph_sequence)

#         xhat=torch.mean(xhat,dim=1)
#         nll.extend(xhat.detach().numpy())


#     nll = np.array(nll)


#     return nll
def test_TRACE(test_sequence, model, batch_size=32, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    nll = []

    # We do not need gradients during testing
    with torch.no_grad():
        # Iterate over test_sequence in batches
        for i in tqdm(range(0, len(test_sequence), batch_size), disable=not verbose):
            batch_sequences = test_sequence[i:i+batch_size]
            
            # Move each Data in each sequence to device
            batch_sequences = [[data.to(device) for data in seq] for seq in batch_sequences]

            # Forward pass on the batch
            xhat = model(batch_sequences)  # [B, W, N]

            # Reduce as before
            xhat = torch.mean(xhat, dim=1)  # [B, N]

            # Move back to CPU and convert to numpy
            nll.extend(xhat.detach().cpu().numpy())

    nll = np.array(nll)
    return nll
    
    