import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from models import SpatioTemporalAutoencoder
from parameters import GATAEParameters, RSTAEParameters, GraphAEParameters, MLPAEParameters, TrainingParameters, GraphTransformerNormalizingFlowParameters, STAEParameters, TransformerAEParameters
from datautils import  get_full_data, normalize_data, generate_edges, generate_relational_edges, load_best_parameters, get_full_data, label_anomalies
from training import train_gatae, train_rstae, train_gcnae, train_mlpae,train_gtnf, test_rstae, test_gcnae, test_mlpae, test_gtnf, train_stae, train_transformerae, test_transformerae, test_f_gtnf, train_f_gtnf, train_TRACE_TCC, train_TRACE_TG, train_TRACE, train_TRACE_Transformer,train_TRACE_GCN
torch_geometric.seed_everything(42)
model_type = None
hide_anomalies = False


def choose_parameters(trial):
    if model_type == "rstae":
        parameters = RSTAEParameters(
            num_features=3,
            latent_dim=trial.suggest_categorical('latent_dim', [32, 64, 128, 256]),
            gcn_hidden_dim=trial.suggest_categorical('gcn_hidden_dim', [16, 32, 64, 128, 256]),
            dropout=trial.suggest_float('dropout', 0, 0.5),
            num_gcn=trial.suggest_int('num_gcn', 1, 5)
        )
    elif model_type == "gcn":
        parameters = GraphAEParameters(
            num_features=3,
            latent_dim=trial.suggest_categorical('latent_dim', [32, 64, 128, 256]),
            gcn_hidden_dim=trial.suggest_categorical('gcn_hidden_dim', [16, 32, 64, 128, 256]),
            dropout=trial.suggest_float('dropout', 0, 0.5),
            num_gcn=trial.suggest_int('num_gcn', 1, 5)
        )
    elif model_type == "gat":
        parameters = GATAEParameters(
            num_features=3,
            latent_dim=trial.suggest_categorical('latent_dim', [32, 64, 128, 256]),
            gcn_hidden_dim=trial.suggest_categorical('gcn_hidden_dim', [16, 32, 64, 128, 256]),
            dropout=trial.suggest_float('dropout', 0, 0.5),
            num_layers=trial.suggest_int('num_layers', 1, 5),
            num_heads=trial.suggest_int('num_heads', 1, 5)
        )
    elif model_type == "mlp":
        parameters = MLPAEParameters(
            num_features=3,
            latent_dim=2,
            hidden_dim=trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256])
        )
    elif model_type == "gcn_lstm":
        parameters = STAEParameters(
            num_features=3,
            latent_dim=trial.suggest_categorical('latent_dim', [32, 64, 128, 256]),
            gcn_hidden_dim=trial.suggest_categorical('gcn_hidden_dim', [16, 32, 64, 128, 256]),
            lstm_hidden_dim=trial.suggest_categorical('lstm_hidden_dim', [16, 32, 64, 128, 256]),
            lstm_num_layers=trial.suggest_int('lstm_num_layers', 1, 5),
            dropout=trial.suggest_float('dropout', 0, 0.5)
        )
    elif model_type == "gtnf":
        parameters  = {
            'input_features': 3,
            'num_sensors': 196,
            'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256]),
            'num_heads': trial.suggest_int('num_heads', 1, 2),
            'num_transformer_layers': trial.suggest_int('num_transformer_layers', 1, 3),
            'output_dim':  trial.suggest_categorical('output_dim', [16, 32, 64, 128]),
            'num_gcn_layers': trial.suggest_int('num_gcn_layers', 1, 5),
            'flow_layers':trial.suggest_int('flow_layers', 1, 5),
            'dropout_rate':trial.suggest_float('dropout_rate', 0, 0.5)

    }
    elif model_type == "f_gtnf" or model_type == "TCC" or model_type == "TRACE_TG" or model_type == "TRACE":
        parameters  = {
            'input_features': 3,
            'num_sensors': 196,
            'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256]),
            'num_heads': 1,
            'num_transformer_layers': trial.suggest_int('num_transformer_layers', 1, 2),
            'output_dim':  trial.suggest_categorical('output_dim', [16, 32, 64, 128]),
            'num_gcn_layers': trial.suggest_int('num_gcn_layers', 1, 5),
            'flow_layers':trial.suggest_int('flow_layers', 1, 5),
            'n_hidden_flow':trial.suggest_int('n_hidden_flow', 1, 5),
            'dropout_rate':trial.suggest_float('dropout_rate', 0, 0.5)

    }
    elif model_type == "TRACE_GCN":
        parameters  = {
            'input_features': 3,
            'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256]),
            'output_dim':  trial.suggest_categorical('output_dim', [16, 32, 64, 128]),
            'num_gcn_layers': trial.suggest_int('num_gcn_layers', 1, 5),
            'flow_layers':trial.suggest_int('flow_layers', 1, 5),
            'n_hidden_flow':trial.suggest_int('n_hidden_flow', 1, 5),
            'dropout_rate':trial.suggest_float('dropout_rate', 0, 0.5)

    }
    elif model_type == "TRACE_Transformer":
        parameters  = {
            'input_features': 3,
            'num_sensors': 196,
            'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256]),
            'num_heads': 1,
            'num_transformer_layers': trial.suggest_int('num_transformer_layers', 1, 2),
            'output_dim':  trial.suggest_categorical('output_dim', [16, 32, 64, 128]),
            'flow_layers':trial.suggest_int('flow_layers', 1, 5),
            'n_hidden_flow':trial.suggest_int('n_hidden_flow', 1, 5),
            'dropout_rate':trial.suggest_float('dropout_rate', 0, 0.5)

    }
    elif model_type == "transformer":
        parameters = TransformerAEParameters(
            num_features=3,
            latent_dim=trial.suggest_categorical('latent_dim', [1,2]),
            hidden_dim=trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
            num_heads=1,
            num_layers=trial.suggest_int('num_layers', 1, 5),
            sequence_length=trial.params['timesteps']
)
    else:
        raise NotImplementedError("Please one of the allowed model types.")
    
    return parameters

def train_model(hyperparams, training_params, training_data, mse_weights, verbose=False):
    if model_type == "rstae":
        model, losses = train_rstae(hyperparams, training_params, training_data, mse_weights, verbose)
    elif model_type == "gcn":
        model, losses = train_gcnae(hyperparams, training_params, training_data, mse_weights, verbose)
    elif model_type == "gat":
        model, losses = train_gatae(hyperparams, training_params, training_data, mse_weights, verbose)
    elif model_type == "mlp":
        model, losses = train_mlpae(hyperparams, training_params, training_data, mse_weights, verbose)
    elif model_type == "gcn_lstm":
        model, losses = train_stae(hyperparams, training_params, training_data, mse_weights, verbose)
    elif model_type == "transformer":
        model, losses = train_transformerae(params=hyperparams, trainingparams=training_params, training_data=training_data, mse_weights=mse_weights, verbose=verbose)
    elif model_type == "gtnf":
        model, losses = train_gtnf(hyperparams, training_params, training_data, verbose)
    elif model_type == "f_gtnf":
        model, losses = train_f_gtnf(hyperparams, training_params, training_data, verbose)
    elif model_type == "TRACE":
        model, losses = train_TRACE(hyperparams, training_params, training_data, verbose)

    elif model_type == "TRACE_GCN":
        model, losses = train_TRACE_GCN(hyperparams, training_params, training_data, verbose)
    elif model_type == "TRACE_Transformer":
        model, losses = train_TRACE_Transformer(hyperparams, training_params, training_data, verbose)
    
    return model, losses


def edge_index_to_adj(edge_index, num_nodes):
    # Initialize an NxN zero matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    # For each edge in edge_index, set adj_matrix[source, target] = 1
    adj_matrix[edge_index[0], edge_index[1]] = 1
    
    adj_matrix=adj_matrix+adj_matrix.t()
    
    return adj_matrix

def sequence_gtnf(data, timesteps, hide_anomalies=False):
    sequence = []
    relational_edges, relations = generate_relational_edges(milemarkers=list(range(49)), timesteps=timesteps)
    static_edges = generate_edges(milemarkers=list(range(49)))
    
    adj_matrix =  edge_index_to_adj(static_edges, 196)
    days = data['day']
    anomalies = data['anomaly']
    data_vals = data[['occ', 'speed', 'volume']]
    unix = data['unix_time']
    unique_times = np.unique(data['unix_time'])
    kept_indices = []

    for index, t in enumerate(unique_times[timesteps:]): # skip first 'timesteps'
        adj_matrices = []
        feature_matrices = []
        anomaly_matrices = []

        backward_index = range(index-1, index-timesteps-1, -1)
        backward_times = [unique_times[i] for i in backward_index]
        curr_day = np.unique(data[data['unix_time']==backward_times[-1]]['day'])[0]
        contains_anomaly = np.any([np.unique(data[data['unix_time']==i]['anomaly'])[0] for i in backward_times])
        is_curr_day = np.all([np.unique(data[data['unix_time']==i]['day'])[0]==curr_day for i in backward_times])

        if (hide_anomalies and contains_anomaly) or not is_curr_day:
            continue
        
        kept_indices.append(index+timesteps)

        for i in backward_times:
            # Node feature matrix
            node_features = data[data['unix_time']==i][['occ', 'speed', 'volume']].to_numpy()
            feature_matrices.append(torch.tensor(node_features, dtype=torch.float32))
            
            # Adjacency matrix (static for each time step)
            adj_matrices.append(adj_matrix)  # Assuming static_edges is in PyTorch Geometric format
            
            # Anomaly matrix (if any)
            anomaly_matrix = data[data['unix_time']==i]['anomaly'].to_numpy()
            anomaly_matrices.append(torch.tensor(anomaly_matrix, dtype=torch.float32))

        # Stack the matrices to create tensors for the sequence
        adj_matrices = torch.stack(adj_matrices)
        feature_matrices = torch.stack(feature_matrices)
        anomaly_matrices = torch.stack(anomaly_matrices)

        # Pack into a torch_geometric Data object
        data_new = Data(
            x=feature_matrices.unsqueeze(0),  # Node features for the window
            adj=adj_matrices.unsqueeze(0),    # Adjacency matrices for the window
            y=anomaly_matrices.unsqueeze(0)   # Anomalies for the window (optional)
        )
        
        sequence.append(data_new)

    return sequence, kept_indices

def sequence_transformer(data, timesteps, hide_anomalies=False):
    sequence = []
    days = data['day']
    anomalies = data['anomaly']
    data_vals = data[['occ', 'speed', 'volume']]
    unix = data['unix_time']
    unique_times = np.unique(data['unix_time'])
    kept_indices = []

    for index, t in enumerate(unique_times[timesteps:]): # skip first 'timesteps'
        data_t = []
        backward_index = range(index-1, index-timesteps-1, -1)
        backward_times = [unique_times[i] for i in backward_index]
        curr_day = np.unique(data[data['unix_time']==backward_times[-1]]['day'])[0]
        contains_anomaly = np.any([np.unique(data[data['unix_time']==i]['anomaly'])[0] for i in backward_times])
        is_curr_day = np.all([np.unique(data[data['unix_time']==i]['day'])[0]==curr_day for i in backward_times])

        if (hide_anomalies and contains_anomaly) or not is_curr_day:
            continue
        
        kept_indices.append(index+timesteps)

        for i in backward_times:
            data_t.append(data[data['unix_time']==i][['occ', 'speed', 'volume']].to_numpy()) # assumes time indices come sequentially, with full data it may not

        combined = np.array(data_t)
        combined = np.swapaxes(combined, 0, 1)
        combined = torch.tensor(combined, dtype=torch.float32)
        
        curr_data = combined[:,-1,:]
        sequence.append([combined, curr_data])

    return sequence, kept_indices

def sequence_gcnae(data, timesteps, hide_anomalies=False):
    sequence = []
    static_edges = generate_edges(milemarkers=list(range(49)))
    unique_times = np.unique(data['unix_time'])
    kept_indices = []

    for index, t in enumerate(unique_times):
        data_t = []
        contains_anomaly = np.any([np.unique(data[data['unix_time']==t]['anomaly'])[0]])

        if (hide_anomalies and contains_anomaly):
            continue
        
        kept_indices.append(index)

        data_t.append(data[data['unix_time']==t][['occ', 'speed', 'volume']].to_numpy()) # assumes time indices come sequentially, with full data it may not
        
        curr_data = data_t[-1]
        curr_graph = Data(x=torch.tensor(curr_data, dtype=torch.float32), edge_index=static_edges)
        sequence.append(curr_graph)

    return sequence, kept_indices


def sequence_mlp(data, timesteps, hide_anomalies=False):
    sequence = []
    days = data['day']
    anomalies = data['anomaly']
    data_vals = data[['occ', 'speed', 'volume']]
    unix = data['unix_time']
    unique_times = np.unique(data['unix_time'])
    kept_indices = []

    for index, t in enumerate(unique_times[timesteps:]): # skip first 'timesteps'
        data_t = []
        backward_index = range(index-1, index-timesteps-1, -1)
        backward_times = [unique_times[i] for i in backward_index]
        curr_day = np.unique(data[data['unix_time']==backward_times[-1]]['day'])[0]
        contains_anomaly = np.any([np.unique(data[data['unix_time']==i]['anomaly'])[0] for i in backward_times])
        is_curr_day = np.all([np.unique(data[data['unix_time']==i]['day'])[0]==curr_day for i in backward_times])

        if (hide_anomalies and contains_anomaly) or not is_curr_day:
            continue
        
        kept_indices.append(index+timesteps)

        for i in backward_times:
            data_t.append(data[data['unix_time']==i][['occ', 'speed', 'volume']].to_numpy()) # assumes time indices come sequentially, with full data it may not

        combined = np.array(data_t)
        combined = np.swapaxes(combined, 0, 1)
        combined = torch.tensor(combined, dtype=torch.float32)
        
        curr_data = combined[:,-1,:]
        sequence.append([combined, curr_data])

    return sequence, kept_indices

def sequence_TRACE(data, timesteps, hide_anomalies=False):
    sequence = []
    # relational_edges, relations = generate_relational_edges(milemarkers=list(range(49)), timesteps=timesteps)
    static_edges = generate_edges(milemarkers=list(range(49)))
    days = data['day']
    anomalies = data['anomaly']
    data_vals = data[['occ', 'speed', 'volume']]
    unix = data['unix_time']
    unique_times = np.unique(data['unix_time'])
    kept_indices = []

    for index, t in enumerate(unique_times[timesteps:]): # skip first 'timesteps'
        data_t = []
        backward_index = range(index-1, index-timesteps-1, -1)
        backward_times = [unique_times[i] for i in backward_index]
        curr_day = np.unique(data[data['unix_time']==backward_times[-1]]['day'])[0]
        contains_anomaly = np.any([np.unique(data[data['unix_time']==i]['anomaly'])[0] for i in backward_times])
        is_curr_day = np.all([np.unique(data[data['unix_time']==i]['day'])[0]==curr_day for i in backward_times])

        if (hide_anomalies and contains_anomaly) or not is_curr_day:
            continue
        
        kept_indices.append(index+timesteps)

        for i in backward_times:
            data_t.append(Data(x=torch.tensor(data[data['unix_time']==i][['occ', 'speed', 'volume']].to_numpy(), dtype=torch.float32), edge_index=static_edges)) # assumes time indices come sequentially, with full data it may not

        curr_graph = data_t[0]
        sequence.append(data_t[::-1])

    return sequence, kept_indices

def sequence_stae(data, timesteps, hide_anomalies=False):
    sequence = []
    relational_edges, relations = generate_relational_edges(milemarkers=list(range(49)), timesteps=timesteps)
    static_edges = generate_edges(milemarkers=list(range(49)))
    days = data['day']
    anomalies = data['anomaly']
    data_vals = data[['occ', 'speed', 'volume']]
    unix = data['unix_time']
    unique_times = np.unique(data['unix_time'])
    kept_indices = []

    for index, t in enumerate(unique_times[timesteps:]): # skip first 'timesteps'
        data_t = []
        backward_index = range(index-1, index-timesteps-1, -1)
        backward_times = [unique_times[i] for i in backward_index]
        curr_day = np.unique(data[data['unix_time']==backward_times[-1]]['day'])[0]
        contains_anomaly = np.any([np.unique(data[data['unix_time']==i]['anomaly'])[0] for i in backward_times])
        is_curr_day = np.all([np.unique(data[data['unix_time']==i]['day'])[0]==curr_day for i in backward_times])

        if (hide_anomalies and contains_anomaly) or not is_curr_day:
            continue
        
        kept_indices.append(index+timesteps)

        for i in backward_times:
            data_t.append(Data(x=torch.tensor(data[data['unix_time']==i][['occ', 'speed', 'volume']].to_numpy(), dtype=torch.float32), edge_index=static_edges)) # assumes time indices come sequentially, with full data it may not

        curr_graph = data_t[0]
        sequence.append([data_t[::-1], curr_graph])

    return sequence, kept_indices

def sequence_rstae(data, timesteps, hide_anomalies=False):
    sequence = []
    relational_edges, relations = generate_relational_edges(milemarkers=list(range(49)), timesteps=timesteps)
    static_edges = generate_edges(milemarkers=list(range(49)))
    days = data['day']
    anomalies = data['anomaly']
    data_vals = data[['occ', 'speed', 'volume']]
    unix = data['unix_time']
    unique_times = np.unique(data['unix_time'])
    kept_indices = []

    for index, t in enumerate(unique_times[timesteps:]): # skip first 'timesteps'
        data_t = []
        backward_index = range(index-1, index-timesteps-1, -1)
        backward_times = [unique_times[i] for i in backward_index]
        curr_day = np.unique(data[data['unix_time']==backward_times[-1]]['day'])[0]
        contains_anomaly = np.any([np.unique(data[data['unix_time']==i]['anomaly'])[0] for i in backward_times])
        is_curr_day = np.all([np.unique(data[data['unix_time']==i]['day'])[0]==curr_day for i in backward_times])

        if (hide_anomalies and contains_anomaly) or not is_curr_day:
            continue
        
        kept_indices.append(index+timesteps)

        for i in backward_times:
            data_t.append(data[data['unix_time']==i][['occ', 'speed', 'volume']].to_numpy()) # assumes time indices come sequentially, with full data it may not

        node_data = np.concatenate(data_t[::-1])
        pyg_data = Data(x=torch.tensor(node_data, dtype=torch.float32), edge_index=relational_edges, edge_attr=torch.tensor(relations, dtype=torch.long))
        
        # curr_data = data_t[-1]
        curr_data = data_t[0]
        
        curr_graph = Data(x=torch.tensor(curr_data, dtype=torch.float32), edge_index=static_edges)
        sequence.append([pyg_data, curr_graph])

    return sequence, kept_indices

def get_sequence(train_data, timesteps):
    if model_type == "rstae" or model_type == "gat" :
        data, indices = sequence_rstae(train_data, timesteps, hide_anomalies=hide_anomalies)
    elif model_type == "gcn":
        data,  indices = sequence_gcnae(train_data, timesteps, hide_anomalies=hide_anomalies)
    elif model_type == "mlp":
        data, indices = sequence_mlp(train_data, timesteps, hide_anomalies=hide_anomalies)
    elif model_type == "gcn_lstm" or model_type == "f_gtnf" or model_type == "TCC" or model_type == "TRACE_TG" or model_type == "TRACE_GCN" or model_type == "TRACE_Transformer":
        data, indices = sequence_stae(train_data, timesteps, hide_anomalies=hide_anomalies)
    elif model_type == "TRACE" :
        data, indices = sequence_TRACE(train_data, timesteps, hide_anomalies=hide_anomalies)
    elif model_type == "transformer":
        data, indices = sequence_transformer(train_data, timesteps, hide_anomalies=hide_anomalies)
    elif model_type == "gtnf":
        data, indices = sequence_gtnf(train_data, timesteps, hide_anomalies=hide_anomalies)
        
    return data, indices

def validate_model(valid_data, mse_weights, model):
    if model_type == "rstae" or model_type == "gat" or model_type == "gcn_lstm":
        errors, _, _ = test_rstae(valid_data, mse_weights, model)
    elif model_type == "gcn":
        errors, _, _ = test_gcnae(valid_data, mse_weights, model)
    elif model_type == "mlp":
        errors, _, _ = test_mlpae(valid_data, mse_weights, model)
    elif model_type == "transformer":
        errors = test_transformerae(valid_data,mse_weights, model)
        
    elif model_type == "gtnf":
        errors = test_gtnf(valid_data, model)
    elif model_type == "f_gtnf" or model_type == "TRACE" or model_type == "TCC" or model_type == "TRACE_TG" or model_type == "TRACE_GCN" or model_type == "TRACE_Transformer":
        errors = test_f_gtnf(valid_data, model)
    
    return errors

def objective(trial):
    torch_geometric.seed_everything(42)
    best_mse = trial.study.user_attrs.get('best_mse', float('inf'))
    train_params = TrainingParameters(
        learning_rate=trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
        batch_size=1 if not model_type=="gtnf" and not model_type== 'TRACE' else trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        timesteps=trial.suggest_int('timesteps', 2, 10) if not model_type=="mlp" else 1,
        n_epochs=trial.suggest_int('epochs', 1, 10) if  not model_type== 'TRACE' else trial.suggest_categorical('n_epoch', [ 20, 30, 40,50]),
    )

    # Crash-free morning as training
    data, test_data, _ = get_full_data()
    data = normalize_data(data)
    data = label_anomalies(data)
    length = len(data.day.unique())
    train_length = int(length * 0.8)
    val_length = length - train_length
    print(train_length, val_length)

    # Get the unique days for train and validation split
    train_days = data.day.unique()[:train_length]
    val_days = data.day.unique()[train_length:]

    # Use .isin() to filter the DataFrame based on the days
    train_data = data[data.day.isin(train_days)]
    val_data = data[data.day.isin(val_days)]
    
    train_data, _ = get_sequence(train_data, train_params.timesteps)
    valid_data, _ = get_sequence(val_data, train_params.timesteps)
    
    if model_type == "gtnf":
        train_data = DataLoader(train_data, batch_size=train_params.batch_size, shuffle=False)
        valid_data = DataLoader(valid_data, batch_size=train_params.batch_size, shuffle=False)

    


    # Sequentially split the dataset

    
    # train_data = get_data(1, train_params.timesteps)

    params = choose_parameters(trial)
    
    mse_weights = [1,1,1] # loss function weights
    try:
        if model_type == "TRACE":
            model, losses, val_loss = train_TRACE(params=params, trainingparams=train_params, training_data=train_data,validation_data=valid_data, verbose=True)
            return val_loss
        model, losses = train_model(hyperparams=params, training_params=train_params, training_data=train_data, mse_weights=mse_weights, verbose=True)
    except (torch.cuda.OutOfMemoryError, ValueError) as e:
        print(f"Error: {e}")
        return float('inf')
    # Using another crash-free morning as validation
    # valid_data = get_data(6, train_params.timesteps) 
    
    errors = validate_model(valid_data, mse_weights, model)
    
    curr_mse = float(np.mean(errors))
    # if better than best mse, call save_model function
    # if curr_mse < best_mse:
    #     trial.study.set_user_attr('best_mse', curr_mse)
    #     save_model(model, f'opt_{model_type}_{trial.number}')

    return curr_mse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-m', '--model',
        choices=['rstae', 'gcn', 'mlp', 'gat','gcn_lstm','transformer', 'gtnf','f_gtnf','TRACE','TCC', 'TRACE_TG', 'TRACE_GCN', 'TRACE_Transformer'],
        required=True,
        help='Choose a model: rstae, gcn, mlp, gat, gcn_lstm, transformer, gtnf, f_gtnf or TRACE'
    )

    
    args = parser.parse_args()
    study_name = args.model # what to call the study
    # study_name = study_name + "_v2"
    model_type = args.model # used in optuna optimization to choose relevant functions
    
    hide_anomalies = False

    
    storage_subdirectory = 'studies_non_hide'
    print(f"Running optimization for {model_type} with hide_anomalies={hide_anomalies}")
    print(type(hide_anomalies))
    storage_url = f'sqlite:///{os.path.join(storage_subdirectory, study_name)}.db'

    study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_url, load_if_exists=True)
    study.optimize(objective, n_trials=1, n_jobs=1,catch=(torch.cuda.OutOfMemoryError, ValueError))