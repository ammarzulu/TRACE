import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from NF import RealNVP
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric as pyg
from parameters import GATAEParameters, RSTAEParameters, STAEParameters, GATSTAEParameters, GraphAEParameters, TransformerAEParameters, MLPAEParameters
from datautils import generate_edges

class GraphEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_layers, dropout_percentage=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Define a ModuleList to store the dynamic number of RGCNConv layers
        self.conv_layers = nn.ModuleList([GCNConv(num_features, hidden_dim)])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, latent_dim)

        self.dropout_percentage = dropout_percentage
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Loop through the RGCNConv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index, edge_attr)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout_percentage, training=self.training)
        x = pyg.nn.global_mean_pool(x, data.batch)
        z = self.fc(x)
        return z
    

class LatentTemporalAggregator(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers):
        super(LatentTemporalAggregator, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, latent_size)

    def forward(self, z):
        # Basically a LSTM with batch size 1
        z = z.unsqueeze(0)
        z, (_, _) = self.lstm(z)
        return self.head(z)
    

# class GraphDecoder(nn.Module):
#     def __init__(self, num_features, hidden_dim, latent_dim, num_nodes=196, replicate_latent=False):
#         super(GraphDecoder, self).__init__()
#         self.fc = nn.Linear(latent_dim, num_nodes * latent_dim)
#         self.conv1 = GCNConv(latent_dim, hidden_dim)
#         # self.conv2 = GCNConv(hidden_dim, num_features)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.conv4 = GCNConv(hidden_dim, num_features)
        
#         # self.head = nn.Linear(num_features)

#         self.hidden_dim = hidden_dim
#         self.num_features = num_features
#         self.latent_dim = latent_dim
#         self.replicate_latent = replicate_latent
#         self.num_nodes = num_nodes

#     def forward(self, z, edge_index):
#         if self.replicate_latent:
#             # 'Trick' to make the decoding work
#             # Replicate the latent vector for every node
#             # Suggested by ChatGPT
#             z = z.unsqueeze(0).expand(self.num_nodes, -1)
#         else:
#             # Nonlinear projection to increased size and give fixed latent space to each node
#             # Makes the speed reconstruction much more expressive than above
#             z = z.unsqueeze(0)
#             z = self.fc(z)
#             z = F.relu(z)
#             z = z.view(self.num_nodes, self.latent_dim)

#         x = self.conv1(z, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x = self.conv4(x, edge_index)
#         return x
    
class GraphDecoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_nodes=196, replicate_latent=False, num_gcn_layers=5):
        super(GraphDecoder, self).__init__()

        self.fc = nn.Linear(latent_dim, num_nodes * latent_dim)
        self.gcn_layers = nn.ModuleList([GCNConv(latent_dim, hidden_dim)])
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn_layers.append(GCNConv(hidden_dim, num_features))

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.replicate_latent = replicate_latent
        self.num_nodes = num_nodes

    def forward(self, z, edge_index):
        if self.replicate_latent:
            x = z.unsqueeze(0).expand(self.num_nodes, -1)
        else:
            x = z.unsqueeze(0)
            x = self.fc(x)
            x= F.relu(x)
            x = x.view(self.num_nodes, self.latent_dim)

        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:  # Apply ReLU for all layers except the last one
                x = F.relu(x)

        return x

    

class SpatioTemporalAutoencoder(nn.Module):
    def __init__(self, params: STAEParameters):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.enc = GraphEncoder(num_features=params.num_features, hidden_dim=params.gcn_hidden_dim, latent_dim=params.latent_dim, dropout_percentage=params.dropout, num_layers=2)
        self.temp_agg = LatentTemporalAggregator(latent_size=params.latent_dim, hidden_size=params.lstm_hidden_dim, num_layers=params.lstm_num_layers)
        self.dec = GraphDecoder(num_features=params.num_features, hidden_dim=params.gcn_hidden_dim, latent_dim=params.latent_dim, num_gcn_layers=2)

    def forward(self, temporal_graphs):
        edge_index = temporal_graphs[-1].edge_index
        # For each graph in the time window, apply the GraphEncoder
        latent_vectors = [self.enc(graph) for graph in temporal_graphs]

        # This gives a matrix of latent features
        latent_mat = torch.cat(latent_vectors) 

        # Run this through LatentTemporalAggregator
        aggregated = self.temp_agg(latent_mat).squeeze()[-1,:]

        # Feed the temporal aggregation through the graph decoder to construct a graph of the same structure
        graph_hat = self.dec(aggregated, edge_index)
        return graph_hat
    
    

class GATDecoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_nodes=196, replicate_latent=False, heads=1):
        super(GATDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, num_nodes * latent_dim)
        self.conv1 = GATConv(latent_dim, hidden_dim, heads=heads) # same as above but using GAT instead of GCN
        self.conv2 = GATConv(hidden_dim * heads, num_features, heads=heads)

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.replicate_latent = replicate_latent
        self.num_nodes = num_nodes
    
    def forward(self, z, edge_index):
        if self.replicate_latent:
            # 'Trick' to make the decoding work
            # Replicate the latent vector for every node
            # Suggested by ChatGPT
            z = z.unsqueeze(0).expand(self.num_nodes, -1)
        else:
            z = z.unsqueeze(0)
            z = self.fc(z)
            z = F.relu(z)
            z = z.view(self.num_nodes, self.latent_dim)

        x = self.conv1(z, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    

class GATSpatioTemporalAutoencoder(nn.Module):
    def __init__(self, params: GATSTAEParameters):
        super(GATSpatioTemporalAutoencoder, self).__init__()
        self.enc = GATEncoder(num_features=params.num_features, hidden_dim=params.gcn_hidden_dim, latent_dim=params.latent_dim, dropout_percentage=params.dropout, heads=params.gat_heads)
        self.temp_agg = LatentTemporalAggregator(latent_size=params.latent_dim, hidden_size=params.lstm_hidden_dim, num_layers=params.lstm_num_layers)
        self.dec = GATDecoder(num_features=params.num_features, hidden_dim=params.gcn_hidden_dim, latent_dim=params.latent_dim, heads=params.gat_heads)

    def forward(self, temporal_graphs):
        edge_index = temporal_graphs[-1].edge_index
        # For each graph in the time window, apply the GATEncoder
        latent_vectors = [self.enc(graph) for graph in temporal_graphs]

        # This gives a matrix of latent features
        latent_mat = torch.cat(latent_vectors) 

        # Run this through LatentTemporalAggregator
        aggregated = self.temp_agg(latent_mat).squeeze()[-1, :]

        # Feed the temporal aggregation through the GATDecoder to construct a graph of the same structure
        graph_hat = self.dec(aggregated, edge_index)
        return graph_hat
    
class GATEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_layers, dropout_percentage=0.1, num_heads=1):
        super().__init__()
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList([GATConv(num_features, hidden_dim, num_heads=num_heads)])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim, num_relations=num_heads))

        self.fc = nn.Linear(hidden_dim, latent_dim)

        self.dropout_percentage = dropout_percentage
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Loop through the RGCNConv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout_percentage, training=self.training)
        x = pyg.nn.global_mean_pool(x, data.batch)
        z = self.fc(x)
        return z
    
class GATDecoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_nodes=196, replicate_latent=False, num_layers=3, num_heads=1):
        super().__init__()

        self.fc = nn.Linear(latent_dim, num_nodes * latent_dim)
        self.gat_layers = nn.ModuleList([GATConv(latent_dim, hidden_dim, num_heads=num_heads)])
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, num_heads=num_heads))
        self.gat_layers.append(GATConv(hidden_dim, num_features, num_heads=num_heads))

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.replicate_latent = replicate_latent
        self.num_nodes = num_nodes

    def forward(self, z, edge_index):
        if self.replicate_latent:
            x = z.unsqueeze(0).expand(self.num_nodes, -1)
        else:
            x = z.unsqueeze(0)
            x = self.fc(x)
            x= F.relu(x)
            x = x.view(self.num_nodes, self.latent_dim)

        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:  # Apply ReLU for all layers except the last one
                x = F.relu(x)

        return x
    
class RelationalGraphEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_layers, dropout_percentage=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Define a ModuleList to store the dynamic number of RGCNConv layers
        self.conv_layers = nn.ModuleList([RGCNConv(num_features, hidden_dim, num_relations=6)])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=6))

        self.fc = nn.Linear(hidden_dim, latent_dim)

        self.dropout_percentage = dropout_percentage
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Loop through the RGCNConv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index, edge_attr)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout_percentage, training=self.training)
        x = pyg.nn.global_mean_pool(x, data.batch)
        z = self.fc(x)
        return z
    
class RelationalSTAE(nn.Module):
    def __init__(self, parameters: RSTAEParameters):
        super().__init__()
        self.reconstructed_index = generate_edges(list(range(49))) # should probably pass the milemarker programitaclly
        self.enc = RelationalGraphEncoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, dropout_percentage=parameters.dropout, num_layers=parameters.num_gcn)
        self.dec = GraphDecoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, num_nodes=196, num_gcn_layers=parameters.num_gcn)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z, self.reconstructed_index)
        return xhat
    
class GraphAE(nn.Module):
    def __init__(self, parameters: GraphAEParameters):
        super().__init__()
        self.reconstructed_index = generate_edges(list(range(49))) # should probably pass the milemarker programitaclly
        self.enc = GraphEncoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, dropout_percentage=parameters.dropout, num_layers=parameters.num_gcn)
        self.dec = GraphDecoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, num_nodes=196, num_gcn_layers=parameters.num_gcn)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z, self.reconstructed_index)
        return xhat
    
    
class TransformerAutoencoder(nn.Module):
    def __init__(self, parameters: TransformerAEParameters):
        super(TransformerAutoencoder, self).__init__()
        self.input_dim = parameters.num_features
        self.sequence_length = parameters.sequence_length 

        # The positional embeddings are learned.
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=parameters.num_features,
                nhead=parameters.num_heads,
                dim_feedforward=parameters.hidden_dim,
                batch_first=True
            ),
            num_layers=parameters.num_layers
        )

        # Projection to Latent Dimension
        # self.projection = nn.Linear(self.input_dim*self.sequence_length, parameters.latent_dim)
        self.projection = nn.Linear(self.input_dim, parameters.latent_dim)
        

        # Decoder
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=parameters.latent_dim,  # Change input dimension for decoder
                nhead=parameters.num_heads,
                dim_feedforward=parameters.hidden_dim,
                batch_first=True
            ),
            num_layers=parameters.num_layers
        )

        # Fully connected layer for output
        self.fc_out = nn.Linear(parameters.latent_dim, parameters.num_features)

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        # encoded = encoded.view(-1, self.input_dim*self.sequence_length)
        projected = self.projection(encoded)
        # projected = projected.unsqueeze(1)
        # print(projected.shape)

        # Decoding
        decoded = self.decoder(projected)

        # Fully connected layer
        output = self.fc_out(decoded)[:,-1,:]

        return output
    
class GATAE(nn.Module):
    def __init__(self, parameters: GATAEParameters):
        super().__init__()
        self.reconstructed_index = generate_edges(list(range(49)))
        self.enc = GATEncoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, dropout_percentage=parameters.dropout, num_layers=parameters.num_layers, num_heads=parameters.num_heads)
        self.dec = GATDecoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, num_nodes=196, num_layers=parameters.num_layers, num_heads=parameters.num_heads)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z, self.reconstructed_index)
        return xhat
    
    
class MLPAutoencoder(nn.Module):
    def __init__(self, parameters: MLPAEParameters):
        super(MLPAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(parameters.num_features, parameters.hidden_dim),
            nn.ReLU(),
            nn.Linear(parameters.hidden_dim, parameters.latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(parameters.latent_dim, parameters.hidden_dim),
            nn.ReLU(),
            nn.Linear(parameters.hidden_dim, parameters.num_features)
        )

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        
        # Decoding
        decoded = self.decoder(encoded)

        return decoded

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, X, adj):
        adj_hat = adj + torch.eye(adj.size(-1)).to(adj.device)  # Add self-loops
        D = torch.diag_embed(torch.sum(adj_hat, dim=-1)**-0.5)  # Degree matrix
        adj_norm = torch.matmul(torch.matmul(D, adj_hat), D)  # Normalized adjacency matrix
        X = torch.matmul(adj_norm, X)  # A'X
        X = self.linear(X)  # Apply linear transformation (XW)
        return F.relu(X)

# GCN Model (modified to handle adjacency per time step)
class GCNModel(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim, num_layers):
        super(GCNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(input_features, hidden_dim))  # First GCN layer
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))  # Hidden GCN layers
        self.layers.append(GCNLayer(hidden_dim, output_dim))  # Final GCN layer

    def forward(self, X, adj):
        B, W, N, _ = adj.shape  # Batch, Time Window, Nodes, Nodes
        output = []
        
        # Loop through each time step to apply GCN with the corresponding adjacency matrix
        for w in range(W):
            X_w = X[:, w, :, :]  # Features at time step w
            adj_w = adj[:, w, :, :]  # Adjacency matrix at time step w
            for layer in self.layers:
                X_w = layer(X_w, adj_w)  # Apply GCN layer at time step w
            output.append(X_w)  # Collect output for each time step
        
        # Stack the outputs from each time step back into shape [B, W, N, output_dim]
        return torch.stack(output, dim=1)

# Affine Coupling Layer for Normalizing Flow
# Affine Coupling Layer for Normalizing Flow (Fixed)
class AffineCouplingLayer(nn.Module):
    def __init__(self, in_features):
        super(AffineCouplingLayer, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(in_features // 2, in_features // 2),
            nn.ReLU(),  # Non-linearity for scale transformation
            nn.Linear(in_features // 2, in_features // 2)
        )
        self.translation_net = nn.Sequential(
            nn.Linear(in_features // 2, in_features // 2),
            nn.ReLU(),  # Non-linearity for translation transformation
            nn.Linear(in_features // 2, in_features // 2)
        )
    
    def forward(self, x):
        # Split the input x into two halves
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        # Compute scale and translation from the first half
        scale = self.scale_net(x1)
        translation = self.translation_net(x1)
        
        # Apply affine transformation to the second half
        y2 = x2 * torch.exp(scale) + translation
        
        # Concatenate the unchanged x1 and transformed y2
        return torch.cat([x1, y2], dim=-1), scale


# Normalizing Flow Model
class NormalizingFlow(nn.Module):
    def __init__(self, in_features, num_layers=3):
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList([AffineCouplingLayer(in_features) for _ in range(num_layers)])
        self.prior = torch.distributions.Normal(torch.zeros(in_features), torch.ones(in_features))
    
    def forward(self, z):
        log_det_jacobian = 0
        for layer in self.layers:
            z, scale = layer(z)
            log_det_jacobian += scale.sum(dim=-1)
        return z, log_det_jacobian

    def log_prob(self, z):
        self.prior.loc = self.prior.loc.to(z.device)
        self.prior.scale = self.prior.scale.to(z.device)
        z, log_det_jacobian = self.forward(z)
        log_prob_z = self.prior.log_prob(z).sum(dim=-1)
        return log_prob_z + log_det_jacobian

# Spatio-Temporal GCN with Transformer and Normalizing Flow
class GraphTransformerNormalizingFlow(nn.Module):
    def __init__(self, input_features,num_sensors, hidden_dim, output_dim, num_gcn_layers, num_transformer_layers, flow_layers, num_heads):
        super(GraphTransformerNormalizingFlow, self).__init__()
        
        # Transformer encoder for temporal processing before GCN
        encoder_layers = TransformerEncoderLayer(d_model=input_features*num_sensors, nhead=num_heads)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)
        
        # GCN for spatial processing
        self.gcn = GCNModel(input_features, hidden_dim, output_dim, num_gcn_layers)
        
        # Dimensionality reduction after concatenation
        self.fc_reduce = nn.Linear(output_dim +input_features, output_dim)  # Adjust input to match concatenated size
        
        # Normalizing flow for anomaly detection
        self.flow = NormalizingFlow(output_dim, num_layers=flow_layers)

    def forward(self, X, adj):
        B, W, N, F = X.shape
        
        # Transformer expects [seq_len, batch_size, feature_dim]
        X_permuted = X.permute(1, 0, 2, 3)
        
        X_transformed = self.transformer(X_permuted.view(W, B ,N * F))  # Temporal processing [W, B, N*F]
        X_transformed = X_transformed.view(W, B, N, F).permute(1, 0, 2, 3)  # [B, W, N, F]
        
        # GCN applied per time step with the corresponding adjacency matrix
        X_gcn = self.gcn(X, adj)  # GCN output [B, W, N, output_dim]
        
        # Concatenate GCN and Transformer outputs
        X_concat = torch.cat([X_transformed, X_gcn], dim=-1) 
        
        # Ensure X_concat has the right shape
        B, W, N, concatenated_dim = X_concat.shape
        assert concatenated_dim == self.fc_reduce.in_features, \
            f"Input to fc_reduce has {concatenated_dim} features, expected {self.fc_reduce.in_features}"
        
        # Dimensionality reduction
        X_reduced = self.fc_reduce(X_concat)  # [B, W, N, output_dim]
        
        # Flatten for flow input
        X_flattened = X_reduced.view(B * W * N, -1)  # [B*W*N, output_dim]
        
        # Pass through normalizing flow
        log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]
        
        # Reshape log_probs to [B, W, N]
        return -log_probs.view(B, W, N)
    

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class FixedGCNModel(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim, num_layers, dropout_rate=0.1):
        super(FixedGCNModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # First GCN layer
        self.layers.append(GCNConv(input_features, hidden_dim))
        
        # Hidden GCN layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            
        # Final GCN layer
        self.layers.append(GCNConv(hidden_dim, output_dim))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data_list):
        # Each item in data_list is a `Data` object containing x and edge_index
        output = []

        for data in data_list:
            x, edge_index = data.x, data.edge_index

            for layer in self.layers:
                x = layer(x, edge_index)
                x = self.dropout(torch.relu(x))  # Apply dropout and ReLU after each layer

            output.append(x)

        # Stack outputs for each time step if necessary
        return torch.stack(output, dim=0)  # [time_steps, nodes, output_dim]
 # Stack outputs for each time step

import numpy as np
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        
# Spatio-Temporal GCN with Transformer and Normalizing Flow with Dropout
class FixedGraphTransformerNormalizingFlow(nn.Module):
    def __init__(self, input_features, num_sensors, hidden_dim, output_dim, num_gcn_layers, num_transformer_layers, flow_layers,n_hidden_flow, num_heads, dropout_rate=0.1):
        super(FixedGraphTransformerNormalizingFlow, self).__init__()
        
        
        # Transformer encoder for temporal processing
        encoder_layers = TransformerEncoderLayer(d_model=input_features*num_sensors, nhead=num_heads, dropout=dropout_rate)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

        
        
        # GCN for spatial processing
        self.gcn = FixedGCNModel(input_features, hidden_dim, output_dim, num_gcn_layers, dropout_rate)
        
        # Dimensionality reduction after concatenation
        self.fc_reduce = nn.Sequential(
            nn.Linear(output_dim+input_features , output_dim),
            nn.Dropout(dropout_rate)  # Dropout after concatenation
        )
        
        # Normalizing flow for anomaly detection
        self.flow = RealNVP(n_blocks=flow_layers, input_size=output_dim, hidden_size=hidden_dim, n_hidden=n_hidden_flow, batch_norm=True)




    def forward(self, X):
        
        x_gcn = self.gcn(X)

        check_for_nan(x_gcn, "GCN Output")

        
        x_seq=torch.stack([i.x for i in X])

        x_seq = x_seq.unsqueeze(0)
        B, W, N, D = x_seq.shape
        x_seq = x_seq.permute( 1, 0, 2, 3)
        x_seq = x_seq.view(W,B,-1)
        x_seq=self.transformer(x_seq)
        x_seq = x_seq.view(W, B, N, D).permute(1, 0, 2, 3)
        x_seq = x_seq.squeeze(0)
        check_for_nan(x_seq, "Transformer Output")


        
        
        # B, W, N, F = X.shape
        
        # # Transformer expects [seq_len, batch_size, feature_dim]
        # X_permuted = X.permute(1, 0, 2, 3)
        
        # # Temporal processing [W, B, N*F]
        # X_transformed = self.transformer(X_permuted.view(W, B ,N * F))  
        # X_transformed = X_transformed.view(W, B, N, F).permute(1, 0, 2, 3)  # [B, W, N, F]
        
        # # GCN applied per time step with the corresponding adjacency matrix
        # X_gcn = self.gcn(X, adj)  # GCN output [B, W, N, output_dim]
        
        # # Concatenate GCN and Transformer outputs
        X_concat = torch.cat([x_gcn,x_seq], dim=-1) 
        
        # # Dimensionality reduction
        X_reduced = self.fc_reduce(X_concat)  # [B, W, N, output_dim]
        check_for_nan(X_reduced, "Dimensionality Reduction Output")


        W= X_reduced.shape[0]
        N= X_reduced.shape[1]
        
        
        # # Flatten for flow input
        
        X_flattened = X_reduced.view(W * N, -1)  # [B*W*N, output_dim]
       
        
        # Pass through normalizing flow
        log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]

        

        # Reshape log_probs to [B, W, N]
        return -log_probs.view(B, W, N)
    
        
# Spatio-Temporal GCN with Transformer and Normalizing Flow with Dropout
class TRACE_Transformer(nn.Module):
    def __init__(self, input_features, num_sensors, hidden_dim, output_dim, num_transformer_layers, flow_layers,n_hidden_flow, num_heads, dropout_rate=0.1):
        super(TRACE_Transformer, self).__init__()
        
        
        # Transformer encoder for temporal processing
        encoder_layers = TransformerEncoderLayer(d_model=input_features*num_sensors, nhead=num_heads, dropout=dropout_rate)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

        
        
        # GCN for spatial processing
        # self.gcn = FixedGCNModel(input_features, hidden_dim, output_dim, num_gcn_layers, dropout_rate)
        
        # Dimensionality reduction after concatenation
        self.fc_reduce = nn.Sequential(
            nn.Linear(input_features , output_dim),
            nn.Dropout(dropout_rate)  # Dropout after concatenation
        )
        
        # Normalizing flow for anomaly detection
        self.flow = RealNVP(n_blocks=flow_layers, input_size=output_dim, hidden_size=hidden_dim, n_hidden=n_hidden_flow, batch_norm=True)




    def forward(self, X):
        
        # x_gcn = self.gcn(X)

        # check_for_nan(x_gcn, "GCN Output")

        
        x_seq=torch.stack([i.x for i in X])

        x_seq = x_seq.unsqueeze(0)
        B, W, N, D = x_seq.shape
        x_seq = x_seq.permute( 1, 0, 2, 3)
        x_seq = x_seq.view(W,B,-1)
        x_seq=self.transformer(x_seq)
        x_seq = x_seq.view(W, B, N, D).permute(1, 0, 2, 3)
        x_seq = x_seq.squeeze(0)
        check_for_nan(x_seq, "Transformer Output")


        
        

        
        # # Dimensionality reduction
        X_reduced = self.fc_reduce(x_seq)  # [B, W, N, output_dim]
        check_for_nan(X_reduced, "Dimensionality Reduction Output")


        W= X_reduced.shape[0]
        N= X_reduced.shape[1]
        
        
        # # Flatten for flow input
        
        X_flattened = X_reduced.view(W * N, -1)  # [B*W*N, output_dim]
       
        
        # Pass through normalizing flow
        log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]

        

        # Reshape log_probs to [B, W, N]
        return -log_probs.view(B, W, N)

    
    import numpy as np
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
# Spatio-Temporal GCN with Transformer and Normalizing Flow with Dropout
# class TRACE(nn.Module):
#     def __init__(self, input_features, num_sensors, hidden_dim, output_dim, num_gcn_layers, num_transformer_layers, flow_layers,n_hidden_flow, num_heads, dropout_rate=0.1):
#         super(TRACE, self).__init__()
        
        
#         # Transformer encoder for temporal processing
#         encoder_layers = TransformerEncoderLayer(d_model=output_dim*num_sensors, nhead=num_heads, dropout=dropout_rate)
#         self.transformer = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

        
        
#         # GCN for spatial processing
#         self.gcn = FixedGCNModel(input_features, hidden_dim, output_dim, num_gcn_layers, dropout_rate)
        
#         # # Dimensionality reduction after concatenation
#         # self.fc_reduce = nn.Sequential(
#         #     nn.Linear(output_dim+input_features , output_dim),
#         #     nn.Dropout(dropout_rate)  # Dropout after concatenation
#         # )
        
#         # Normalizing flow for anomaly detection
#         self.flow = RealNVP(n_blocks=flow_layers, input_size=output_dim, hidden_size=hidden_dim, n_hidden=n_hidden_flow, batch_norm=True)




#     def forward(self, X):
        
#         x_gcn = self.gcn(X)

#         check_for_nan(x_gcn, "GCN Output")

        
#         # x_seq=torch.stack([i.x for i in X])

#         x_seq = x_gcn.unsqueeze(0)
#         B, W, N, D = x_seq.shape
#         x_seq = x_seq.permute( 1, 0, 2, 3)
#         x_seq = x_seq.view(W,B,-1)
#         x_seq=self.transformer(x_seq)
#         x_seq = x_seq.view(W, B, N, D).permute(1, 0, 2, 3)
#         x_seq = x_seq.squeeze(0)
#         check_for_nan(x_seq, "Transformer Output")


        
        
#         # B, W, N, F = X.shape
        
#         # # Transformer expects [seq_len, batch_size, feature_dim]
#         # X_permuted = X.permute(1, 0, 2, 3)
        
#         # # Temporal processing [W, B, N*F]
#         # X_transformed = self.transformer(X_permuted.view(W, B ,N * F))  
#         # X_transformed = X_transformed.view(W, B, N, F).permute(1, 0, 2, 3)  # [B, W, N, F]
        
#         # # GCN applied per time step with the corresponding adjacency matrix
#         # X_gcn = self.gcn(X, adj)  # GCN output [B, W, N, output_dim]
        
#         # # Concatenate GCN and Transformer outputs
#         # X_concat = torch.cat([x_gcn,x_seq], dim=-1) 
        
#         # # # Dimensionality reduction
#         # X_reduced = self.fc_reduce(X_concat)  # [B, W, N, output_dim]
#         # check_for_nan(X_reduced, "Dimensionality Reduction Output")


#         W= x_seq.shape[0]
#         N= x_seq.shape[1]
        
        
#         # # Flatten for flow input
        
#         X_flattened = x_seq.view(W * N, -1)  # [B*W*N, output_dim]
       
        
#         # Pass through normalizing flow
#         log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]

        

#         # Reshape log_probs to [B, W, N]
#         return -log_probs.view(B, W, N)
    
    
# Spatio-Temporal GCN with GRU and Normalizing Flow with Dropout
class TRACE_GCN(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim, num_gcn_layers, flow_layers, n_hidden_flow, dropout_rate=0.1):
        super(TRACE_GCN, self).__init__()
        
        # # GRU for temporal processing
        # self.gru = nn.GRU(input_features * num_sensors, input_features * num_sensors, num_layers=num_transformer_layers, batch_first=False, dropout=dropout_rate)
        
        # GCN for spatial processing
        self.gcn = FixedGCNModel(input_features, hidden_dim, output_dim, num_gcn_layers, dropout_rate)
        
        # # Dimensionality reduction after concatenation
        self.fc_reduce = nn.Sequential(
            nn.Linear(output_dim , output_dim),
            nn.Dropout(dropout_rate)  # Dropout after concatenation
        )
        
        # Normalizing flow for anomaly detection
        self.flow = RealNVP(n_blocks=flow_layers, input_size=output_dim, hidden_size=hidden_dim, n_hidden=n_hidden_flow, batch_norm=True)

    def forward(self, X):
        
        # GCN processing
        x_gcn = self.gcn(X)
        check_for_nan(x_gcn, "GCN Output")

        # # Temporal processing with GRU
        # x_seq = torch.stack([i.x for i in X])  # Stack each time step's features into a tensor
        # x_seq = x_seq.unsqueeze(0)
        # B, W, N, D = x_seq.shape
        # x_seq = x_seq.permute(1, 0, 2, 3).view(W, B, -1)  # Reshape for GRU input: [W, B, N*D]
        
        # x_seq, _ = self.gru(x_seq)  # GRU output
        # x_seq = x_seq.view(W, B, N, -1).permute(1, 0, 2, 3)  # Reshape back: [B, W, N, output_dim]
        # x_seq = x_seq.squeeze(0)
        # check_for_nan(x_seq, "GRU Output")

        # # Concatenate GCN and GRU outputs
        # X_concat = torch.cat([x_gcn, x_seq], dim=-1)
        
        # # Dimensionality reduction
        X_reduced = self.fc_reduce(x_gcn)  # [B, W, N, output_dim]
        check_for_nan(X_reduced, "Dimensionality Reduction Output")

        # Flatten for flow input
        B=1
        W, N = X_reduced.shape[0], X_reduced.shape[1]
        X_flattened = x_gcn.view(W * N, -1)  # [B*W*N, output_dim]
        
        # Pass through normalizing flow
        log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]
        
        # Reshape log_probs to [B, W, N]
        return -log_probs.view(B, W, N)
    
class TRACE_GCN(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim, num_gcn_layers, flow_layers, n_hidden_flow, dropout_rate=0.1):
        super(TRACE_GCN, self).__init__()
        
        # # GRU for temporal processing
        # self.gru = nn.GRU(input_features * num_sensors, input_features * num_sensors, num_layers=num_transformer_layers, batch_first=False, dropout=dropout_rate)
        
        # GCN for spatial processing
        self.gcn = FixedGCNModel(input_features, hidden_dim, output_dim, num_gcn_layers, dropout_rate)
        
        # # Dimensionality reduction after concatenation
        self.fc_reduce = nn.Sequential(
            nn.Linear(output_dim , output_dim),
            nn.Dropout(dropout_rate)  # Dropout after concatenation
        )
        
        # Normalizing flow for anomaly detection
        self.flow = RealNVP(n_blocks=flow_layers, input_size=output_dim, hidden_size=hidden_dim, n_hidden=n_hidden_flow, batch_norm=True)

    def forward(self, X):
        
        # GCN processing
        x_gcn = self.gcn(X)
        check_for_nan(x_gcn, "GCN Output")

        # # Temporal processing with GRU
        # x_seq = torch.stack([i.x for i in X])  # Stack each time step's features into a tensor
        # x_seq = x_seq.unsqueeze(0)
        # B, W, N, D = x_seq.shape
        # x_seq = x_seq.permute(1, 0, 2, 3).view(W, B, -1)  # Reshape for GRU input: [W, B, N*D]
        
        # x_seq, _ = self.gru(x_seq)  # GRU output
        # x_seq = x_seq.view(W, B, N, -1).permute(1, 0, 2, 3)  # Reshape back: [B, W, N, output_dim]
        # x_seq = x_seq.squeeze(0)
        # check_for_nan(x_seq, "GRU Output")

        # # Concatenate GCN and GRU outputs
        # X_concat = torch.cat([x_gcn, x_seq], dim=-1)
        
        # # Dimensionality reduction
        X_reduced = self.fc_reduce(x_gcn)  # [B, W, N, output_dim]
        check_for_nan(X_reduced, "Dimensionality Reduction Output")

        # Flatten for flow input
        B=1
        W, N = X_reduced.shape[0], X_reduced.shape[1]
        X_flattened = x_gcn.view(W * N, -1)  # [B*W*N, output_dim]
        
        # Pass through normalizing flow
        log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]
        
        # Reshape log_probs to [B, W, N]
        return -log_probs.view(B, W, N)
class TRACE_LSTM(nn.Module):
    def __init__(self, input_features, num_sensors, hidden_dim, output_dim, num_gcn_layers, num_transformer_layers, flow_layers, n_hidden_flow, dropout_rate=0.1):
        super(TRACE_LSTM, self).__init__()
        
        # GRU for temporal processing
        self.gru = nn.LSTM(input_features * num_sensors, input_features * num_sensors, num_layers=num_transformer_layers, batch_first=False, dropout=dropout_rate)
        
        # GCN for spatial processing
        self.gcn = FixedGCNModel(input_features, hidden_dim, output_dim, num_gcn_layers, dropout_rate)
        
        # Dimensionality reduction after concatenation
        self.fc_reduce = nn.Sequential(
            nn.Linear(input_features+output_dim , output_dim),
            nn.Dropout(dropout_rate)  # Dropout after concatenation
        )
        
        # Normalizing flow for anomaly detection
        self.flow = RealNVP(n_blocks=flow_layers, input_size=output_dim, hidden_size=hidden_dim, n_hidden=n_hidden_flow, batch_norm=True)

    def forward(self, X):
        
        # GCN processing
        x_gcn = self.gcn(X)
        check_for_nan(x_gcn, "GCN Output")

        # Temporal processing with GRU
        x_seq = torch.stack([i.x for i in X])  # Stack each time step's features into a tensor
        x_seq = x_seq.unsqueeze(0)
        B, W, N, D = x_seq.shape
        x_seq = x_seq.permute(1, 0, 2, 3).view(W, B, -1)  # Reshape for GRU input: [W, B, N*D]
        
        x_seq, _ = self.gru(x_seq)  # GRU output
        x_seq = x_seq.view(W, B, N, -1).permute(1, 0, 2, 3)  # Reshape back: [B, W, N, output_dim]
        x_seq = x_seq.squeeze(0)
        check_for_nan(x_seq, "GRU Output")

        # Concatenate GCN and GRU outputs
        X_concat = torch.cat([x_gcn, x_seq], dim=-1)
        
        # Dimensionality reduction
        X_reduced = self.fc_reduce(X_concat)  # [B, W, N, output_dim]
        check_for_nan(X_reduced, "Dimensionality Reduction Output")

        # Flatten for flow input
        W, N = X_reduced.shape[0], X_reduced.shape[1]
        X_flattened = X_reduced.view(W * N, -1)  # [B*W*N, output_dim]
        
        # Pass through normalizing flow
        log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]
        
        # Reshape log_probs to [B, W, N]
        return -log_probs.view(B, W, N)
    
class TRACE_GRU(nn.Module):
    def __init__(self, input_features, num_sensors, hidden_dim, output_dim, num_gcn_layers, num_transformer_layers, flow_layers, n_hidden_flow, dropout_rate=0.1):
        super(TRACE_GRU, self).__init__()
        
        # GRU for temporal processing
        self.gru = nn.GRU(input_features * num_sensors, input_features * num_sensors, num_layers=num_transformer_layers, batch_first=False, dropout=dropout_rate)
        
        # GCN for spatial processing
        self.gcn = FixedGCNModel(input_features, hidden_dim, output_dim, num_gcn_layers, dropout_rate)
        
        # Dimensionality reduction after concatenation
        self.fc_reduce = nn.Sequential(
            nn.Linear(input_features+output_dim , output_dim),
            nn.Dropout(dropout_rate)  # Dropout after concatenation
        )
        
        # Normalizing flow for anomaly detection
        self.flow = RealNVP(n_blocks=flow_layers, input_size=output_dim, hidden_size=hidden_dim, n_hidden=n_hidden_flow, batch_norm=True)

    def forward(self, X):
        
        # GCN processing
        x_gcn = self.gcn(X)
        check_for_nan(x_gcn, "GCN Output")

        # Temporal processing with GRU
        x_seq = torch.stack([i.x for i in X])  # Stack each time step's features into a tensor
        x_seq = x_seq.unsqueeze(0)
        B, W, N, D = x_seq.shape
        x_seq = x_seq.permute(1, 0, 2, 3).view(W, B, -1)  # Reshape for GRU input: [W, B, N*D]
        
        x_seq, _ = self.gru(x_seq)  # GRU output
        x_seq = x_seq.view(W, B, N, -1).permute(1, 0, 2, 3)  # Reshape back: [B, W, N, output_dim]
        x_seq = x_seq.squeeze(0)
        check_for_nan(x_seq, "GRU Output")

        # Concatenate GCN and GRU outputs
        X_concat = torch.cat([x_gcn, x_seq], dim=-1)
        
        # Dimensionality reduction
        X_reduced = self.fc_reduce(X_concat)  # [B, W, N, output_dim]
        check_for_nan(X_reduced, "Dimensionality Reduction Output")

        # Flatten for flow input
        W, N = X_reduced.shape[0], X_reduced.shape[1]
        X_flattened = X_reduced.view(W * N, -1)  # [B*W*N, output_dim]
        
        # Pass through normalizing flow
        log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]
        
        # Reshape log_probs to [B, W, N]
        return -log_probs.view(B, W, N)
    
    

    
from torch_geometric.data import Data    
class TRACE_TG(nn.Module):
    def __init__(self, input_features, num_sensors, hidden_dim, output_dim, num_gcn_layers, num_transformer_layers, flow_layers,n_hidden_flow, num_heads, dropout_rate=0.1):
        super(TRACE_TG, self).__init__()
        
        
        # Transformer encoder for temporal processing
        encoder_layers = TransformerEncoderLayer(d_model=input_features*num_sensors, nhead=num_heads, dropout=dropout_rate)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

        self.connecting_layer = nn.Sequential(
            nn.Linear(output_dim , input_features),
            nn.Dropout(dropout_rate)  # Dropout after concatenation
        )
        
        # GCN for spatial processing
        self.gcn = FixedGCNModel(input_features, hidden_dim, output_dim, num_gcn_layers, dropout_rate)
        
        # Dimensionality reduction after concatenation
        # self.fc_reduce = nn.Sequential(
        #     nn.Linear(input_features , output_dim),
        #     nn.Dropout(dropout_rate)  # Dropout after concatenation
        # )
        
        # Normalizing flow for anomaly detection
        self.flow = RealNVP(n_blocks=flow_layers, input_size=input_features, hidden_size=hidden_dim, n_hidden=n_hidden_flow, batch_norm=True)




    def forward(self, X):
        


        
        x_seq=torch.stack([i.x for i in X])

        x_seq = x_seq.unsqueeze(0)
        B, W, N, D = x_seq.shape
        x_seq = x_seq.permute( 1, 0, 2, 3)
        x_seq = x_seq.view(W,B,-1)
        x_seq=self.transformer(x_seq)
        x_seq = x_seq.view(W, B, N, D).permute(1, 0, 2, 3)
        x_seq = x_seq.squeeze(0)
        check_for_nan(x_seq, "Transformer Output")

        edge_index= X[0].edge_index
        data_list = []
        for i in range(x_seq.size(0)):  # Iterate over the first dimension (7 graphs)
            x = x_seq[i]  # Select features for the i-th graph
            data = Data(x=x, edge_index=edge_index)  # Create Data object
            data_list.append(data)
                
        x_seq=self.gcn(data_list)
        # print(x_seq.shape)
        x_seq=self.connecting_layer(x_seq)


        

        
        # # Dimensionality reduction
        # X_reduced = self.fc_reduce(x_seq)  # [B, W, N, output_dim]
        # check_for_nan(X_reduced, "Dimensionality Reduction Output")


        # W= x_seq.shape[0]
        # N= x_seq.shape[1]
        
        
        # # Flatten for flow input
        
        X_flattened = x_seq.view(W * N, -1)  # [B*W*N, output_dim]
       
        
        # Pass through normalizing flow
        log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]

        

        # Reshape log_probs to [B, W, N]
        return -log_probs.view(B, W, N)

import numpy as np
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
# Spatio-Temporal GCN with Transformer and Normalizing Flow with Dropout
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm




def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")

class BatchedGCNModel(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim, num_layers, dropout_rate=0.1):
        super(BatchedGCNModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # First GCN layer
        self.layers.append(GCNConv(input_features, hidden_dim))
        
        # Hidden GCN layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            
        # Final GCN layer
        self.layers.append(GCNConv(hidden_dim, output_dim))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sequences):
        # sequences: [B, W] each element is a Data object
        # We process each sequence separately for clarity
        # If you want to be more efficient, you can flatten and use Batch.from_data_list
        # as shown in comments below.

        # outputs = []
        # for seq in sequences:
        #     # seq is [Data, Data, ..., Data], length W
        #     seq_out = []
        #     for data in seq:
        #         x, edge_index = data.x, data.edge_index
                
        #         for i, layer in enumerate(self.layers):
        #             x = layer(x, edge_index)
        #             x = torch.relu(x)
        #             x = self.dropout(x)
        #         seq_out.append(x)  # [N, output_dim]

        #     # Stack time steps for one sequence
        #     seq_out = torch.stack(seq_out, dim=0)  # [W, N, output_dim]
        #     outputs.append(seq_out)

        # # Stack all sequences in the batch
        # return torch.stack(outputs, dim=0)  # [B, W, N, output_dim]

        # If you prefer a single batch approach (more efficient):
        all_data = [g for seq in sequences for g in seq]
        batch = Batch.from_data_list(all_data)
        x, edge_index = batch.x, batch.edge_index
        
        for layer in self.layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # x now corresponds to all graphs concatenated
        # Assuming each graph has N nodes, and each sequence is length W, and batch size B
        B = len(sequences)
        W = len(sequences[0])
        N = sequences[0][0].x.shape[0]
        
        # Reshape x back into [B, W, N, output_dim]
        x = x.view(B, W, N, -1)
        return x


class TRACE(nn.Module):
    def __init__(self, input_features, num_sensors, hidden_dim, output_dim, num_gcn_layers, num_transformer_layers, flow_layers, n_hidden_flow, num_heads, dropout_rate=0.1):
        super(TRACE, self).__init__()
        
        # Transformer encoder for temporal processing
        encoder_layers = TransformerEncoderLayer(d_model=input_features*num_sensors, nhead=num_heads, dropout=dropout_rate)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

        # GCN for spatial processing
        self.gcn = BatchedGCNModel(input_features, hidden_dim, output_dim, num_gcn_layers, dropout_rate)
        
        # Dimensionality reduction after concatenation
        self.fc_reduce = nn.Sequential(
            nn.Linear(output_dim+input_features, output_dim),
            nn.Dropout(dropout_rate)  # Dropout after concatenation
        )
        
        # Normalizing flow for anomaly detection
        self.flow = RealNVP(n_blocks=flow_layers, input_size=output_dim, hidden_size=hidden_dim, n_hidden=n_hidden_flow, batch_norm=True)

    def forward(self, batch_sequences):
        # batch_sequences: [B, W, Data], each Data has .x: [N, F], .edge_index: [2, E]

        # GCN output: [B, W, N, output_dim]
        x_gcn = self.gcn(batch_sequences)
        check_for_nan(x_gcn, "GCN Output")

        # Extract node features for Transformer
        # [B, W, N, F]
        x_seq_list = []
        for seq in batch_sequences:
            seq_x = torch.stack([g.x for g in seq], dim=0)  # [W, N, F]
            x_seq_list.append(seq_x)
        x_seq = torch.stack(x_seq_list, dim=0)  # [B, W, N, F]

        # Transformer expects [seq_len, batch_size, feature_dim]
        B, W, N, F = x_seq.shape
        x_seq = x_seq.permute(1, 0, 2, 3).contiguous().view(W, B, N*F)  # [W, B, N*F]
        x_seq = self.transformer(x_seq)  # [W, B, N*F]
        x_seq = x_seq.view(W, B, N, F).permute(1, 0, 2, 3).contiguous()  # [B, W, N, F]
        check_for_nan(x_seq, "Transformer Output")

        # Concatenate GCN and Transformer outputs along the feature dimension
        # x_gcn: [B, W, N, output_dim], x_seq: [B, W, N, F]
        X_concat = torch.cat([x_gcn, x_seq], dim=-1)  # [B, W, N, output_dim+F]

        # Dimensionality reduction
        X_reduced = self.fc_reduce(X_concat)  # [B, W, N, output_dim]
        check_for_nan(X_reduced, "Dimensionality Reduction Output")

        # Flatten for flow
        X_flattened = X_reduced.view(B * W * N, -1)  # [B*W*N, output_dim]

        # Pass through normalizing flow
        log_probs = self.flow.log_prob(X_flattened)  # [B*W*N]

        # Reshape log_probs to [B, W, N]
        return -log_probs.view(B, W, N)




