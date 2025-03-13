"""
We use the following prompt to generate comments on our code for further refinement.

I want your help to comment my code. Here are the guidelines:
1. Use Comments to Explain "Why," Not "What"
2. Avoid Redundant Comments
3. Write Concise and Clear Comments
4. Use Docstrings for Functions and Classes
5. Use Comments for Non-Obvious Solutions or Complex Logic
6. Mark TODOs and FIXMEs
7. Avoid Over-Commenting
8. Comment Sections of Code, Not Just Lines

Whenever I send you a code, send it back commented. Do not change any of the logic in the code.


"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time, asctime
import math
from glob import glob
from scipy import stats
import os

from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR


 np.random.seed(1234)

EVENT_VARS = ['h', 'e', 'x', 'y', 't', 'a']
CONTEXT_VARS = ['p', 'm', 's', 'hg', 'ag', 'hr', 'ar', 'hy', 'ay']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESTRICTIONS = {
    0: 2,
    1: 32,
    2: 101,
    3: 101,
    4: 101,
    5: 2,
    'h': 2,
    'e': 32,
    'x': 101,
    'y': 101,
    't': 101,
    'a': 2,
}

class LEMTokenizer:
    def __init__(self):
        # Initialize a vocabulary dictionary where keys and values are the same for numbers 0 to 100
        self.vocab = {i: i for i in range(0, 101)}
        # Add an unknown token '<UNK>' to handle out-of-vocabulary terms
        self.vocab['<UNK>'] = -1

        # List of predefined event types for encoding and decoding event actions
        self.event_types_list = [
            'pass', 'long_pass', 'cross', 'touch', 'aerial_duel', 'clearance', 'interception', 
            'loose_ball_duel', 'defensive_duel', 'offensive_duel', 'dribble', 'carry', 
            'game_interruption', 'own_goal', 'throw_in', 'free_kick', 'goal_kick', 'infraction', 
            'corner', 'acceleration', 'offside', 'right_foot_shot', 'left_foot_shot', 'head_shot', 
            'goalkeeper_exit', 'save', 'shot_against', 'fairplay', 'yellow_card', 'red_card', 
            'first_half_end', 'game_end'
        ]

        # Reverse vocabulary for decoding: maps event type indexes back to event names
        self.event_types_reverse_vocab = {}
        
        # Populate vocab with event types, mapping each type to a unique integer ID
        for i, event_type in enumerate(self.event_types_list):
            self.vocab[event_type] = i
            self.event_types_reverse_vocab[i] = event_type

        # Reverse vocab to allow decoding IDs back to their original tokens
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Define a constant for unknown token ID to handle unrecognized tokens in input
        self.UNK_TOKEN_ID = self.vocab['<UNK>']
    
    def encode_event_types(self, data_to_encode):
        """
        Encodes a sequence of event types into corresponding integer IDs using the vocabulary.

        Args:
            data_to_encode (Series): Data structure (e.g., pandas Series) of event types to encode.

        Returns:
            Series: A series where each event type is replaced with its corresponding integer ID.
        """
        return data_to_encode.map(self.vocab)
    
    def decode_event_types(self, data_to_decode):
        """
        Decodes a sequence of integer IDs back into their respective event type names.

        Args:
            data_to_decode (Series): Data structure (e.g., pandas Series) of integer IDs to decode.

        Returns:
            Series: A series where each integer ID is replaced with its corresponding event type name.
        """
        return data_to_decode.map(self.event_types_reverse_vocab)

TOKENIZER = LEMTokenizer()

class MLP(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, dropout_rate=0.25):
        super(MLP, self).__init__()

        # Store sequence length for reference, though it's not directly used in layer definitions
        self.seq_len = seq_len
        
        # Compute input size based on sequence length (helper function assumed)
        self.input_size = get_input_size(seq_len)


        # Initialize the first layer with input size and first hidden layer size
        layers = [
            nn.Linear(self.input_size, hidden_size[0]),
            nn.BatchNorm1d(hidden_size[0]),  # Batch normalization to stabilize training
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ]

        # Add hidden layers dynamically based on the hidden_size configuration
        for i in range(len(hidden_size) - 1):
            layers.extend([
                nn.Linear(hidden_size[i], hidden_size[i+1]),
                nn.BatchNorm1d(hidden_size[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        # Add the output layer with no activation (handled by separate method if needed)
        layers.extend([nn.Linear(hidden_size[-1], output_size)])

        # Create a sequential model to simplify forward pass through stacked layers
        self.model = nn.Sequential(*layers)
        
        # Sigmoid activation for output probabilities in predict_proba
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the model, applying each layer sequentially
        return self.model(x)
    
    def predict(self, x):
        """
        Predict class labels by taking the argmax of output logits.
        
        Args:
            x (DataFrame): Input data to predict on, compatible with sklearn.

        Returns:
            np.ndarray: Predicted class labels as integers.
        """
        # Use no_grad for inference to save memory and computation
        with torch.no_grad():
            # Convert input to a tensor and move it to the designated device
            x = torch.tensor(x.values, dtype=torch.float32).to(DEVICE)
            # Compute predictions and return as numpy array
            return torch.argmax(self.forward(x), dim=1).cpu().numpy()
        
    def predict_proba(self, x):
        """
        Predict class probabilities using sigmoid activation.
        
        Args:
            x (DataFrame): Input data to predict probabilities for, compatible with sklearn.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        # Use no_grad for inference to avoid tracking gradients
        with torch.no_grad():
            # Convert input to a tensor and move it to the designated device
            x = torch.tensor(x.values, dtype=torch.float32).to(DEVICE)
            # Apply sigmoid to output layer for probability values
            return self.sigmoid(self.forward(x)).cpu().numpy()

class Transformer(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, nhead=2, dropout_rate=0.1):
        super(Transformer, self).__init__()

        # Store sequence length for reference, though it’s not used directly in this module
        self.seq_len = seq_len
        
        # Compute input size based on sequence length (helper function assumed)
        input_size = get_input_size(seq_len)

        # Define shared activation layer to be reused
        activation = nn.ReLU()

        # Initialize the transformer encoder layer with specified attention heads and hidden size
        layers = [
            nn.TransformerEncoderLayer(
                d_model=input_size,          # Model dimension matching input size
                nhead=nhead,                 # Number of attention heads
                dim_feedforward=hidden_size[0],  # Size of feedforward network inside encoder layer
                activation=activation,       # Activation function for non-linearity
                dropout=dropout_rate         # Dropout rate for regularization
            ),
            nn.Dropout(dropout_rate),       # Additional dropout after the transformer encoder layer
            nn.Linear(input_size, hidden_size[0]),  # Linear transformation to first hidden size
            activation,
            nn.Dropout(dropout_rate)        # Dropout for regularization after linear layer
        ]

        # Add hidden layers dynamically based on the hidden_size configuration
        for i in range(len(hidden_size) - 1):
            layers.extend([
                nn.Linear(hidden_size[i], hidden_size[i+1]),  # Linear transformation
                activation,
                nn.Dropout(dropout_rate)       # Dropout for regularization after each hidden layer
            ])

        # Add final output layer without activation (activation handled separately if needed)
        layers.extend([
            nn.Linear(hidden_size[-1], output_size)
        ])

        # Create a sequential model to handle forward pass through all layers
        self.model = nn.Sequential(*layers)

        # Sigmoid activation for output probabilities in predict_proba
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the sequential model, applying all layers in order
        return self.model(x)
    
    def predict(self, x):
        """
        Predict class labels by taking the argmax of output logits.
        
        Args:
            x (DataFrame): Input data to predict on, compatible with sklearn.

        Returns:
            np.ndarray: Predicted class labels as integers.
        """
        # Inference with no_grad to save memory and computation
        with torch.no_grad():
            # Convert input to tensor and move to device for computation
            x = torch.tensor(x.values, dtype=torch.float32).to(DEVICE)
            # Compute predictions and return as numpy array
            return torch.argmax(self.forward(x), dim=1).cpu().numpy()
        
    def predict_proba(self, x):
        """
        Predict class probabilities using sigmoid activation.
        
        Args:
            x (DataFrame): Input data to predict probabilities for, compatible with sklearn.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        # Inference with no_grad to avoid tracking gradients
        with torch.no_grad():
            # Convert input to tensor and move to device for computation
            x = torch.tensor(x.values, dtype=torch.float32).to(DEVICE)
            # Apply sigmoid to output layer for probability values
            return self.sigmoid(self.forward(x)).cpu().numpy()

class FlatLSTM(nn.Module):
    def __init__(self, seq_len, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(FlatLSTM, self).__init__()

        # Store sequence length for potential reference
        self.seq_len = seq_len
        # Determine input size based on sequence length (helper function assumed)
        input_size = get_input_size(seq_len)

        # Store LSTM hidden dimensions and layer count
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Linear layer to project inputs to match LSTM's hidden dimension
        self.input_projection = nn.Linear(input_size, hidden_dim)
        # Dropout for regularization after the input projection
        self.input_dropout = nn.Dropout(dropout)

        # LSTM layer with specified number of layers and dropout
        self.lstm = nn.LSTM(
            hidden_dim,          # Hidden dimension size
            hidden_dim,          # Output dimension matches hidden for subsequent processing
            num_layers,          # Number of LSTM layers
            batch_first=True,    # Ensures batch dimension is first
            dropout=dropout      # Dropout applied between LSTM layers
        )
        
        # Dropout layer after LSTM to avoid overfitting
        self.lstm_dropout = nn.Dropout(dropout)

        # Fully connected layer to project the LSTM output to the number of classes
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Sigmoid activation for output probabilities in predict_proba
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Project input features to match hidden dimension of LSTM and add batch dimension
        x = self.input_projection(x).unsqueeze(1)
        x = self.input_dropout(x)  # Apply dropout after projection

        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Pass through LSTM; out holds outputs for each timestep, hn/cn are hidden and cell states
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.lstm_dropout(out)  # Apply dropout to LSTM outputs

        # Apply final fully connected layer on the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        """
        Predict class labels by taking the argmax of output probabilities.
        
        Args:
            x (DataFrame): Input data for prediction, compatible with sklearn.

        Returns:
            np.ndarray: Predicted class labels as integers.
        """
        # Inference with no_grad to save memory and computation
        with torch.no_grad():
            # Convert input to tensor and move to designated device
            x = torch.tensor(x.values, dtype=torch.float32).to(DEVICE)
            # Compute predictions and return as numpy array
            return torch.argmax(self.forward(x), dim=1).cpu().numpy()
        
    def predict_proba(self, x):
        """
        Predict class probabilities using sigmoid activation.
        
        Args:
            x (DataFrame): Input data for predicting probabilities, compatible with sklearn.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        # Inference with no_grad to avoid tracking gradients
        with torch.no_grad():
            # Convert input to tensor and move to device for computation
            x = torch.tensor(x.values, dtype=torch.float32).to(DEVICE)
            # Apply sigmoid to final layer output for probability values
            return self.sigmoid(self.forward(x)).cpu().numpy()

def arrange_data_for_tabular(fname, seq_len, n_files_save):
    data = pd.read_feather(fname)

    # Tokenize the event types
    data['e'] = TOKENIZER.encode_event_types(data['e'])

    # Add context
    for i in range(1, seq_len + 1):
        data_context_1 = data.shift(i).fillna(TOKENIZER.UNK_TOKEN_ID)
        data_context_1.loc[data_context_1['match_id'] != data['match_id'], EVENT_VARS] = TOKENIZER.UNK_TOKEN_ID
        data_context_1 = data_context_1[EVENT_VARS].add_prefix(f'c{i}_').astype(np.int8)
        data = pd.concat([data, data_context_1], axis=1)

    data_lst = []
    for i, f in enumerate(EVENT_VARS):
        edit_data = data.copy()
        edit_data['target'] = edit_data[f].clip(0, 100)
        edit_data[EVENT_VARS[i:]] = TOKENIZER.UNK_TOKEN_ID
        edit_data = edit_data.drop(columns=['match_id'])
        data_lst.append(edit_data)
    
    data = pd.concat(data_lst)
    data = data.astype(np.int8)
    data = data.sample(frac=1, random_state=42)

    for i in range(n_files_save):
        data.iloc[i*len(data)//n_files_save:(i+1)*len(data)//n_files_save].to_feather(fname.replace('/raw_lem/', '/tabular_lem/').split('.')[0] + f'_sq{seq_len}_rs42_{i}.feather')

def arrange_data_for_time_series(fname, seq_len, n_files_save):
    df = pd.read_feather(fname)
    # df.columns = ['match_id'] + EVENT_VARS + CONTEXT_VARS

    # Encoding event types
    df['e'] = TOKENIZER.encode_event_types(df['e'])

    data_events = []
    data_contexts = []
    for match_id in tqdm(df['match_id'].unique()):
        match_data = df[df.match_id == match_id]
        data_events.append(match_data[EVENT_VARS].values.ravel())
        data_contexts.append(match_data[CONTEXT_VARS].values.ravel())
    data_events = pd.DataFrame(data_events)
    data_contexts = pd.DataFrame(data_contexts)

    data = []
    for i in range(n_files_save):
        open(f'temp/arrange_data_as_time_series_{i}.csv', 'w').close()
    for i in tqdm(range(int(data_events.shape[1]))):
        event_no = i // len(EVENT_VARS)
        event_var_id = i % len(EVENT_VARS)

        series = data_events.iloc[:, max(0, i-(len(EVENT_VARS) * seq_len)):i].clip(lower=0, upper=100)
        if series.shape[1] < (len(EVENT_VARS) * seq_len):
            series = pd.concat([pd.DataFrame([[TOKENIZER.UNK_TOKEN_ID]*(len(EVENT_VARS)*seq_len - series.shape[1])]*data_events.shape[0]), series], axis=1)
        series.columns = [f'i{seq_len*len(EVENT_VARS) - j}' for j in range(series.shape[1])]

        target = data_events.iloc[:, i].rename('target').clip(lower=0, upper=100)

        context = data_contexts.iloc[:, max(0, event_no * len(CONTEXT_VARS) - len(CONTEXT_VARS)):event_no * len(CONTEXT_VARS)].clip(lower=0, upper=100)
        if context.shape[1] < len(CONTEXT_VARS):
            context = pd.DataFrame([[0]*len(CONTEXT_VARS)]*data_events.shape[0])
        context.columns = [f'c{j}' for j in range(len(CONTEXT_VARS))]

        data = pd.concat([context, series, target], axis=1)
        data['event_var_id'] = event_var_id

        data = data.dropna()
        data = data.sample(frac=1).reset_index(drop=True)

        if data.shape[0] > 0:
            for i in range(n_files_save):
                data[i*(data.shape[0]//n_files_save):(i+1)*(data.shape[0]//n_files_save)].to_csv(f'temp/arrange_data_as_time_series_{i}.csv', mode='a', header=False, index=False)

    col_names = data.columns.tolist()
    
    for i in range(n_files_save):
        data = pd.read_csv(f'temp/arrange_data_as_time_series_{i}.csv', names=col_names)
        data = data.astype(np.int8)
        data.to_feather(fname.replace('/raw_lem/', '/time_series_lem/').split('.')[0] + f'_sq{seq_len}_rs42_{i}.feather')
        os.remove(f'temp/arrange_data_as_time_series_{i}.csv')

def write_to_tracker(fname, line):
    if os.path.exists(fname):
        tracker_f = open(fname, "a")
    else:
        tracker_f = open(fname, "w")

    tracker_f.write(line)
    tracker_f.close()




def get_test_report(data, model, pred_seq=None, tokenizer=None):
    if pred_seq is None:
        pred_seq = EVENT_VARS

    if tokenizer is None:
        tokenizer = LEMTokenizer()

    metrics_data = []
    for i, f in tqdm(enumerate(pred_seq), total=len(pred_seq)):
        if i == 0:
            edit_data = data[data[pred_seq[0]] == -1].copy()
        else:
            edit_data = data[(data[pred_seq[i]] == -1) & (data[pred_seq[i-1]] != -1)].copy()
        target = edit_data['target'].copy()
        edit_data = edit_data.drop(columns=['target'])
        # edit_data[pred_seq[i:]] = tokenizer.UNK_TOKEN_ID
        t0 = time()
        pred = model.predict(edit_data)
        t1 = time()

        metrics_data.append((
            f,
            accuracy_score(target, pred), 
            f1_score(target, pred, average="weighted"), 
            r2_score(target, pred), 
            mean_absolute_error(target, pred),
            t1 - t0, 
            ))
        
    return pd.DataFrame(metrics_data, columns=['target_var', 'accuracy', 'f1', 'r2', 'mae', 'inf_time'])


def simulate_event(model, X, pred_seq=None, tokenizer=None):
    if tokenizer is None:
        tokenizer = LEMTokenizer()
    if pred_seq is None:
        pred_seq = EVENT_VARS

    for i, stage in enumerate(pred_seq):
        proba = model.forward(X)
        proba = proba.data[:, :RESTRICTIONS[i]]
        proba = model.sigmoid(proba)
        sampled = torch.multinomial(proba, 1)
        X[:, i] = sampled.squeeze()

    return X[:, :6]

    
def simulate_game(model, context_X, max_sims=2500, tokenizer=None, pred_seq=None, save_sim_fname=None, tqdm_disable=False, return_unfinished=False, return_type='results'):
    if tokenizer is None:
        tokenizer = LEMTokenizer()
    if pred_seq is None:
        pred_seq = EVENT_VARS

    registering_enabled = type(save_sim_fname) == str

    X = context_X.clone()

    restrictions = RESTRICTIONS

    previous_events_start = 15
    size_previous_event = 6
    p, m, s = 6, 7, 8
    context_home_goals = 9
    context_away_goals = 10
    context_reds_home = 11
    context_reds_away = 12
    context_yellows_home = 13
    context_yellows_away = 14
    h, e, x, y, t, a = 0, 1, 2, 3, 4, 5

    goals_delta = torch.zeros(X.shape[0], dtype=torch.float32).to(DEVICE)
    goals_home = torch.zeros(X.shape[0], dtype=torch.float32).to(DEVICE)
    goals_away = torch.zeros(X.shape[0], dtype=torch.float32).to(DEVICE)
    indexes = torch.arange(X.shape[0], dtype=torch.int64).to(DEVICE)
    if return_type == 'results+inspect':
        inspect_e = []
        inspect_x = []
        inspect_y = []
        inspect_t = []
        inspect_uncertainty = [[], [], [], [], [], []]
        inspect_xg = [[], []]
        inspect_shots = [[], []]

    # time everything to check where we can make gains TODO
    for n in tqdm(range(max_sims), disable=tqdm_disable):
        save_data = []
        for i, stage in enumerate(pred_seq):
            proba = model.forward(X)
            torch.cuda.synchronize()


            proba = proba.data[:, :restrictions[i]]
            proba = model.sigmoid(proba)
            # forced game over/first half end
            if i == e:
                proba[:, 30] += (X[:, m] > 52) * (X[:, p] == 0)
                proba[:, 31] += X[:, m] > 97

                proba[:, 30] -= proba[:, 30] * (X[:, m] < 45)
                proba[:, 30] -= proba[:, 30] * (X[:, p] > 0)
                proba[:, 31] -= proba[:, 31] * (X[:, m] < 90)
            sampled = torch.multinomial(proba, 1)

            X[:, i] = sampled.squeeze()

            if return_type == 'results+inspect':
                inspect_uncertainty[i].append(proba.max(axis=1).values.cpu().numpy())

            if registering_enabled:
                if i == h:
                    save_data.append(proba[:, 1].cpu().numpy().reshape(-1, 1))
                elif i == e:
                    save_data.append(proba.cpu().numpy())
                elif i == a:
                    save_data.append(proba[:, 1].cpu().numpy().reshape(-1, 1))
        
        if registering_enabled:
            save_data.append(indexes.cpu().numpy().reshape(-1, 1))
            save_data.append(X[:, :6].cpu().numpy())
            save_data.append(X[:, previous_events_start + 4:previous_events_start + 7].cpu().numpy())
            save_data = np.concatenate(save_data, axis=1)
            pd.DataFrame(
                save_data, 
                columns=['token_h'] + [f'token_e_{i}' for i in range(32)] + ['token_a'] + ['index', 'is_home_team', 'type_primary', 'x', 'y', 'time_elapsed', 'success'] + ['period', 'minute', 'second']).to_csv(save_sim_fname, mode='a', header=False, index=False)
            
        if return_type == 'results+inspect':
            inspect_e.append(X[:, e].cpu().numpy())
            inspect_x.append(X[:, x].cpu().numpy())
            inspect_y.append(X[:, y].cpu().numpy())
            inspect_t.append(X[:, t].cpu().numpy())

            inspect_shots[0].append((((X[:, e] == 21) + (X[:, e] == 22) + (X[:, e] == 23)) * (X[:, h] == 1)).sum().cpu().numpy())
            inspect_shots[1].append((((X[:, e] == 21) + (X[:, e] == 22) + (X[:, e] == 23)) * (X[:, h] == 0)).sum().cpu().numpy())

            inspect_xg[0].append(proba[((X[:, e] == 21) + (X[:, e] == 22) + (X[:, e] == 23)) * (X[:, h] == 1), 1].cpu().numpy())
            inspect_xg[1].append(proba[((X[:, e] == 21) + (X[:, e] == 22) + (X[:, e] == 23)) * (X[:, h] == 0), 1].cpu().numpy())
        
        # Update the context
        torch.cuda.synchronize()
        # print(previous_events_start,  previous_events_start + size_previous_event * (model.seq_len - 1), previous_events_start + size_previous_event, previous_events_start + size_previous_event * (model.seq_len))
        tmp = X[:, previous_events_start: previous_events_start + size_previous_event * (model.seq_len - 1)].clone()
        X[:, previous_events_start + size_previous_event: previous_events_start + size_previous_event * (model.seq_len)] = tmp
        # print(X[0, 0: previous_events_start + size_previous_event])
        # print(X[:, e])
        # print(X[:, m])


        X[:, previous_events_start] = X[:, h]
        X[:, previous_events_start + e] = X[:, e]
        X[:, previous_events_start + x] = X[:, x]
        X[:, previous_events_start + y] = X[:, y]
        X[:, previous_events_start + t] = X[:, t]
        X[:, previous_events_start + a] = X[:, a]
        X[:, p] += (X[:, e] == 30) + (X[:, e] == 31)
        X[:, m] += (X[:, s] + X[:, t]) >= 60
        X[:, s] = (X[:, s] + X[:, t]) % 60

        X[:, m] = X[:, m] * (X[:, e] != 30) + 45 * (X[:, e] == 30)
        X[:, s] = X[:, s] * (X[:, e] != 30)

        # Updating goals
        X[:, context_home_goals] += ((X[:, e] == 21) + (X[:, e] == 22) + (X[:, e] == 23)) * (X[:, h] == 1) * (X[:, a] == 1)
        X[:, context_away_goals] += ((X[:, e] == 21) + (X[:, e] == 22) + (X[:, e] == 23)) * (X[:, h] == 0) * (X[:, a] == 1)
        # Updating cards
        X[:, context_yellows_home] += (X[:, e] == 28) * (X[:, h] == 1)
        X[:, context_yellows_away] += (X[:, e] == 28) * (X[:, h] == 0)
        X[:, context_reds_home] += (X[:, e] == 29) * (X[:, h] == 1)
        X[:, context_reds_away] += (X[:, e] == 29) * (X[:, h] == 0)

        # Save rows where the game is over
        goals_delta[indexes[X[:, e] == 31]] = X[X[:, e] == 31][:, context_home_goals] - X[X[:, e] == 31][:, context_away_goals]
        goals_home[indexes[X[:, e] == 31]] = X[X[:, e] == 31][:, context_home_goals]
        goals_away[indexes[X[:, e] == 31]] = X[X[:, e] == 31][:, context_away_goals]
        
        # Remove rows where the game is over
        indexes = indexes[X[:, e] != 31]
        X = X[X[:, e] != 31]
        
        # If all games are over, break
        if X.shape[0] == 0:
            break

        # Reset target tokens
        X[:, :6] = tokenizer.UNK_TOKEN_ID

    if return_unfinished:
        goals_delta[indexes] = X[:, context_home_goals] - X[:, context_away_goals]
        goals_home[indexes] = X[:, context_home_goals]
        goals_away[indexes] = X[:, context_away_goals]

    if return_type == 'results':
        return goals_delta.cpu().numpy(), goals_home.cpu().numpy(), goals_away.cpu().numpy(), n
    elif return_type == 'state':
        return X[:, previous_events_start: previous_events_start + size_previous_event].cpu().numpy()
    elif return_type == 'results+inspect':
        return goals_delta.cpu().numpy(), goals_home.cpu().numpy(), goals_away.cpu().numpy(), n, inspect_e, inspect_x, inspect_y, inspect_t, inspect_uncertainty, inspect_shots, inspect_xg

def get_fidelity_report(model, val_data, n_event_sims=50000, tokenizer=False, save_sim_fname=None):
    if tokenizer is False:
        tokenizer = LEMTokenizer()

    # Simulate the match
    base_start_game_tensor = torch.Tensor([
        tokenizer.UNK_TOKEN_ID, tokenizer.UNK_TOKEN_ID, tokenizer.UNK_TOKEN_ID, tokenizer.UNK_TOKEN_ID, tokenizer.UNK_TOKEN_ID, tokenizer.UNK_TOKEN_ID,
        0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0] + [1, 0, 50, 50, 0, 0, 1, 1] + [tokenizer.UNK_TOKEN_ID] * 8 * (model.seq_len - 1))
    context_X_base_sims = base_start_game_tensor.repeat(n_event_sims, 1).to(DEVICE)
    
    t0 = time()
    res_goals_delta, res_goals_home, res_goals_away, n_sims = simulate_full_game(model, context_X_base_sims, save_sim_fname=save_sim_fname)
    sim_time = time() - t0

    # Get the target distribution
    target_hist_home_goals = (val_data.groupby('match_id')['context_goals_home'].max().clip(-5, 5).value_counts().sort_index() / val_data.groupby('match_id')['context_goals_home'].max().value_counts().sum())
    target_hist_away_goals = (val_data.groupby('match_id')['context_goals_away'].max().clip(-5, 5).value_counts().sort_index() / val_data.groupby('match_id')['context_goals_away'].max().value_counts().sum())
    target_hist_goals_delta = val_data.groupby('match_id')['context_goals_home'].max() - val_data.groupby('match_id')['context_goals_away'].max()
    target_hist_goals_delta = (target_hist_goals_delta.clip(-5, 5).value_counts().sort_index() / target_hist_goals_delta.value_counts().sum())

    # Get the simulated distribution
    goal_delta_dist = pd.Series(res_goals_delta).clip(lower=target_hist_goals_delta.index.min(), upper=target_hist_goals_delta.index.max()).value_counts() / n_event_sims
    for val in target_hist_goals_delta.index:
        if val not in goal_delta_dist.index:
            goal_delta_dist[val] = 0
    goal_delta_dist = goal_delta_dist.sort_index()

    home_goals_dist = pd.Series(res_goals_home).clip(lower=target_hist_home_goals.index.min(), upper=target_hist_home_goals.index.max()).value_counts() / n_event_sims
    for val in target_hist_home_goals.index:
        if val not in home_goals_dist.index:
            home_goals_dist[val] = 0
    home_goals_dist = home_goals_dist.sort_index()

    away_goals_dist = pd.Series(res_goals_away).clip(lower=target_hist_away_goals.index.min(), upper=target_hist_away_goals.index.max()).value_counts() / n_event_sims
    for val in target_hist_away_goals.index:
        if val not in away_goals_dist.index:
            away_goals_dist[val] = 0
    away_goals_dist = away_goals_dist.sort_index()

    # Return the results
    return {
        'kl_goals_delta': stats.entropy(target_hist_goals_delta, goal_delta_dist), 
        'kl_goals_home': stats.entropy(target_hist_home_goals, home_goals_dist), 
        'kl_goals_away': stats.entropy(target_hist_away_goals, away_goals_dist), 
        'n_sims': n_sims, 
        'sim_time': sim_time
    }





# Utils

def flatten(l):
    return [item for sublist in l for item in sublist]

def reset_cuda(model, optimizer):
    model.cpu()
    del model
    del optimizer
    torch.cuda.empty_cache()

def get_input_size(n_tokens):
    return 6 + 9 + 6 * n_tokens












