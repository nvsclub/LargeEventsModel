import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import lib.draw as draw

COLORS = ['white', '#eae2b7', '#fcbf49', '#f77f00', '#d62828', '#003049']
BOUNDARIES = [-1, 0.001, 0.10, 0.25, 0.5, 0.75, 1]
CMAP = mcolors.ListedColormap(COLORS)
NORM = mcolors.BoundaryNorm(BOUNDARIES, CMAP.N, clip=True)
ROSE = '#ff99ac'

def get_restrictions(model_k):
    restrictions = {
        (11 * model_k): [i for i in range(101, 139)],
        (11 * model_k) + 1: [0, 1],
        (11 * model_k) + 2: [0, 1],
        (11 * model_k) + 3: [0, 1],
        (11 * model_k) + 4: [i for i in range(0, 101)],
        (11 * model_k) + 5: [i for i in range(0, 101)],
        (11 * model_k) + 6: [i for i in range(0, 101)],
        }

    offset_restrictions = {
        (11 * model_k): 101,
        (11 * model_k) + 1: 0,
        (11 * model_k) + 2: 0,
        (11 * model_k) + 3: 0,
        (11 * model_k) + 4: 0,
        (11 * model_k) + 5: 0,
        (11 * model_k) + 6: 0,
        }
    
    return restrictions, offset_restrictions

def get_loc_ids(model_k):
    return {
        'tgt_event': model_k * 11,
        'tgt_is_goal': model_k * 11 + 1,
        'tgt_is_accurate': model_k * 11 + 2,
        'tgt_is_home': model_k * 11 + 3,
        'tgt_time': model_k * 11 + 4,
        'tgt_x': model_k * 11 + 5,
        'tgt_y': model_k * 11 + 6,

        'prev_event': (model_k-1) * 11,
        'prev_is_goal': (model_k-1) * 11 + 1,
        'prev_is_accurate': (model_k-1) * 11 + 2,
        'prev_is_home': (model_k-1) * 11 + 3,
        'prev_period': (model_k-1) * 11 + 4,
        'prev_minute': (model_k-1) * 11 + 5,
        'prev_second': (model_k-1) * 11 + 6,
        'prev_x': (model_k-1) * 11 + 7,
        'prev_y': (model_k-1) * 11 + 8,
        'prev_home_score': (model_k-1) * 11 + 9,
        'prev_away_score': (model_k-1) * 11 + 10,
    }

def get_tokenizer_map(dir='models/llm/tokenizer_map.json'):
    tokenizer_map = json.load(open('models/llm/tokenizer_map.json', 'r'))
    detokenizer_map = {v: k for k, v in tokenizer_map.items()}
    return tokenizer_map, detokenizer_map

def simulate_from_dataframe(forecast_df, model, model_k=1, max_events=2000, n_sims_per_row=1, DEVICE='cuda:0', end_on_ht=False, disable_tqdm=True):
    RES, OFF_RES = get_restrictions(model_k)
    LID = get_loc_ids(model_k)


    indexes = torch.tensor(np.concatenate([forecast_df.index] * n_sims_per_row), dtype=torch.float32).to(DEVICE)
    X_train = torch.tensor(np.concatenate([forecast_df.values] * n_sims_per_row), dtype=torch.float32).to(DEVICE)

    savepoints = []
    savepoint_indexes = []
    for i in tqdm(range(max_events * 7), disable=disable_tqdm):
        if X_train.shape[0] == 0:
            break

        token_id = 11 + (i%7)
        pred = model(X_train)
        
        if (i % 7) == 0:
            # TODO: limitation: Game Over token is not well estimated.
            pred[:, RES[token_id][-1]] += 0.01 * (X_train[:,LID['prev_minute']] - 45)
            pred[:, RES[token_id][-2]] += 0.01 * (X_train[:,LID['prev_minute']] - 45)
            # Add restrictions for <GAME_OVER> and <PERIOD_OVER>
            pred[:, RES[token_id][-2]] *= (X_train[:,LID['prev_minute']] > 45) * (X_train[:,LID['prev_period']] == 1)
            pred[:, RES[token_id][-1]] *= (X_train[:,LID['prev_minute']] > 45) * (X_train[:,LID['prev_period']] == 2)

        # Kill the sims that have gone rogue
        mask = (pred[:, RES[token_id]].sum(dim=1) == 0)
        if mask.sum():
            indexes = indexes[~mask]
            X_train = X_train[~mask]
            pred = pred[~mask]

        pred = torch.multinomial(pred[:, RES[token_id]], 1) + OFF_RES[token_id]
        X_train[:, token_id] = torch.squeeze(pred)

        if (i % 7) == 0:
            # Checking end of game
            if (X_train[:, LID['tgt_event']] == 138).sum():
                mask = (X_train[:, LID['tgt_event']] == 138)
                savepoint_indexes.append(indexes[mask])
                savepoints.append(X_train[mask,:-7])
                indexes = indexes[~mask]
                X_train = X_train[~mask]
            
            # Checking end of period
            if end_on_ht:
                if (X_train[:, LID['tgt_event']] == 137).sum():
                    mask = (X_train[:, LID['tgt_event']] == 137)
                    savepoint_indexes.append(indexes[mask])
                    savepoints.append(X_train[mask,:-7])
                    indexes = indexes[~mask]
                    X_train = X_train[~mask]

            else:
                if (X_train[:, LID['tgt_event']] == 137).sum():
                    mask = (X_train[:, LID['tgt_event']] == 137)
                    X_train[mask, LID['prev_event']] = 101
                    X_train[mask, LID['prev_is_goal']] = 0
                    X_train[mask, LID['prev_is_accurate']] = 1
                    X_train[mask, LID['prev_is_home']] = random.randint(0,1)
                    X_train[mask, LID['prev_period']] = 2
                    X_train[mask, LID['prev_minute']] = 0
                    X_train[mask, LID['prev_second']] = 1
                    X_train[mask, LID['prev_x']] = 50
                    X_train[mask, LID['prev_y']] = 50
                    continue

        # Checking event complete
        if (i % 7) == 6:
            X_train[:, LID['prev_event']: LID['prev_period']] = X_train[:, LID['tgt_event']: LID['tgt_time']].clone()
            #X_train[:, LID['prev_period']] += (X_train[:, 11] == 137)
            X_train[:, LID['prev_minute']] = ((X_train[:, LID['prev_minute']] * 60 + X_train[:, LID['prev_second']] + X_train[:, LID['tgt_time']]) / 60).floor()
            X_train[:, LID['prev_second']] = ((X_train[:, LID['prev_second']] + X_train[:, LID['tgt_time']]) % 60)
            X_train[:, LID['prev_x']] = X_train[:, LID['tgt_x']]
            X_train[:, LID['prev_y']] = X_train[:, LID['tgt_y']]
            # Penalty + Shot + Free kick shot
            X_train[:, LID['prev_home_score']] += (X_train[:, LID['prev_is_home']] == 1) * (X_train[:, LID['prev_is_goal']] == 1) * (
                (X_train[:, LID['prev_event']] == 130) + (X_train[:, LID['prev_event']] == 116) + (X_train[:, LID['prev_event']] == 128))
            X_train[:, LID['prev_away_score']] += (X_train[:, LID['prev_is_home']] == 0) * (X_train[:, LID['prev_is_goal']] == 1) * (
                (X_train[:, LID['prev_event']] == 130) + (X_train[:, LID['prev_event']] == 116) + (X_train[:, LID['prev_event']] == 128))
            X_train[:,-7:] = 138

    
    savepoint_indexes.append(indexes)
    savepoints.append(X_train[:,:-7])

    savepoint_indexes = torch.cat(savepoint_indexes, dim=0).cpu().numpy()
    savepoints = torch.cat(savepoints, dim=0).cpu().numpy()

    return pd.DataFrame(savepoints, index=savepoint_indexes).sort_index()

def flatten(l):
    return [item for sublist in l for item in sublist]

class MultiLayerBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        super(MultiLayerBinaryClassifier, self).__init__()

        activation_dict = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
        }
        layers = [
            nn.Linear(input_size, hidden_size[0]),
            activation_dict[activation]()
        ] + flatten([
            [nn.Linear(hidden_size[i], hidden_size[i+1]),
            activation_dict[activation]()] for i in range(len(hidden_size) - 1)
        ]) + [
            nn.Linear(hidden_size[-1], output_size),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*layers)
        
        # Initialize the linear layers
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)
    
def draw_xg_map(base_state, model, k=1, n_sims=1000000, DEVICE='cuda:0'):
    restrictions, offset_restrictions = get_restrictions(k)
    tokenizer_map, detokenizer_map = get_tokenizer_map()

    xg_df = []
    for _ in range(n_sims):
        xg_df.append(base_state + ['shot', '<NaN>', '<NaN>', '<NaN>', '<NaN>', '<NaN>', '<NaN>'])
    xg_df = pd.DataFrame(xg_df)
    for i in range(0, 11*k + 7):
        xg_df[i] = xg_df[i].map(tokenizer_map)

    X_train = xg_df.astype(float).values
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

    for i in range(11*k + 1, 11*k + 7):
        pred = model(X_train_tensor)
        if i == (11*k + 1):
            xg = pred[:, restrictions[i]]
        pred = torch.multinomial(pred[:, restrictions[i]], 1) + offset_restrictions[i]
        X_train_tensor[:, i] = torch.squeeze(pred)

    xg_df = pd.DataFrame(X_train_tensor.cpu().detach().numpy())
    for i in range(0, 11*k + 7):
        xg_df[i] = xg_df[i].map(detokenizer_map)
    xg_df['xg'] = xg[:, 0].cpu().detach().numpy()
    xg_df[[11*k + 1, 11*k + 5, 11*k + 6]] = xg_df[[11*k + 1, 11*k + 5, 11*k + 6]].astype(int)

    draw.pitch()
    xg_map = (xg_df.groupby([11*k + 5, 11*k + 6])[11*k + 1].mean().clip(0, 0.2) * 5).reset_index()
    xg_map = xg_map.pivot_table(index=11*k + 6, columns=11*k + 5, values=11*k + 1, fill_value=0)
    for i in range(0, 101):
        if i not in xg_map.columns:
            xg_map[i] = 0
        if i not in xg_map.index:
            xg_map.loc[i] = 0
    xg_map = xg_map.sort_index(axis=0).sort_index(axis=1)
    plt.imshow(xg_map, aspect='auto', cmap=CMAP, norm=NORM, zorder=1);

    plt.scatter([105 for _ in range(25)], [i*3+12.5 for i in range(25)], c=[i/24 for i in range(25)], edgecolors='black', cmap=CMAP, norm=NORM, marker='s', s=70);
    plt.text(103.3, 12.5, '0', ha='right', va='center', fontsize=9, rotation=-90)
    plt.text(103.3, 20, '.1', ha='right', va='center', fontsize=9, rotation=-90)
    plt.text(103.3, 29, '.25', ha='right', va='center', fontsize=9, rotation=-90)
    plt.text(103.3, 47, '.5', ha='right', va='center', fontsize=9, rotation=-90)
    plt.text(103.3, 65, '.75', ha='right', va='center', fontsize=9, rotation=-90)
    plt.text(103.3, 87, '1', ha='right', va='center', fontsize=9, rotation=-90)
    plt.text(108, 50, 'Frequency', ha='center', va='center', rotation=-90, fontsize=16)
    plt.xlim(-5, 110);

BOUNDARIES = [-1, 0.001, 0.10, 0.25, 0.5, 0.75, 1]