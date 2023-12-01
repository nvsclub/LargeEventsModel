import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
import numpy as np

class Simulator:
    def __init__(self, model_type_path, model_acc_path, model_data_path, device=None):

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model_type = torch.load(model_type_path).to(self.device)
        self.model_acc = torch.load(model_acc_path).to(self.device)
        self.model_data = torch.load(model_data_path).to(self.device)
            
        self.model_type.eval()
        self.model_acc.eval()
        self.model_data.eval()

    def simulate(self, initial_state, n_sims=1000, game_length=2000, store_full_sim=False, disable_tqdm=False):
        init_feature_tensor = handle_initial_state(initial_state, n_sims)
        feature_tensor = init_feature_tensor.to(self.device) # shape: (1000, 42)
        if store_full_sim:
            all_sims_data = [init_feature_tensor]
        for k in tqdm(range(game_length), disable=disable_tqdm):
            with torch.no_grad():
                pred_type_probs = self.model_type(feature_tensor) # shape: (1000, 33)
                pred_type = torch.multinomial(pred_type_probs, 1) # shape: (1000, 1)
                pred_type = one_hot(pred_type, num_classes=pred_type_probs.shape[1]).squeeze(1) # shape: (1000, 33)
                
                pred_acc_input = torch.cat([feature_tensor, pred_type], dim=-1) # shape: (1000, 77)
                pred_acc_probs = self.model_acc(pred_acc_input) # shape: (1000, 2)
                pred_acc = torch.bernoulli(pred_acc_probs) # shape: (1000, 2)

                pred_data_input = torch.cat([pred_acc_input, pred_acc], dim=-1) # shape: (1000, 77)
                pred_data_probs = self.model_data(pred_data_input) # shape: (1000, 62)
                one_hot_probs_1 = pred_data_probs[:, :61] # shape: (1000, 61)
                one_hot_probs_1 = one_hot_probs_1 / one_hot_probs_1.sum(dim=-1, keepdim=True)
                one_hot_probs_x = pred_data_probs[:, 61:162] # shape: (1000, 101)
                one_hot_probs_x = one_hot_probs_x / one_hot_probs_x.sum(dim=-1, keepdim=True)
                one_hot_probs_y = pred_data_probs[:, 162:263] # shape: (1000, 101)
                one_hot_probs_y = one_hot_probs_y / one_hot_probs_y.sum(dim=-1, keepdim=True)                
                binary_prob = pred_data_probs[:, -1] # shape: (1000)

                pred_next_time = torch.multinomial(one_hot_probs_1, 1) # shape: (1000, 1)
                pred_next_x = torch.multinomial(one_hot_probs_x, 1) # shape: (1000, 1)
                pred_next_y = torch.multinomial(one_hot_probs_y, 1) # shape: (1000, 1)
                pred_next_team = torch.bernoulli(binary_prob).unsqueeze(1) # shape: (1000, 1)

            feature_tensor, all_simulations_finished = refresh_feature_tensor(feature_tensor, pred_type, pred_acc, pred_next_time, pred_next_x, pred_next_y, pred_next_team)
            if store_full_sim:
                all_sims_data.append(feature_tensor)
            
            if all_simulations_finished:
                break

        if store_full_sim:
            return feature_tensor, all_sims_data
        else:
            return feature_tensor

def repeat_init_tensor(values, k):
    tensor_values = torch.tensor(values, dtype=torch.float32)
    repeated_tensor = tensor_values.repeat(k, 1)
    return repeated_tensor

def refresh_feature_tensor(feature_tensor, pred_type_tensor, pred_acc_tensor, pred_next_time_tensor, pred_next_x_tensor, pred_next_y_tensor, pred_next_team_tensor):
    pred_time_tensor = feature_tensor[:, 33:35]

    pred_time_tensor[:, 1] = pred_time_tensor[:, 1] + pred_next_time_tensor.squeeze(1) / 60 / 60
    pred_time_tensor[:, 0][pred_time_tensor[:, 1] > 0.75] += 1
    pred_time_tensor[:, 1][pred_time_tensor[:, 1] > 0.75] = 0

    ongoing_game_tensor = (pred_time_tensor[:, 0] <= 1)

    pred_next_score_tensor = feature_tensor[:, 40:42]
    is_shot_tensor = (pred_type_tensor[:, 32] == 1) | (pred_type_tensor[:, 31] == 1) | (pred_type_tensor[:, 16] == 1)
    pred_next_score_tensor[:, 0] = pred_next_score_tensor[:, 0] + ongoing_game_tensor * pred_acc_tensor[:,1] * pred_next_team_tensor.squeeze(1) * is_shot_tensor / 10
    pred_next_score_tensor[:, 1] = pred_next_score_tensor[:, 1] + ongoing_game_tensor * pred_acc_tensor[:,1] * (pred_next_team_tensor.squeeze(1) == 0) * is_shot_tensor / 10

    feature_tensor = torch.cat((
        pred_type_tensor, 
        pred_time_tensor, 
        pred_next_x_tensor / 100, 
        pred_next_y_tensor / 100, 
        pred_next_team_tensor, 
        pred_acc_tensor, 
        pred_next_score_tensor), 1)
    
    all_simulations_finished = (pred_time_tensor[:, 0] > 1).sum() == pred_time_tensor.shape[0]

    return feature_tensor, all_simulations_finished

def handle_initial_state(initial_state, n_sims=1000):
    if not isinstance(initial_state, np.ndarray):
        initial_state = np.array(initial_state)

    if len(initial_state.shape) == 1:
        init_feature_tensor = repeat_init_tensor(initial_state, n_sims)

    else:
        init_feature_tensor = torch.tensor(initial_state, dtype=torch.float32)

    return init_feature_tensor