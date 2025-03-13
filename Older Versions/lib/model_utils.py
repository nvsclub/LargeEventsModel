import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss

class SingleLayerBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleLayerBinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
        # Initialize the linear layers
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)
    
class TripleLayerBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TripleLayerBinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
        # Initialize the linear layers
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)
    
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
    
class TransferModel(nn.Module):
    def __init__(self, base_model):
        super(TransferModel, self).__init__()

        modules = list(base_model.children())[0][:-1]
        self.base_layers = nn.Sequential(*modules)
        if 'ReLU' in str(self.base_layers[-2]):
            self.transfer_activation = nn.ReLU()
        elif 'Sigmoid' in str(self.base_layers[-2]):
            self.transfer_activation = nn.Sigmoid()
        else:
            self.transfer_activation = nn.Sigmoid()

        output_size = list(base_model.children())[0][-2].out_features
        self.transfer_layer = nn.Linear(output_size, output_size)
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_layers(x)
        x = self.transfer_activation(x)
        x = self.transfer_layer(x)
        x = self.output_sigmoid(x)
        return x


def train(model, dataloader, criterion, optimizer, device, weights=None, l1_lambda=None):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        if weights == None:
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs[:,0], labels[:,0]) * weights[0]
            for i in range(1, len(weights)):
                loss += criterion(outputs[:,i], labels[:,i]) * weights[i]

        if l1_lambda != None:
            l1_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += l1_lambda * l1_reg

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_log_loss(model, dataloader, device):
    model.eval()
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).cpu().numpy()
            true_labels.extend(labels.cpu().numpy().tolist())
            predicted_probs.extend(outputs.tolist())

        # THE ERROR IS HERE

    epoch_log_loss = log_loss(true_labels, predicted_probs)
    return epoch_log_loss

def predict(model, inputs, device):
    model.eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.cpu().numpy()


# Define the objective function for Optuna optimization
def objective(trial, X_train_tensor, Y_train_tensor, model_name, device=None, train_test_split=0.7, complexity_penalty=0.0):
    if device==None:
        device = torch.device("cpu")

    input_size = X_train_tensor.shape[1]
    output_size = Y_train_tensor.shape[1]
    num_epochs = 100
    patience = 3
    counter = 0
    best_val_loss = 1000

    # Define hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 1, 3)
    hidden_size_list = [2 ** trial.suggest_int(f"hidden_size_{i}", 4, 8) for i in range(3)]
    lr = round(trial.suggest_float("lr", 1e-4, 1e-1), 4)
    batch_size = 2 ** trial.suggest_int("batch_size", 5, 10)
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

    # Create the neural network
    model = MultiLayerBinaryClassifier(input_size, [hidden_size_list[i] for i in range(hidden_size)], output_size, activation=activation)

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create the loss function
    criterion = nn.BCELoss()

    # 
    split_id = int(train_test_split * len(X_train_tensor))
    _train_dataset = TensorDataset(X_train_tensor[:split_id], Y_train_tensor[:split_id])
    _test_dataset = TensorDataset(X_train_tensor[split_id:], Y_train_tensor[split_id:])

    # Create dataloaders
    _train_dataloader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True)
    _test_dataloader = DataLoader(_test_dataset, batch_size=batch_size, shuffle=False)

    # Train the neural network
    for epoch in range(num_epochs):
        train_loss = train(model, _train_dataloader, criterion, optimizer, device)
        test_log_loss = evaluate_log_loss(model, _test_dataloader, device)

        if test_log_loss < best_val_loss:
            best_val_loss = test_log_loss
            counter = 0
            torch.save(model, f'models/lem/optuna_trials/{model_name}_{trial.number}.pt')
        else:
            counter += 1
            if counter >= patience:
                break

    f = open(f'res/model_tunning/lem_trial_results.csv', 'a')
    f.write(f'{model_name},{trial.number},{round(best_val_loss, 4)},{round(best_val_loss * (1 + complexity_penalty * hidden_size), 4)},{hidden_size},{hidden_size_list},{lr},{batch_size},{activation},{epoch}\n')
    f.close()

    return best_val_loss * (1 + complexity_penalty * hidden_size)