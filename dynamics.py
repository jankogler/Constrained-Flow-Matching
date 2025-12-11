import os
import minari
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
DATASET_ID = "mujoco/hopper/medium-v0"  # Replace with your specific Minari dataset ID
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
HIDDEN_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# --- 1. Dynamics Model Definition ---
class DynamicsModel(nn.Module):
    """
    A Deterministic Dynamics Model that predicts the next state directly.
    
    Inputs: State (s), Action (a)
    Outputs: Delta (s' - s)
    """
    def __init__(self, obs_dim, act_dim, hidden_size=512):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Input: State + Action
        # Output: Delta (obs_dim)
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, obs_dim) # Outputs delta directly
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        delta = self.network(x)
        return delta

# --- 2. Data Loading & Preprocessing ---
def load_and_process_minari_data(dataset_id):
    print(f"Loading Minari dataset: {dataset_id}...")
    try:
        dataset = minari.load_dataset(dataset_id)
    except ValueError:
        print(f"Dataset {dataset_id} not found locally. Downloading...")
        minari.download_dataset(dataset_id)
        dataset = minari.load_dataset(dataset_id)

    # Containers for data
    observations = []
    actions = []
    next_observations = []

    # Iterate over episodes
    print("Extracting transitions...")
    for episode in dataset.iterate_episodes():
        # Minari episodes store:
        # observations: [T+1, obs_dim]
        # actions: [T, act_dim]

        obs_data = episode.observations
        
        # Handle Dictionary Observations (common in GoalEnv/PointMaze)
        if isinstance(obs_data, dict):
            obs_data = obs_data["observation"]
        
        obs = obs_data[:-1]      # s_t
        next_obs = obs_data[1:]  # s_{t+1}
        act = episode.actions[:]             # a_t

        observations.append(obs)
        next_observations.append(next_obs)
        actions.append(act)

    # Concatenate all transitions
    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)
    
    # Compute Deltas (Target = s' - s)
    deltas = next_observations - observations

    print(f"Total Transitions: {len(observations)}")
    return observations, actions, deltas

# --- 3. Normalization Helpers ---
class StandardScaler:
    def __init__(self, device):
        self.mean = None
        self.std = None
        self.device = device

    def fit(self, data):
        data_t = torch.tensor(data, dtype=torch.float32)
        self.mean = torch.mean(data_t, dim=0).to(self.device)
        self.std = torch.std(data_t, dim=0).to(self.device)
        # Handle zero std to avoid division by zero
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data):
        if torch.is_tensor(data):
            return (data - self.mean) / self.std
        return (torch.tensor(data, dtype=torch.float32, device=self.device) - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean'].to(self.device)
        self.std = state_dict['std'].to(self.device)

# --- 4. Main Training Script ---
def main():
    print(f"Using device: {DEVICE}")

    # A. Prepare Data
    obs_raw, act_raw, deltas_raw = load_and_process_minari_data(DATASET_ID)
    
    # B. Normalization
    print("Normalizing data...")
    scaler_obs = StandardScaler(DEVICE)
    scaler_act = StandardScaler(DEVICE)
    scaler_delta = StandardScaler(DEVICE)

    scaler_obs.fit(obs_raw)
    scaler_act.fit(act_raw)
    scaler_delta.fit(deltas_raw)

    # Transform data
    obs_norm = scaler_obs.transform(obs_raw).cpu()
    act_norm = scaler_act.transform(act_raw).cpu()
    delta_norm = scaler_delta.transform(deltas_raw).cpu()

    # C. Create Dataset (Full dataset, no validation split)
    dataset_full = TensorDataset(obs_norm, act_norm, delta_norm)
    train_loader = DataLoader(dataset_full, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training on full dataset ({len(dataset_full)} samples)...")

    # D. Initialize Model
    obs_dim = obs_raw.shape[1]
    act_dim = act_raw.shape[1]
    
    model = DynamicsModel(obs_dim, act_dim, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # E. Training Loop
    print("\nStarting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss_epoch = 0
        
        for batch_obs, batch_act, batch_delta in train_loader:
            batch_obs, batch_act, batch_delta = batch_obs.to(DEVICE), batch_act.to(DEVICE), batch_delta.to(DEVICE)
            
            # Forward pass
            pred_delta = model(batch_obs, batch_act)
            
            # Loss (MSE)
            loss = criterion(pred_delta, batch_delta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()

        avg_train_loss = train_loss_epoch / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE Loss: {avg_train_loss:.6f}")

    # Save Final Model AND Scalers
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_obs': scaler_obs.state_dict(),
        'scaler_act': scaler_act.state_dict(),
        'scaler_delta': scaler_delta.state_dict(),
        'config': {
            'obs_dim': obs_dim,
            'act_dim': act_dim,
            'hidden_size': HIDDEN_SIZE
        }
    }
    torch.save(checkpoint, f"model_weights/{DATASET_ID.replace('/', '_')}_dynamics.pth")
    print("\nTraining Complete. Model and Scalers saved as 'dynamics_checkpoint.pth'.")

    # Example Prediction
    print("\n--- Example Prediction (Unnormalized) ---")
    model.eval()
    with torch.no_grad():
        # Take a random sample from the dataset
        sample_obs, sample_act, sample_target_delta = dataset_full[0]
        sample_obs = sample_obs.unsqueeze(0).to(DEVICE)
        sample_act = sample_act.unsqueeze(0).to(DEVICE)
        
        # Predict delta (Normalized)
        pred_delta_norm = model(sample_obs, sample_act)
        
        # Inverse transform to get real world values
        real_obs = scaler_obs.inverse_transform(sample_obs)
        real_pred_delta = scaler_delta.inverse_transform(pred_delta_norm)
        real_target_delta = scaler_delta.inverse_transform(sample_target_delta.unsqueeze(0).to(DEVICE))
        
        predicted_next_state = real_obs + real_pred_delta
        true_next_state = real_obs + real_target_delta

        print(f"Current State[0]: {real_obs[0,0]:.4f}")
        print(f"Predicted Next State[0]: {predicted_next_state[0,0]:.4f}")
        print(f"True Next State[0]: {true_next_state[0,0]:.4f}")

if __name__ == "__main__":
    main()