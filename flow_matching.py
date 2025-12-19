import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import minari
import time
import math

# Import from the local dynamics.py file
try:
    from dynamics import DynamicsModel, StandardScaler
except ImportError:
    # Fallback for standalone testing
    class DynamicsModel(nn.Module):
        def __init__(self, o, a, hidden_size=256):
            super().__init__()
            self.net = nn.Linear(o+a, o)
        def forward(self, s, a): return self.net(torch.cat([s,a], -1))
    class StandardScaler(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.mean = torch.zeros(1).to(device)
            self.std = torch.ones(1).to(device)
        def load_state_dict(self, d): pass

# --- TorchCFM Imports ---
try:
    from torchcfm import ExactOptimalTransportConditionalFlowMatcher
except ImportError:
    print("Error: torchcfm not installed. Please install via 'pip install torchcfm'")
    exit()

# --- Global Defaults ---
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_STEPS = 100_000
BATCH_SIZE = 256
LEARNING_RATE = 2e-4
# Transformer Hyperparameters
HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 4
CONST_T = 0.5

# ==========================================
# 1. CONFIGURATION DICTIONARY
# ==========================================
CONFIGS = {
    'hopper': {
        'dataset_id': 'mujoco/hopper/medium-v0',
        'horizon': 32, # Full episode length
        'obs_dim': 11,
        'act_dim': 3,
        'dynamics_path': 'model_weights/mujoco_hopper_medium-v0_dynamics.pth',
        'safety_thresholds': {'height_max': 1.6}, 
        'action_bounds': {'min': -1.0, 'max': 1.0}
    },
    'walker2d': {
        'dataset_id': 'mujoco/walker2d/medium-v0',
        'horizon': 32, # Full episode length
        'obs_dim': 17,
        'act_dim': 6,
        'dynamics_path': 'model_weights/mujoco_walker2d_medium-v0_dynamics.pth',
        'safety_thresholds': {'height_max': 1.6},
        'action_bounds': {'min': -1.0, 'max': 1.0}
    },
    'maze2d-umaze': {
        'dataset_id': 'D4RL/pointmaze/umaze-v2', 
        'horizon': 64, # Standard Diffuser horizon for Umaze
        'obs_dim': 4, # [x, y, vx, vy]
        'act_dim': 2,
        'dynamics_path': 'model_weights/D4RL_pointmaze_umaze-v2_dynamics.pth',
        'safety_thresholds': {
            'obstacles': [
                # Obstacle 1: Degree 2 (Ellipse)
                {'x0': 4.5, 'y0': 5.3, 'a': 1.0, 'b': 1.0, 'degree': 2}, 
                # Obstacle 2: Degree 4 (Super-Ellipse)
                {'x0': 1.5, 'y0': 4.8, 'a': 1.0, 'b': 1.0, 'degree': 4}  
            ]
        },
        'action_bounds': {'min': -0.1, 'max': 0.1} 
    },
    'maze2d-large': {
        'dataset_id': 'D4RL/pointmaze/large-v2',
        'horizon': 128, # Standard Diffuser horizon for Large
        'obs_dim': 4,
        'act_dim': 2,
        'dynamics_path': 'model_weights/D4RL_pointmaze_large-v2_dynamics.pth',
        'safety_thresholds': {
            'obstacles': [
                # SAD-Flower Obstacles
                {'x0': 4.5, 'y0': 5.3, 'a': 1.0, 'b': 1.0, 'degree': 2}, 
                {'x0': 1.5, 'y0': 4.8, 'a': 1.0, 'b': 1.0, 'degree': 4}  
            ]
        },
        'action_bounds': {'min': -0.1, 'max': 0.1}
    }
}

# ==========================================
# 2. MODULAR CONSTRAINT SYSTEM
# ==========================================

class SADConstraintFunction(nn.Module):
    """
    Unified constraint class. 
    Implements Super-Ellipse obstacle avoidance (degree 2, 4, etc.)
    and strict action bounds checking.
    """
    def __init__(self, config, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.obs_dim = config['obs_dim']
        self.act_dim = config['act_dim']
        self.horizon = config['horizon']
        
        # 1. Load Dynamics Model
        path = config['dynamics_path']
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            # Handle variable hidden sizes if present
            h_size = checkpoint.get('config', {}).get('hidden_size', 512)
            self.dyn_model = DynamicsModel(self.obs_dim, self.act_dim, hidden_size=h_size)
            
            if 'model_state_dict' in checkpoint:
                self.dyn_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load Scalers
            self.scaler_obs = StandardScaler(device)
            if 'scaler_obs' in checkpoint: self.scaler_obs.load_state_dict(checkpoint['scaler_obs'])
            
            self.scaler_act = StandardScaler(device)
            if 'scaler_act' in checkpoint: self.scaler_act.load_state_dict(checkpoint['scaler_act'])
            
            self.scaler_delta = StandardScaler(device)
            if 'scaler_delta' in checkpoint: self.scaler_delta.load_state_dict(checkpoint['scaler_delta'])
            
            print(f"Loaded Dynamics Model & Scalers from {path}")
        else:
            print(f"Warning: Dynamics weights not found at {path}. Using random init.")
            self.dyn_model = DynamicsModel(self.obs_dim, self.act_dim)
            self.scaler_obs = StandardScaler(device)
            self.scaler_act = StandardScaler(device)
            self.scaler_delta = StandardScaler(device)
            
        self.dyn_model.to(device)
        self.dyn_model.eval()

        self.act_bounds = config.get('action_bounds', {'min': -1.0, 'max': 1.0})

    def interval_distance(self, val, min_val, max_val):
        """Calculates violation distance outside [min, max]."""
        return torch.relu(min_val - val) + torch.relu(val - max_val)

    def _adm_safety_constraints(self, traj_norm):
        """
        Calculates Admissibility (Actions) and Safety (States) constraints.
        Input: Full Trajectory (Normalized) [Batch, Horizon, Obs+Act]
        """
        batch_size = traj_norm.shape[0]
        thresholds = self.cfg.get('safety_thresholds', {})

        # Split trajectory
        states_norm = traj_norm[:, :, :self.obs_dim]
        actions_norm = traj_norm[:, :, self.obs_dim:]

        # --- 1. Admissibility: Action Bounds ---
        # Normalize the bounds (e.g. [-0.1, 0.1])
        act_mean = self.scaler_act.mean.view(1, 1, -1)
        act_std = self.scaler_act.std.view(1, 1, -1)

        a_min_norm = (self.act_bounds['min'] - act_mean) / act_std
        a_max_norm = (self.act_bounds['max'] - act_mean) / act_std
        
        g_adm = self.interval_distance(actions_norm, a_min_norm, a_max_norm)
        g_adm = g_adm.reshape(batch_size, -1)

        # --- 2. Safety Constraints ---
        g_safe = torch.zeros(batch_size, 0, device=self.device)

        is_maze = 'maze' in self.cfg['dataset_id']
        is_mujoco = 'hopper' in self.cfg['dataset_id'] or 'walker' in self.cfg['dataset_id']

        if is_mujoco:
            # Combined logic: Safety is ONLY height (index 0) <= 1.6 (Unnormalized)
            h_norm = states_norm[:, :, 0:1] # Keep dim for broadcasting
            
            # Max Height Constraint
            h_max_phys = thresholds.get('height_max', 1.6)
            # Manually normalize the limit to compare in norm space (optional, but consistent)
            h_mean = self.scaler_obs.mean[0]
            h_std = self.scaler_obs.std[0]
            h_max_norm = (h_max_phys - h_mean) / h_std
            
            # Constraint violation: max(0, h - 1.6)
            c_h_max = torch.relu(h_norm - h_max_norm)
            g_safe = c_h_max.reshape(batch_size, -1)

        elif is_maze:
            # Maze2d Constraints (Super-Ellipse Obstacles ONLY)
            
            # Use normalized states directly
            x_norm = states_norm[:, :, 0]
            y_norm = states_norm[:, :, 1]
            
            # Get scaler stats for transforming the obstacles to normalized space
            x_mean = self.scaler_obs.mean[0]
            x_std = self.scaler_obs.std[0]
            y_mean = self.scaler_obs.mean[1]
            y_std = self.scaler_obs.std[1]

            obstacle_constraints = []
            if 'obstacles' in thresholds:
                for obs in thresholds['obstacles']:
                    # Physical parameters
                    x0_phys, y0_phys = obs['x0'], obs['y0']
                    a_phys, b_phys = obs['a'], obs['b']
                    degree = obs.get('degree', 2.0) 
                    
                    # Transform center to Normalized Space: (val - mean) / std
                    x0_norm = (x0_phys - x_mean) / x_std
                    y0_norm = (y0_phys - y_mean) / y_std
                    
                    # Transform radii to Normalized Space: val / std (Scale only)
                    a_norm = a_phys / x_std
                    b_norm = b_phys / y_std
                    
                    # Super-Ellipse formula in Norm Space: 
                    # (|x_n - x0_n|/a_n)^d + (|y_n - y0_n|/b_n)^d <= 1 (Safe)
                    
                    term_x = (torch.abs(x_norm - x0_norm) / a_norm)**degree
                    term_y = (torch.abs(y_norm - y0_norm) / b_norm)**degree
                    
                    # Violation if sum < 1 (Inside obstacle)
                    obs_val = torch.relu(1.0 - (term_x + term_y))
                    obstacle_constraints.append(obs_val.reshape(batch_size, -1))
            
            if obstacle_constraints:
                g_safe = torch.cat(obstacle_constraints, dim=1)
            else:
                g_safe = torch.zeros(batch_size, 0, device=self.device)

        return g_safe, g_adm

    def forward(self, trajectory_flat):
        """
        Calculates the constraint vector g(T).
        Returns: [Batch, num_constraints]
        """
        batch_size = trajectory_flat.shape[0]
        traj = trajectory_flat.view(batch_size, self.horizon, self.obs_dim + self.act_dim)

        # B. Admissibility and Safety
        g_s, g_a = self._adm_safety_constraints(traj)

        # C. Dynamics Consistency
        states_norm = traj[:, :, :self.obs_dim]
        actions_norm = traj[:, :, self.obs_dim:]
        
        s_t_norm = states_norm[:, :-1, :]
        a_t_norm = actions_norm[:, :-1, :]
        s_t1_norm_target = states_norm[:, 1:, :]
        
        # 1. Model Prediction
        pred_delta_norm = self.dyn_model(s_t_norm, a_t_norm)
        
        # 2. Actual Delta (Careful unnormalization logic)
        obs_mean = self.scaler_obs.mean.view(1, 1, -1)
        obs_std = self.scaler_obs.std.view(1, 1, -1)
        delta_mean = self.scaler_delta.mean.view(1, 1, -1)
        delta_std = self.scaler_delta.std.view(1, 1, -1)

        s_t_real = s_t_norm * obs_std + obs_mean
        s_t1_real_target = s_t1_norm_target * obs_std + obs_mean
        
        actual_delta_real = s_t1_real_target - s_t_real
        
        # 3. Re-normalize using DELTA scaler
        actual_delta_norm = (actual_delta_real - delta_mean) / delta_std
        
        g_dyn = (actual_delta_norm - pred_delta_norm).reshape(batch_size, -1)

        # return torch.cat([g_a, g_s, g_dyn], dim=1)
        return torch.cat([g_a, g_s, g_dyn], dim=1)

# ==========================================
# 3. CONSTRAINED VECTOR FIELD (PDF Formula)
# ==========================================

def compute_constrained_velocity(flow_model, x, t, constraint_fn, T_thresh=CONST_T, training=False):
    """
    Implements the constrained velocity field \tilde{v} from the PDF (Section 1.2).
    """
    x_in = x.detach() 
    x_in.requires_grad_(True)
    
    if isinstance(t, (float, int)):
        t_val = t
        t_tensor = torch.tensor([t], device=x.device).float()
    else:
        t_val = t.item() if t.numel() == 1 else t[0].item()
        t_tensor = t

    # 1. Base Velocity
    v_theta = flow_model(t_tensor, x_in)
    
    # If before T_thresh, return base velocity
    if t_val < T_thresh:
        return v_theta if training else v_theta.detach()

    # 2. Compute Constraints
    g_s, g_a = constraint_fn._adm_safety_constraints(x_in.view(x_in.shape[0], constraint_fn.horizon, -1))
    
    # 3. Gradient 'a' = J^T * g
    constraint_energy = 0.5 * torch.sum(torch.cat([g_a, g_s], dim=1)**2, dim=1).sum()
    
    a = torch.autograd.grad(constraint_energy, x_in, create_graph=False)[0]
    
    # 4. Projection
    a_norm_val = torch.norm(a, dim=1, keepdim=True) + 1e-8
    a_tilde = a / a_norm_val
    rho_t = torch.exp(4.0 * t_tensor)

    # 5. Inner Product <\tilde{a}, v_theta>
    av_dot = torch.sum(a_tilde * v_theta, dim=1, keepdim=True)
    
    # Correction: (max{<a,v>, 0} + rho(t))
    correction_scalar = torch.clamp(av_dot, min=0.0) + rho_t
    
    # \tilde{v} = v - correction * \tilde{a}
    v_constrained = v_theta - (correction_scalar * a_tilde)
    
    if training:
        return v_constrained
    else:
        return v_constrained.detach()

# ==========================================
# 4. TRANSFORMER FLOW MODEL
# ==========================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class TransformerFlowModel(nn.Module):
    """
    Transformer-based Flow Model.
    Replaces the previous MLP-based FlowNetwork.
    
    Structure:
    1. Input Embedding (x_traj -> hidden)
    2. Positional Embedding (Sequence pos + Diffusion time)
    3. Transformer Encoder Blocks
    4. Output Projection (hidden -> v_traj)
    """
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, num_layers=4, num_heads=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.input_dim = obs_dim + act_dim
        self.hidden_dim = hidden_dim

        # Input Projection
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)

        # Time Embedding (for diffusion time t)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Sequence Positional Embedding (for trajectory steps 0...H)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, hidden_dim))

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # Output Projection
        self.output_proj = nn.Linear(hidden_dim, self.input_dim)
        
        # Initialize Positional Embeddings
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, t, x_flat):
        """
        Args:
            t: Diffusion time [Batch, 1] or [Batch]
            x_flat: Flattened trajectory [Batch, Horizon * (Obs+Act)]
        Returns:
            v_flat: Flattened velocity [Batch, Horizon * (Obs+Act)]
        """
        batch_size = x_flat.shape[0]

        # 1. Reshape Input: [B, H*D] -> [B, H, D]
        x = x_flat.view(batch_size, self.horizon, self.input_dim)

        # 2. Embed Trajectory
        x_emb = self.input_proj(x) # [B, H, Hidden]

        # 3. Add Sequence Positional Embeddings
        x_emb = x_emb + self.pos_emb # Broadcasts to batch

        # 4. Process Diffusion Time 't'
        if t.dim() == 1: t = t.unsqueeze(1) # [B, 1]
        if t.dim() == 2 and t.shape[1] == 1:
            t_emb = self.time_mlp(t.squeeze(1)) # [B, Hidden]
        else:
            t_emb = self.time_mlp(t)
        
        # 5. Condition: Add time embedding to every sequence token
        # (Alternative: Concat t_emb as a token, but adding is standard in DiT/Adan)
        x_emb = x_emb + t_emb.unsqueeze(1) 

        # 6. Transformer Passes
        for block in self.blocks:
            x_emb = block(x_emb)

        # 7. Output Projection
        v = self.output_proj(x_emb) # [B, H, Input_Dim]

        # 8. Flatten Output: [B, H, D] -> [B, H*D]
        v_flat = v.view(batch_size, -1)
        
        return v_flat

def get_trajectory_dataloader(dataset_id, horizon, obs_dim, act_dim, batch_size, dynamics_path):
    print(f"Loading Dataset: {dataset_id}...")
    trajectories = []
    
    # Helper to clean up raw data
    def process_episodes(episodes_iterable):
        traj_list = []
        for episode in episodes_iterable:
            obs = episode.observations
            # Handle Minari dictionary observations if present
            if isinstance(obs, dict): obs = obs['observation']
            
            act = episode.actions
            
            # Basic D4RL/Minari slicing: obs is usually N+1, act is N
            if len(obs) == len(act) + 1:
                obs_seq = obs[:-1]
                act_seq = act[:]
            else:
                obs_seq = obs[:]
                act_seq = act[:]

            # Validation
            if len(act_seq) < horizon: continue
                
            joint_data = np.concatenate([obs_seq, act_seq], axis=-1)
            # Sliding window
            for i in range(0, len(act_seq) - horizon + 1, 1):
                window = joint_data[i : i+horizon] 
                traj_list.append(window.flatten())
        return traj_list

    try:
        # Try Loading via Minari
        try:
            dataset = minari.load_dataset(dataset_id)
            trajectories = process_episodes(dataset.iterate_episodes())
        except (ValueError, KeyError):
            print(f"Minari load failed for {dataset_id}. Checking for D4RL download...")
            try:
                minari.download_dataset(dataset_id)
                dataset = minari.load_dataset(dataset_id)
                trajectories = process_episodes(dataset.iterate_episodes())
            except Exception as e:
                # Fallback logic if user provides D4RL ID but not Minari
                raise ValueError(f"Dataset {dataset_id} not found in M_(True)")

        if len(trajectories) == 0: raise ValueError("No valid trajectories found in dataset")
        trajectories = np.array(trajectories, dtype=np.float32)
        print(f"Extracted {len(trajectories)} trajectories.")
        
        if os.path.exists(dynamics_path):
            print(f"Loading scalers from {dynamics_path} for normalization consistency...")
            checkpoint = torch.load(dynamics_path, map_location='cpu')
            
            # Logic to extract scaler stats
            if 'scaler_obs' in checkpoint:
                obs_mean = checkpoint['scaler_obs']['mean'].cpu().numpy()
                obs_std = checkpoint['scaler_obs']['std'].cpu().numpy()
                act_mean = checkpoint['scaler_act']['mean'].cpu().numpy()
                act_std = checkpoint['scaler_act']['std'].cpu().numpy()
                
                step_mean = np.concatenate([obs_mean, act_mean])
                step_std = np.concatenate([obs_std, act_std])
                
                full_mean = np.tile(step_mean, horizon)
                full_std = np.tile(step_std, horizon)
                
                trajectories = (trajectories - full_mean) / full_std
            else:
                 print("WARNING: Checkpoint exists but no scaler_obs found. Using data stats.")
                 mean = np.mean(trajectories, axis=0)
                 std = np.std(trajectories, axis=0) + 1e-6
                 trajectories = (trajectories - mean) / std
        else:
            print("WARNING: Dynamics checkpoint not found. Using simple stats (Possible Mismatch with Constraint Fn).")
            mean = np.mean(trajectories, axis=0)
            std = np.std(trajectories, axis=0) + 1e-6
            trajectories = (trajectories - mean) / std
        
    except Exception as e:
        print(f"CRITICAL ERROR LOADING DATA ({e}). Generating DUMMY data for verification.")
        trajectories = np.random.randn(100, horizon * (obs_dim + act_dim)).astype(np.float32)

    dataset = TensorDataset(torch.tensor(trajectories))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper', choices=CONFIGS.keys())
    parser.add_argument('--level', type=str, default='medium', help="Dataset level (e.g. medium, expert, umaze)")
    args = parser.parse_args()

    # Load Config
    cfg = CONFIGS[args.env]
    
    # If the user selects hopper/walker, apply level patching as before.
    # For maze2d, we rely on the specific keys 'maze2d-umaze' or 'maze2d-large' being selected.
    if 'mujoco' in cfg['dataset_id']:
        if args.level != 'medium':
            if 'medium' in cfg['dataset_id']:
                cfg['dataset_id'] = cfg['dataset_id'].replace('medium', args.level)
            if 'medium' in cfg['dynamics_path']:
                cfg['dynamics_path'] = cfg['dynamics_path'].replace('medium', args.level)

    print(f"\n=== Training Configuration: {args.env.upper()} ===")
    print(f"Dataset: {cfg['dataset_id']}")
    
    # Fix: Calculate obstacle count outside f-string to avoid syntax highlighting issues with nested {}
    num_obstacles = len(cfg.get('safety_thresholds', {}).get('obstacles', []))
    print(f"Obstacles: {num_obstacles}")
    
    print(f"Action Bounds: {cfg.get('action_bounds', 'Default')}")
    print("========================================\n")

    # 1. Prepare Data
    dataloader = get_trajectory_dataloader(
        cfg['dataset_id'], cfg['horizon'], cfg['obs_dim'], cfg['act_dim'], BATCH_SIZE, cfg['dynamics_path']
    )

    # 2. Initialize Models (TRANSFORMER REPLACEMENT)
    # Note: input_dim for Transformer logic is handled inside the class, we pass obs/act dim separately
    flow_net = TransformerFlowModel(
        obs_dim=cfg['obs_dim'], 
        act_dim=cfg['act_dim'], 
        horizon=cfg['horizon'],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS
    ).to(DEVICE)
    
    optimizer = optim.Adam(flow_net.parameters(), lr=LEARNING_RATE)
    # constraints = SADConstraintFunction(cfg, DEVICE)
    
    # 3. CFM Optimizer
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    print(f"Starting Training with Transformer (Dim: {HIDDEN_DIM}, Layers: {NUM_LAYERS})...")
    
    traj_dim_flat = cfg['horizon'] * (cfg['obs_dim'] + cfg['act_dim']) # For sanity check shape later
    
    EPOCHS = math.ceil(TRAINING_STEPS / len(dataloader))
    for epoch in range(EPOCHS):
        flow_net.train()
        epoch_loss = 0
        
        for batch_idx, (x1,) in enumerate(dataloader):
            x1 = x1.to(DEVICE)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            
            # Use Constrained Velocity for Prediction (Training Time)
            vt_pred = flow_net(t, xt)
            # vt_pred = compute_constrained_velocity(
            #     flow_net, xt, t, constraints, 
            #     T_thresh=CONST_T,
            #     training=True 
            # )
            
            loss = torch.mean((vt_pred - ut) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")
            
            # Quick sanity check of constraint field strength
            # with torch.no_grad():
            #     x_sample = torch.randn(1, traj_dim_flat, device=DEVICE)
            #     t_sample = torch.tensor([0.9], device=DEVICE)
            #     v_base = flow_net(t_sample, x_sample)
            #     with torch.enable_grad():
            #         v_constrained = compute_constrained_velocity(
            #             flow_net, x_sample, t_sample, constraints, 
            #             T_thresh=CONST_T, training=False
            #         )
            #     diff = (v_base - v_constrained).norm().item()
            #     if diff > 1e-5:
            #         print(f"  > Constraint Correction Active: {diff:.4f}")

    if 'mujoco' in cfg['dataset_id']:
        save_name = f"model_weights/flow_model_{args.env}_{args.level}.pth"
    else:
        save_name = f"model_weights/flow_model_{args.env}.pth"
    torch.save(flow_net.state_dict(), save_name)
    print(f"\nTraining Complete. Model saved to {save_name}")

if __name__ == "__main__":
    START_TIME = time.time()
    main()
    END_TIME = time.time()
    print("Elapsed time:", END_TIME - START_TIME, "seconds")
