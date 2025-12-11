import os
import argparse
import numpy as np
import torch
import minari
from collections import defaultdict

# Import your local modules
from flow_matching import FlowNetwork, SADConstraintFunction, compute_constrained_velocity, CONFIGS, get_trajectory_dataloader
from dynamics import DynamicsModel, StandardScaler

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ODE_STEPS = 1e5  # Number of steps for ODE solver (Euler integration)

# --- D4RL Normalization Constants ---
D4RL_REF_SCORES = {
    'hopper':   {'min': -20.272305, 'max': 3234.3},
    'walker2d': {'min': 1.629008,   'max': 4592.3},
    'pointmaze': {'min': 0.0, 'max': 1.0} 
}

# --- Maze Configuration ---
# Fixed Start and Goal locations
MAZE_GOALS = {
    'maze2d-umaze': np.array([3.0, 1.0]),
    'maze2d-large': np.array([7.0, 9.0])
}
MAZE_STARTS = {
    'maze2d-umaze': np.array([1.0, 1.0]),
    'maze2d-large': np.array([1.0, 1.0])
}
MAZE_GOAL_THRESH = 0.5

def solve_ode_constrained(flow_model, x_init, constraint_fn, steps=ODE_STEPS, const_t=0.5, 
                          cond_mask=None, cond_targets=None):
    """
    Generates trajectories by solving the ODE.
    Includes 'Inpainting' conditioning for start/goal states.
    """
    batch_size = x_init.shape[0]
    dt = 1.0 / steps
    x = x_init.clone()
    
    # Pre-computation for conditioning
    has_condition = (cond_mask is not None) and (cond_targets is not None)

    for i in range(steps):
        t_float = i * dt
        t_next = (i + 1) * dt
        t_tensor = torch.full((batch_size, 1), t_float, device=x.device)
        
        # 1. Calculate Velocity (with Safety/Dynamics correction)
        with torch.enable_grad():
            v = compute_constrained_velocity(
                flow_model, x, t_tensor, constraint_fn, 
                T_thresh=const_t,
                training=False
            )
        
        # 2. Euler Step
        x = x + v * dt
        
        # 3. Apply Conditioning (Inpainting)
        if has_condition:
            interpolated_cond = (1 - t_next) * x_init + t_next * cond_targets
            x = torch.where(cond_mask, interpolated_cond, x)
        
    return x

def calculate_maze_path_length(trajectories_norm, env_name, config, scaler_obs):
    """
    Calculates Path Length (Time-to-Goal).
    Since arrival is guaranteed by inpainting, this measures EFFICIENCY.
    (i.e., did we reach the goal radius at t=0.8 or wait until t=1.0?)
    """
    if 'pointmaze' not in env_name and 'maze' not in env_name:
        return 0.0

    batch_size = trajectories_norm.shape[0]
    obs_dim = config['obs_dim']
    horizon = config['horizon']
    
    traj_reshaped = trajectories_norm.view(batch_size, horizon, -1)
    obs_norm = traj_reshaped[:, :, :obs_dim]
    obs_real = scaler_obs.inverse_transform(obs_norm).cpu().numpy() # [B, H, Obs]
    
    # Determine Goal
    if 'large' in env_name: goal = MAZE_GOALS['maze2d-large']
    elif 'umaze' in env_name: goal = MAZE_GOALS['maze2d-umaze']
    else: goal = MAZE_GOALS['maze2d-umaze'] # Fallback
    
    # Vectorized Calculation
    path_xy = obs_real[:, :, :2]
    # Axis 2 is correct for [Batch, Horizon, XY]
    dists = np.linalg.norm(path_xy - goal, axis=2) 
    
    success_mask = dists < MAZE_GOAL_THRESH
    
    # Check if ANY point in trajectory is within threshold
    any_success = np.any(success_mask, axis=1)
    
    # Find first index where agent enters goal radius
    first_success_indices = np.argmax(success_mask, axis=1)
    
    if np.any(any_success):
        avg_path_len = np.mean(first_success_indices[any_success])
    else:
        avg_path_len = horizon

    return avg_path_len

def evaluate_rewards(trajectories_norm, env_name, config, scaler_obs, scaler_act):
    batch_size = trajectories_norm.shape[0]
    obs_dim = config['obs_dim']
    act_dim = config['act_dim']
    horizon = config['horizon']

    traj_reshaped = trajectories_norm.view(batch_size, horizon, obs_dim + act_dim)
    obs_norm = traj_reshaped[:, :, :obs_dim]
    act_norm = traj_reshaped[:, :, obs_dim:]
    
    obs_real = scaler_obs.inverse_transform(obs_norm).cpu().numpy()
    act_real = scaler_act.inverse_transform(act_norm).cpu().numpy()
    
    total_rewards = []
    print(f"  > Calculating analytical rewards for {env_name}...")
    
    for i in range(batch_size):
        traj_obs = obs_real[i]
        traj_act = act_real[i]
        episode_reward = 0.0
        
        if 'hopper' in env_name:
            if obs_dim >= 6:
                x_vel = traj_obs[:, 5] 
                ctrl_cost = 1e-3 * np.sum(np.square(traj_act), axis=1)
                healthy_reward = 1.0
                episode_reward = np.sum(x_vel - ctrl_cost + healthy_reward)

        elif 'walker2d' in env_name:
            if obs_dim >= 9:
                x_vel = traj_obs[:, 8]
                ctrl_cost = 1e-3 * np.sum(np.square(traj_act), axis=1)
                healthy_reward = 1.0
                episode_reward = np.sum(x_vel - ctrl_cost + healthy_reward)
        else:
            episode_reward = 0.0
            
        total_rewards.append(episode_reward)

    return np.array(total_rewards)

def calculate_normalized_score(env_key, raw_score_mean, horizon):
    if env_key not in D4RL_REF_SCORES: return 0.0
    if horizon < 1000:
        extrapolated_score = raw_score_mean * (1000.0 / horizon)
    else:
        extrapolated_score = raw_score_mean

    ref = D4RL_REF_SCORES[env_key]
    return 100.0 * (extrapolated_score - ref['min']) / (ref['max'] - ref['min'])

def prepare_conditions(env_name, config, batch_size, device, scaler_obs):
    """
    Prepares condition vectors.
    - Maze: Uses FIXED Start and Goal locations.
    - Locomotion: Uses uniform distribution centered at 0 (height at 1.25).
    """
    obs_dim = config['obs_dim']
    act_dim = config['act_dim']
    step_dim = obs_dim + act_dim
    horizon = config['horizon']
    
    # Initialize target/mask tensors
    cond_targets = torch.zeros((batch_size, horizon * step_dim), device=device)
    mask = torch.zeros((batch_size, horizon * step_dim), dtype=torch.bool, device=device)
    
    # 1. INITIAL STATE CONDITIONING (s_0)
    print("  > preparing initial state conditions...")
    real_s0_np = np.zeros((batch_size, obs_dim), dtype=np.float32)
    
    is_maze = 'maze' in config['dataset_id'] or 'point' in config['dataset_id']
    
    if is_maze:
        # Maze: Fixed Start
        if 'large' in env_name: start_xy = MAZE_STARTS['maze2d-large']
        elif 'umaze' in env_name: start_xy = MAZE_STARTS['maze2d-umaze']
        else: start_xy = MAZE_STARTS['maze2d-umaze']
        
        # Set pos=start, vel=0
        real_s0_np[:, 0] = start_xy[0]
        real_s0_np[:, 1] = start_xy[1]
        # Rest of dims (velocities) stay 0.0
        print(f"    > Using Fixed Start: {start_xy}")
        
    else:
        # Locomotion: Uniform Initialization (Replacing Gym)
        # Uniform distribution from -5e-3 to +5e-3
        print(f"    > Sampling uniform initial states (Hopper/Walker style)...")
        real_s0_np = np.random.uniform(low=-5e-3, high=5e-3, size=(batch_size, obs_dim))
        
        # Set height (index 0) to be centered at 1.25
        # 1.25 is added to the small noise centered at 0
        real_s0_np[:, 0] += 1.25
    
    # Normalize and Apply s_0
    real_s0_tensor = torch.tensor(real_s0_np, dtype=torch.float32, device=device)
    norm_s0 = scaler_obs.transform(real_s0_tensor)
    
    cond_targets[:, :obs_dim] = norm_s0
    mask[:, :obs_dim] = True
    
    # 2. FINAL STATE CONDITIONING (s_T) - Maze Only
    if is_maze:
        if 'large' in env_name: goal_xy = MAZE_GOALS['maze2d-large']
        elif 'umaze' in env_name: goal_xy = MAZE_GOALS['maze2d-umaze']
        else: goal_xy = MAZE_GOALS['maze2d-umaze']

        # Construct Normalized Goal State (Pos=Goal, Vel=0)
        raw_goal_state = torch.zeros((batch_size, obs_dim), device=device)
        raw_goal_state[:, 0] = goal_xy[0]
        raw_goal_state[:, 1] = goal_xy[1]
        norm_goal_state = scaler_obs.transform(raw_goal_state)
        
        # Fill last step obs with goal
        last_step_start = (horizon - 1) * step_dim
        last_step_end_obs = last_step_start + obs_dim
        
        cond_targets[:, last_step_start:last_step_end_obs] = norm_goal_state
        mask[:, last_step_start:last_step_end_obs] = True
        
        print(f"    > Using Fixed Goal: {goal_xy}")

    return mask, cond_targets

def main():
    parser = argparse.ArgumentParser(description="Benchmark Flow Matching Model")
    parser.add_argument('--env', type=str, default='hopper', choices=CONFIGS.keys())
    parser.add_argument('--level', type=str, default='medium', help="Dataset level")
    parser.add_argument('--samples', type=int, default=128, help="Number of trajectories")
    parser.add_argument('--model_path', type=str, default=None, help="Path to trained model")
    args = parser.parse_args()

    if args.env not in CONFIGS: raise ValueError(f"Unknown environment: {args.env}")
    cfg = CONFIGS[args.env]
    if args.level != 'medium':
        cfg['dataset_id'] = cfg['dataset_id'].replace('medium', args.level)
        cfg['dynamics_path'] = cfg['dynamics_path'].replace('medium', args.level)
    if args.model_path is None:
        args.model_path = f"model_weights/flow_model_{args.env}_{args.level}.pth"

    print(f"=== Benchmarking {args.env.upper()} ({args.level}) ===")
    
    # Load Models
    constraints = SADConstraintFunction(cfg, DEVICE)
    traj_dim = cfg['horizon'] * (cfg['obs_dim'] + cfg['act_dim'])
    flow_net = FlowNetwork(input_dim=traj_dim, hidden_dim=512).to(DEVICE)
    if os.path.exists(args.model_path):
        flow_net.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    else:
        print(f"ERROR: Model not found at {args.model_path}"); return
    flow_net.eval()

    # Prepare Conditions (Updated)
    cond_mask, cond_targets = prepare_conditions(
        args.env, cfg, args.samples, DEVICE, constraints.scaler_obs
    )

    # Generate
    print(f"\nGenerating {args.samples} trajectories...")
    x0 = torch.randn(args.samples, traj_dim, device=DEVICE)
    with torch.no_grad():
        generated_trajs = solve_ode_constrained(
            flow_net, x0, constraints, steps=100, const_t=0.5,
            cond_mask=cond_mask, cond_targets=cond_targets
        )

    # --- METRICS LOGIC ---
    print("Computing SAD Metrics...")
    g_vals = constraints(generated_trajs)
    
    dummy_traj = generated_trajs[0:1]
    g_s, g_a = constraints._adm_safety_constraints(dummy_traj.view(1, cfg['horizon'], -1))
    dim_adm = g_a.shape[1]
    dim_safe = g_s.shape[1]
    
    g_adm_vals = g_vals[:, :dim_adm]
    g_safe_vals = g_vals[:, dim_adm : dim_adm+dim_safe]
    g_dyn_vals = g_vals[:, dim_adm+dim_safe:]

    def get_inequality_stats(tensor):
        # Max violation per trajectory
        max_violation_per_traj, _ = torch.max(tensor, dim=1)
        violation_rate = (max_violation_per_traj > 1e-4).float().mean().item() * 100
        avg_max_violation = max_violation_per_traj.mean().item()
        return violation_rate, avg_max_violation

    def get_equality_stats(tensor):
        # Energy per trajectory
        energy_per_traj = 0.5 * torch.sum(tensor ** 2, dim=1)
        violation_rate = (energy_per_traj > 1e-4).float().mean().item() * 100
        avg_energy = energy_per_traj.mean().item()
        return violation_rate, avg_energy

    adm_rate, adm_mag = get_inequality_stats(g_adm_vals)
    safe_rate, safe_mag = get_inequality_stats(g_safe_vals)
    dyn_rate, dyn_mag = get_equality_stats(g_dyn_vals)
    
    # Overall Success (All constraints satisfied)
    s_success = (torch.max(g_safe_vals, dim=1)[0] <= 1e-4)
    a_success = (torch.max(g_adm_vals, dim=1)[0] <= 1e-4)
    d_success = (0.5 * torch.sum(g_dyn_vals ** 2, dim=1) <= 1e-4)
    success_rate = (s_success & a_success & d_success).float().mean().item() * 100

    # Rewards / Path Metrics
    if 'maze' in args.env or 'point' in args.env:
        # NOTE: Geometric success is guaranteed by inpainting. 
        # The metric of interest is VALID PATH RATE (Constraint Success).
        maze_len = calculate_maze_path_length(generated_trajs, args.env, cfg, constraints.scaler_obs)
        reward_str = f"Valid Path %: {success_rate:.1f}% | Avg Steps: {maze_len:.1f}"
    else:
        raw_rewards = evaluate_rewards(generated_trajs, args.env, cfg, constraints.scaler_obs, constraints.scaler_act)
        mean_raw = raw_rewards.mean()
        norm_score = calculate_normalized_score(args.env, mean_raw, cfg['horizon'])
        reward_str = f"Raw(H={cfg['horizon']}): {mean_raw:.1f} | Norm(1k): {norm_score:.1f}"

    print("\n" + "="*70)
    print(f"BENCHMARK RESULTS: {args.env} ({args.level})")
    print("="*70)
    print(f"{'Metric':<25} | {'Viol. Rate (%)':<15} | {'Mean Mag.':<20}")
    print("-" * 70)
    print(f"{'Safety (S)':<25} | {safe_rate:<15.2f} | {safe_mag:<20.4f} (Max)")
    print(f"{'Admissibility (A)':<25} | {adm_rate:<15.2f} | {adm_mag:<20.4f} (Max)")
    print(f"{'Dynamics (D)':<25} | {dyn_rate:<15.2f} | {dyn_mag:<20.4f} (0.5*||g||^2)")
    print("-" * 70)
    print(f"{'VALID PLAN RATE':<25} | {success_rate:<15.2f} | (S+A+D Satisfied)")
    print(f"{'TASK PERFORMANCE':<25} | {reward_str}")
    print("="*70)

if __name__ == "__main__":
    main()