from torchcfm import ExactOptimalTransportConditionalFlowMatcher
from classes import DoublePendulumTraj, ConstrainedVelocityField
import time
import torch
import numpy as np
torch.manual_seed(42)
pend = DoublePendulumTraj()

def generate_data():
    test = torch.pi/16 * torch.randn(256,4)
    test[:,1:] -= torch.pi/32
    test[:128,0] += 15*torch.pi/32 
    test[128:,0] -= 17*torch.pi/32
    torch.save(pend.trajectory_ode(test), "double_pendulum/test.pth")
    train = torch.pi/16 * torch.randn(128*1000,4)
    train[:,1:] -= torch.pi/32
    train[:,0] += 15*torch.pi/32
    torch.save(pend.trajectory_ode(train).reshape(-1,84), "double_pendulum/train.pth")

def train_and_eval(model, name, constraint_norm=None):
    optimizer = torch.optim.Adam(model.parameters())
    ot_cfm = ExactOptimalTransportConditionalFlowMatcher()
    losses = []
    mse_in = []
    mse_out = []
    constraint = []
    train = torch.load("double_pendulum/train.pth")
    test = torch.load("double_pendulum/test.pth")
    start = time.time()
    for _ in range(10):
        for B in range(1000):
            optimizer.zero_grad()
            x1 = train[128*B:128*(B+1)]
            x0 = torch.randn_like(x1)
            x0[:,:4] = x1[:,:4]
            t, xt, ut = ot_cfm.sample_location_and_conditional_flow(x0, x1)
            model.c = xt[:,:4]
            vt = model(torch.cat((xt[:,4:], t[:,None]), dim=1))
            loss = torch.mean((vt - ut[:,4:])**2)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        prior = torch.randn(256,80)
        model_traj = model.inference(prior, test[:,0,:])
        mse_in.append(((model_traj[:128] - test[:128])**2).mean().item())
        mse_out.append(((model_traj[128:] - test[128:])**2).mean().item())
        if constraint_norm != None:
            constraint.append(constraint_norm(model_traj))
    print(f"Time {name}: {time.time() - start}")
    np.save(f"double_pendulum/results/{name}_losses", np.array(losses))
    np.save(f"double_pendulum/results/{name}_msein", np.array(mse_in))
    np.save(f"double_pendulum/results/{name}_mseout", np.array(mse_out))
    if constraint_norm != None:
        np.save(f"double_pendulum/results/{name}_constr", np.array(constraint))

def main():
    generate_data()
    train_and_eval(
        ConstrainedVelocityField(None, None, None),
        "unconstrained"
    )
    for alpha in range(5):
        train_and_eval(
            ConstrainedVelocityField(
                pend.velocity_a,
                lambda t: torch.ones_like(t),
                lambda t: torch.exp(alpha * t)
            ),
            f"vel_T0_exp{alpha}",
            pend.velocity_norm
        )
        train_and_eval(
            ConstrainedVelocityField(
                pend.energy_a,
                lambda t: torch.ones_like(t),
                lambda t: torch.exp(alpha * t)
            ),
            f"en_T0_exp{alpha}",
            pend.energy_norm
        )
    for T in range(2,9,2):
        train_and_eval(
            ConstrainedVelocityField(
                pend.velocity_a,
                lambda t: torch.where(10*t >= T, torch.ones_like(t), torch.zeros_like(t)),
                lambda t: torch.where(10*t >= T, torch.exp(2 * t), torch.zeros_like(t))
            ),
            f"vel_T{T}_exp2",
            pend.velocity_norm
        )
        train_and_eval(
            ConstrainedVelocityField(
                pend.energy_a,
                lambda t: torch.where(10*t >= T, torch.ones_like(t), torch.zeros_like(t)),
                lambda t: torch.where(10*t >= T, torch.exp(2 * t), torch.zeros_like(t))
            ),
            f"en_T{T}_exp2",
            pend.energy_norm
        )

if __name__ == "__main__": 
  main()