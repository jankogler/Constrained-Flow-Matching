from torchcfm import ExactOptimalTransportConditionalFlowMatcher
from classes import DoublePendulumTraj, ConstrainedVelocityField
import time
import torch
import numpy as np
torch.manual_seed(42)
pend = DoublePendulumTraj()

def generate_data():
    test = torch.pi/32 * torch.randn(256,4)
    test[:,1:] -= torch.pi/64
    test[:128,0] += 31*torch.pi/64 
    test[128:,0] -= 33*torch.pi/64
    torch.save(pend.trajectory_ode(test), "double_pendulum/test.pth")
    train = torch.pi/32 * torch.randn(128*5000,4)
    train[:,1:] -= torch.pi/64
    train[:,0] += 31*torch.pi/64
    torch.save(pend.trajectory_ode(train).reshape(-1,84), "double_pendulum/train.pth")

def train_and_eval(model, name):
    optimizer = torch.optim.Adam(model.parameters())
    ot_cfm = ExactOptimalTransportConditionalFlowMatcher()
    losses = []
    mse_in = []
    mse_out = []
    constraints = []
    train = torch.load("double_pendulum/train.pth")
    test = torch.load("double_pendulum/test.pth")
    start = time.time()
    for B in range(5000):
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
        if (B+1) % 500 == 0:
            prior = torch.randn(256,80)
            model_traj = model.inference(prior, test[:,0,:])
            mse_in.append(torch.norm(model_traj[:128] - test[:128], dim=(1,2)).mean().item())
            mse_out.append(torch.norm(model_traj[128:] - test[128:], dim=(1,2)).mean().item())
            if model.constraint != None:
                constraints.append(torch.norm(torch.func.vmap(model.constraint)(model_traj), dim=1).mean().item())
    print("Time", time.time() - start)
    np.save(f"double_pendulum/results/{name}_losses", np.array(losses))
    np.save(f"double_pendulum/results/{name}_msein", np.array(mse_in))
    np.save(f"double_pendulum/results/{name}_mseout", np.array(mse_out))
    if model.constraint != None:
        np.save(f"double_pendulum/results/{name}_constraints", np.array(constraints))

def main():
    train_and_eval(
        ConstrainedVelocityField(
            pend.velocity_constraint,
            lambda t: torch.ones_like(t),
            lambda t: torch.exp(4*t)
        ),
        "vel_T0_exp4"
    )

if __name__ == "__main__": 
  main()