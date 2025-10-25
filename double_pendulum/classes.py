import torch
from torch.func import jacrev, vmap
from torch.nn import Linear, SiLU, Sequential
from torchcfm.utils import torch_wrapper
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from matplotlib import animation
from matplotlib import pyplot as plt

class ConstrainedVelocityField(torch.nn.Module):
    def __init__(self, constraint, nu, rho, unconstrained=False, d=80):
        super(ConstrainedVelocityField, self).__init__()
        self.mlp = Sequential(
            Linear(d+5,4*d), SiLU(),
            Linear(4*d,4*d), SiLU(),
            Linear(4*d,d)
        )
        self.constraint = constraint
        self.nu = nu
        self.rho = rho
        self.c = None
        self.unconstrained = unconstrained

    def forward(self, x):
        assert self.c != None, "need to set initial state"
        # compute unconstrained velocity field
        t = x[:,-1].unsqueeze(-1)
        vt = self.mlp(torch.cat((self.c,x), dim=1))
        if self.unconstrained:
            return vt
        # compute trajectory jacobian and pinv
        with torch.enable_grad():
            traj = torch.cat((self.c, x[:,:-1]), dim=1).detach()
            g = vmap(self.constraint)(traj).detach()
            jac = vmap(jacrev(self.constraint))(traj)[:,:,4:].detach()
            inverse_jac = torch.linalg.pinv(jac)

        # output projected velocity minus normal stabilizer
        projected_vel = vt - self.nu(t) * inverse_jac.bmm(jac.bmm(vt.unsqueeze(-1))).squeeze(-1)
        normal_stab = self.rho(t) * inverse_jac.bmm(g.unsqueeze(-1)).squeeze(-1)
        return projected_vel - normal_stab

    def inference(self, prior, condition):
        with torch.no_grad():
            self.c = condition
            node = NeuralODE(torch_wrapper(self), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            traj = node.trajectory(prior, t_span=torch.linspace(0, 1, 2))
        return torch.cat((condition, traj[1]), dim=1).reshape(condition.shape[0], -1, 4)

class DoublePendulumTraj():
    def __init__(self, Nt=20, tfinal=2.0, M1=2, M2=1, L1=2, L2=1, g=9.81):
        self.Nt = Nt
        self.tfinal = tfinal
        self.tdel = tfinal/Nt
        self.M1 = M1
        self.M2 = M2
        self.L1 = L1
        self.L2 = L2
        self.g = g
    
    def trajectory_ode(self, u0):
        def _derive(t, u):
            theta1, omega1, theta2, omega2 = u.unbind(dim=1)
            
            delta = theta1 - theta2
            denom = (2*self.M1 + self.M2 - self.M2*torch.cos(2*delta))
            
            domega1 = (
                -self.g*(2*self.M1 + self.M2)*torch.sin(theta1)
                - self.M2*self.g*torch.sin(theta1 - 2*theta2)
                - 2*torch.sin(delta)*self.M2*(omega2**2*self.L2 + omega1**2*self.L1*torch.cos(delta))
            ) / (self.L1 * denom)
            
            domega2 = (
                2*torch.sin(delta)*(
                    omega1**2*self.L1*(self.M1 + self.M2)
                    + self.g*(self.M1 + self.M2)*torch.cos(theta1)
                    + omega2**2*self.L2*self.M2*torch.cos(delta)
                )
            ) / (self.L2 * denom)
            
            return torch.stack([omega1, domega1, omega2, domega2], dim=1)

        t = torch.linspace(0.0, self.tfinal, self.Nt+1)
        sol = odeint(_derive, u0, t)
        return sol.permute(1, 0, 2)

    def velocity_constraint(self, x):
        traj = x.reshape(-1, 4)
        loss1 = traj[:-1,0] + self.tdel * traj[:-1,1] - traj[1:,0]
        loss2 = traj[:-1,2] + self.tdel * traj[:-1,3] - traj[1:,2]
        return torch.cat([loss1, loss2])

    def energy_constraint(self, x):
        traj = x.reshape(-1, 4)
        theta1, omega1, theta2, omega2 = traj.unbind(dim=1)
        # Kinetic energy
        KE = 0.5 * self.M1 * (self.L1**2) * omega1**2 + 0.5 * self.M2 * (self.L1**2 * omega1**2 + self.L2**2 * omega2**2 + 2 * self.L1 * self.L2 * omega1 * omega2 * torch.cos(theta1 - theta2))
        # Potential energy
        PE = - self.M1 * self.g * self.L1 * torch.cos(theta1) - self.M2 * self.g * (self.L1 * torch.cos(theta1) + self.L2 * torch.cos(theta2))
        # Total energy at each timestep
        E = KE + PE
        # Conservation loss: squared deviation from initial energy
        return E[1:] - E[0]

    def visualize(self,traj):
        x1 = self.L1 * torch.sin(traj[:, 0]).detach().cpu().numpy()
        y1 = -self.L1 * torch.cos(traj[:, 0]).detach().cpu().numpy()
        x2 = x1 + self.L2 * torch.sin(traj[:, 2]).detach().cpu().numpy()
        y2 = y1 - self.L2 * torch.cos(traj[:, 2]).detach().cpu().numpy()

        plt.figure()
        plt.plot(x1,y1,'.',color = '#0077BE',label = 'mass 1')
        plt.plot(x2,y2,'.',color = '#f66338',label = 'mass 2' )
        plt.legend()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

        fig = plt.figure()
        ax = plt.axes(xlim=(-self.L1-self.L2-0.5, self.L1+self.L2+0.5), ylim=(-self.L1-self.L2-0.5, self.L1+self.L2+0.5))
        line1, = ax.plot([], [], 'o-',color = '#d2eeff',markersize = 12, markerfacecolor = '#0077BE',lw=2, markevery=10000, markeredgecolor = 'k')
        line2, = ax.plot([], [], 'o-',color = '#ffebd8',markersize = 12, markerfacecolor = '#f66338',lw=2, markevery=10000, markeredgecolor = 'k')
        line3, = ax.plot([], [], color='k', linestyle='-', linewidth=2)
        line4, = ax.plot([], [], color='k', linestyle='-', linewidth=2)
        line5, = ax.plot([], [], 'o', color='k', markersize = 10)
        time_template = 'Time = %.1f s'
        time_string = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # animation function.  This is called sequentially
        def animate(i):
            # Motion trail sizes. Defined in terms of indices. Length will vary with the time step, dt. E.g. 5 indices will span a lower distance if the time step is reduced.
            trail1 = 6              # length of motion trail of weight 1 
            trail2 = 8              # length of motion trail of weight 2
            
            dt = self.tdel          # time step
            
            line1.set_data(x1[i:max(1,i-trail1):-1], y1[i:max(1,i-trail1):-1])   # marker + line of first weight
            line2.set_data(x2[i:max(1,i-trail2):-1], y2[i:max(1,i-trail2):-1])   # marker + line of the second weight
            
            line3.set_data([x1[i], x2[i]], [y1[i], y2[i]])       # line connecting weight 2 to weight 1
            line4.set_data([x1[i], 0], [y1[i],0])                # line connecting origin to weight 1
            
            line5.set_data([0, 0], [0, 0])
            time_string.set_text(time_template % (i*dt))
            return  line1, line2, line3, line4, line5, time_string

        anim = animation.FuncAnimation(fig, animate, frames=len(x1), blit=True)
        anim.save('double_pendulum_animation.gif', writer=animation.PillowWriter(fps=1/self.tdel))
        plt.close()
