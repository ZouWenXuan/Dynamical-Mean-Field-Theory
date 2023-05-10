# =============================================================================
# Rate model 
# =============================================================================

import numpy as np

class rate_model:
    def __init__(self, N, g, eta, sigma, nl, T, dt, dist, scale):
        ## Neural structure 
        self.N = N
        self.g = g
        self.eta = eta
        self.sigma = sigma
        
        # non-linear function
        if nl == 'None':
            self.phi = lambda x: x
            self.phid = lambda x: 1
        elif nl == 'ReLU':
            self.phi = lambda x: np.maximum(0, x)
            self.phid = lambda x: np.where(x>0, 1, 0)
        elif nl == 'Tanh':
            self.phi = lambda x: np.tanh(x)
            self.phid = lambda x: 1 - (np.tanh(x))**2
        else:
            raise TypeError("Give a correct nonlinear function(None, tanh, ReLU)!")

        ## Simulations
        self.T = T
        self.dt = dt

        # x0
        self.dist = dist
        self.scale = scale


    def set_coupling(self):
        # coupling matrix
        if self.eta != -1:
            k = np.sqrt((1-self.eta)/(1+self.eta))
            Js_init = np.random.normal(0, 1/np.sqrt(self.N*(1+k**2)), [self.N, self.N])
            Js = np.triu(Js_init, k=1) + (np.triu(Js_init, k=1)).T
            Ja_init = np.random.normal(0, 1/np.sqrt(self.N*(1+k**2)), [self.N, self.N])
            Ja = np.triu(Ja_init, k=1) - (np.triu(Ja_init, k=1)).T
            J = Js + k*Ja
        else:
            Ja_init = np.random.normal(0, 1/np.sqrt(self.N), [self.N, self.N])
            Ja = np.triu(Ja_init, k=1) - (np.triu(Ja_init, k=1)).T
            J = np.copy(Ja)
        return J
    

    def initialize_x0(self):
        # Function to sample initial x0
        if self.dist == "normal":
            x0 = self.scale * np.random.normal(0, 1, self.N)
        elif self.dist == "uniform":
            x0 = self.scale * np.random.uniform(0, 1, self.N)
        else:
            raise TypeError("Give a correct initialization (normal or uniform)!")
        return x0


    def integ_xtraj(self):
        # get the coupling
        J = self.set_coupling()
        
        # save all the x_trajories
        nTime = int(self.T/self.dt)
        x_traj = np.zeros([nTime, self.N])

        # initialized
        x0 = self.initialize_x0()
        x_traj[0, :] = np.copy(x0)
        
        # run the dynamics
        x_temp = x0 * 1
        noise = np.random.normal(0, self.sigma*np.sqrt(self.dt), size=[nTime, self.N])
        for s in range(1, nTime, 1):
            x_current = (1 - self.dt) * x_temp
            x_current += self.dt * self.g * np.dot(J, self.phi(x_temp))
            x_current += noise[s-1,:]
            
            # record and set V
            x_traj[s, :] = np.copy(x_current)
            x_temp = np.copy(x_current)
            print("\rTime: {:.2f}/{:.2f}ms".format((s+1)*self.dt, self.T), end='')
        print('')
        return x_traj, noise
    
    
    def compute_observables(self):
        # run x trajories
        x_traj, noise = self.integ_xtraj()
        
        # m and C
        mx = x_traj.mean(1)
        mp = self.phi(x_traj).mean(1)
        Cx = x_traj.dot(x_traj.T)/self.N
        Cp = self.phi(x_traj).dot(self.phi(x_traj).T)/self.N
        
        # Chi
        if self.sigma!=0:
            Chix = 1/self.sigma**2/self.dt * x_traj.dot(noise.T)/self.N
            Chip = 1/self.sigma**2/self.dt * (self.phid(x_traj)*x_traj).dot(noise.T)/self.N
        else:
            Chix = None
            Chip = None
        return mx, mp, Cx, Cp, Chix, Chip