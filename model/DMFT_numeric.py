# =============================================================================
# Numerical solution of DMFT equation
# =============================================================================

import numpy as np

class DMFT_rate_model:
    def __init__(self, g, eta, sigma, nl, T, dt, nTraj_list, nIte_list, damp, threshold, dist, scale):
        ## Neural structure 
        self.g = g
        self.eta = eta
        self.sigma = sigma
        
        # Non-linear function
        if nl == 'None':
            self.phi = lambda x: x
            self.phid = lambda x: 1
        elif nl == 'ReLU':
            self.phi = lambda x: np.maximum(0,x)
            self.phid = lambda x: np.where(x>0, 1, 0)
        elif nl == 'Tanh':
            self.phi = lambda x: np.tanh(x)
            self.phid = lambda x: 1 - (np.tanh(x))**2
        else:
            raise TypeError("Give a correct nonlinear function(None, tanh, ReLU)!")
         
        ## Simulations
        self.T = T
        self.dt = dt
        self.nTime = int(T/dt)
        self.nTraj_list = nTraj_list
        self.nIte_list = nIte_list
        self.damp = damp  
        self.threshold = threshold

        # x0
        self.dist = dist
        self.scale = scale
        

    def integ_Chix_Chip(self, x_trajAA, ChipAA):
        # Funtion to integrate Chi(t,t') dynamics
        # arguments:
            # x_trajAA: nTraj trajectories of the DMFT
            # ChipAA: phi response function chi_phi(t,t')
            
        # returns:
            # ChixAA: averaged response function over these trajectories
            # ChipAA: averaged phi response function over these trajectories
        
        nTraj = x_trajAA.shape[1]
        chix_TrajAAA = np.zeros( (self.nTime, self.nTime, nTraj) )
        chip_TrajAAA = np.zeros( (self.nTime, self.nTime, nTraj) )
        for t2 in range(self.nTime-1):
            # start from Chi(t2+1,t2,:)=1, note that Chi(t2,t2,:)=0
            t1 = t2
            tempChiA = np.ones(nTraj)
            chix_TrajAAA[ t1+1, t2, :] = np.copy( tempChiA )
            chip_TrajAAA[ t1+1, t2, :] = tempChiA * self.phid(x_trajAA[t1+1, :])
            t1 += 1
            while t1<self.nTime-1:
                AdditiveA = self.dt**2*self.eta*self.g**2 * ChipAA[t1,t2:t1].dot(chip_TrajAAA[t2:t1,t2, :])
                tempChiA = (1-self.dt)*tempChiA + AdditiveA
                chix_TrajAAA[t1+1,t2,:] = np.copy( tempChiA )
                chip_TrajAAA[t1+1,t2,:] = tempChiA * self.phid(x_trajAA[t1+1, :])
                t1 += 1
        ChixAA = chix_TrajAAA.mean(axis=2)
        ChipAA = chip_TrajAAA.mean(axis=2)
        return ChixAA, ChipAA
    
    
    def sample_gamma(self, CpAA):
        # Function to sample effective noise variable gamma(t)
        # arguments:
            # Cp_AA: phi correlation function C_phi(t,t')
            
        # returns:
            # gammaAA: trajories of effective noise 
         
        # sample multi-gaussian by eigen decomposition (U \Lambda U^T)
        CgammaAA = self.g**2*CpAA
        CgammaAA += self.sigma**2/self.dt*np.diag(np.ones(self.nTime))
        eigenValues, orthogonalBasis = np.linalg.eigh(CgammaAA)
        positiveEigenValues = np.real(eigenValues) * (eigenValues>0)
        tempGaussian = np.random.normal(0, 1, (self.nTime, self.nTraj))
        gammaAA = orthogonalBasis.dot(np.diag(np.sqrt(positiveEigenValues)).dot(tempGaussian))
        return gammaAA
    
    
    def initialize_x0(self):
        # Function to sample initial x0
        if self.dist == "normal":
            x0 = self.scale * np.random.normal(0, 1, self.nTraj)
        elif self.dist == "uniform":
            x0 = self.scale * np.random.uniform(0, 1, self.nTraj)
        else:
            raise TypeError("Give a correct initialization (normal or uniform)!")
        return x0
        
    
    def integ_xtraj(self, CpAA, ChipAA):
        # Fuction to integrate x trajories
        # arguments:
            # CpAA, ChipAA: previous correlation and response 
                
        # returns:
            # x_trajAA: every trajectories
    
        # trajAA will contain every trajectories
        x_trajAA = np.zeros((self.nTime, self.nTraj))
        
        # sample the noise
        gammaAA = self.sample_gamma(CpAA)
        
        # Initial conditions
        tempA = self.initialize_x0()
        x_trajAA[0, :] = np.copy(tempA)
    
        # Run the trajories
        for i in range(self.nTime-1):
            tempA = (1-self.dt)*tempA + self.dt*gammaAA[i,:]
            # tempA += self.sigma*np.random.normal(0, np.sqrt(self.dt), self.nTraj)
            tempA += self.dt**2*self.eta*self.g**2 * ChipAA[i,:i].dot(self.phi(x_trajAA[:i,:]))
            x_trajAA[i+1,:] = np.copy(tempA)
        
        return x_trajAA
        
        
    def main_loop(self, record_traj=False):
        # Fuction to iterate C and Chi to convergence
        # returns:
            # mxA, mpA: mean function for x and phi
            # CxAA, CpAA: correlation function for x and phi
            # ChixAA, ChipAA: response function for x and phi 
            
        # Construct trajories matrix (if needed)
        if record_traj:
            mx_traj, Cx_traj, ChixInt_traj = [], [], []
            mp_traj, Cp_traj, ChipInt_traj = [], [], []
            
        # Initialize guess for correlation and response function
        tempmxA = np.zeros(self.nTime)
        tempmpA = np.zeros(self.nTime)
        tempCxAA = 0.1*np.diag(np.ones(self.nTime))
        tempCpAA = 0.1*np.diag(np.ones(self.nTime))
        tempChixAA = np.diag(np.ones(self.nTime-1), k=-1)
        tempChipAA = np.diag(np.ones(self.nTime-1), k=-1)

        if record_traj:
            mx_traj.append(np.copy(tempmxA))
            mp_traj.append(np.copy(tempmpA))
            Cx_traj.append(np.copy(np.diag(tempCxAA)))
            Cp_traj.append(np.copy(np.diag(tempCpAA)))
            ChixInt_traj.append(np.copy(tempChixAA.sum(0)*self.dt))
            ChipInt_traj.append(np.copy(tempChipAA.sum(0)*self.dt))

        # Iterate the update of observables
        Ite_count = 0
        nIte_Steps = len(self.nIte_list)     
        for i in range(nIte_Steps):  
            self.nIte, self.nTraj = self.nIte_list[i], self.nTraj_list[i] 
            for j in range(self.nIte):
                Ite_count += 1
                # integrate the x trajories
                x_trajAA = self.integ_xtraj(tempCpAA, tempChipAA)
                
                # update the observables m, C
                new_mxA = self.damp*tempmxA + (1-self.damp)*x_trajAA.mean(1)
                new_mpA = self.damp*tempmpA + (1-self.damp)*self.phi(x_trajAA).mean(1)
                new_CxAA = self.damp*tempCxAA + (1-self.damp)*x_trajAA.dot(x_trajAA.T)/self.nTraj 
                new_CpAA = self.damp*tempCpAA + (1-self.damp)*self.phi(x_trajAA).dot(self.phi(x_trajAA).T)/self.nTraj 
                
                
                # Response function
                new_ChixAA = np.zeros((self.nTime, self.nTime))
                new_ChipAA = np.zeros((self.nTime, self.nTime))
                nTraj_Bund = int(1e9/self.nTime**2)
                nBundles = self.nTraj//nTraj_Bund
                count_Bundles = 0
                for bund in range(nBundles):
                    count_Bundles += 1
                    ChixAA_Bund, ChipAA_Bund = self.integ_Chix_Chip(x_trajAA[:, bund*nTraj_Bund:(bund+1)*nTraj_Bund], tempChipAA)
                    new_ChixAA += ChixAA_Bund
                    new_ChipAA += ChipAA_Bund
                if nBundles*nTraj_Bund < self.nTraj:
                    count_Bundles += 1
                    ChixAA_Bund, ChipAA_Bund = self.integ_Chix_Chip(x_trajAA[:, nBundles*nTraj_Bund:], tempChipAA)
                    new_ChixAA += ChixAA_Bund
                    new_ChipAA += ChipAA_Bund
                
                new_ChixAA = self.damp*tempChixAA + (1-self.damp)*new_ChixAA/count_Bundles
                new_ChipAA = self.damp*tempChipAA + (1-self.damp)*new_ChipAA/count_Bundles
                
                
                # convergence indicator
                normDifCx = np.trace((tempCxAA-new_CxAA).dot((tempCxAA-new_CxAA).T)) / self.nTime**2
                normDifChix = np.trace((tempChixAA-new_ChixAA).dot((tempChixAA-new_ChixAA).T)) / (self.nTime*(self.nTime-1)/2)
        
                
                # If the step is small enough, stop the loop
                print("Iteration {}, norm diff C: {:.1e}, norm diff R: {:.1e}".format(Ite_count, normDifCx, normDifChix))
                if normDifCx < self.threshold:
                    if not record_traj:
                        return new_mxA, new_mpA, new_CxAA, new_CpAA, new_ChixAA, new_ChipAA
                    else:
                        Traj_dict = dict(mx=np.array(mx_traj), mp=np.array(mp_traj),
                                         Cx=np.array(Cx_traj), Cp=np.array(Cp_traj),
                                         Chix=np.array(ChixInt_traj), Chip=np.array(ChipInt_traj))
                        return new_mxA, new_mpA, new_CxAA, new_CpAA, new_ChixAA, new_ChipAA, Traj_dict
                
                # Else, continue new iterations
                tempmxA = np.copy( new_mxA )
                tempmpA = np.copy( new_mpA )
                tempCxAA = np.copy( new_CxAA )
                tempCpAA = np.copy( new_CpAA )
                tempChixAA = np.copy( new_ChixAA )
                tempChipAA = np.copy( new_ChipAA )
                
                if record_traj:
                    mx_traj.append(np.copy(tempmxA))
                    mp_traj.append(np.copy(tempmpA))
                    Cx_traj.append(np.copy(np.diag(tempCxAA)))
                    Cp_traj.append(np.copy(np.diag(tempCpAA)))
                    ChixInt_traj.append(np.copy(tempChixAA.sum(0)*self.dt))
                    ChipInt_traj.append(np.copy(tempChipAA.sum(0)*self.dt))
                
    
        print('Did not converge!')
        if not record_traj:
            return new_mxA, new_mpA, new_CxAA, new_CpAA, new_ChixAA, new_ChipAA
        else:
            Traj_dict = dict(mx=np.array(mx_traj), mp=np.array(mp_traj),
                             Cx=np.array(Cx_traj), Cp=np.array(Cp_traj),
                             Chix=np.array(ChixInt_traj), Chip=np.array(ChipInt_traj))
            return new_mxA, new_mpA, new_CxAA, new_CpAA, new_ChixAA, new_ChipAA, Traj_dict