import numpy as np
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class critic(nn.Module):
    
    def __init__(self,beta,inp_dims,n_actions,fcl1_dims=256,fcl2_dims=256,name="Critic",checkpoint_dir = "temp/sac"):


        super(critic,self).__init__()

        self.inp_dims = inp_dims
        self.fcl1_dims = fcl1_dims
        self.fcl2_dims = fcl2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+"_sac")
        self.name = name
        

        self.fcl1 =nn.Linear(self.inp_dims[0]  + n_actions,self.fcl1_dims)      #state-action pair evaluation
        self.fcl2 = nn.Linear(self.fcl1_dims,self.fcl2_dims)
        self.q1 = nn.Linear(self.fcl2_dims,1)
        self.optimizer = optim.Adam(self.parameters(),lr=beta)

        self.device  = T.device("cuda:0" if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self,state,action):
        q1_act_val = self.fcl1(T.cat([state,action],dim=1).float())
        q1_act_val=F.relu(q1_act_val)
        q1_act_val = self.fcl2(q1_act_val)
        q1_act_val = F.relu(q1_act_val)

        q1 = self.q1(q1_act_val)


        return q1


    def checkpoint_save(self):
        print("................Saving the checkpoint............")
        T.save(self.state_dict(),self.checkpoint_file)

    def checkpoint_load(self):
        print("................Loading the checkpoint............")
        self.load_state_dict(T.load(self.checkpoint_file))




class value(nn.Module):
    
    def __init__(self,beta,inp_dims,fcl1_dims=256,fcl2_dims=256,name="Value",checkpoint_dir = "temp/sac"):


        super(value,self).__init__()

        self.inp_dims = inp_dims
        self.fcl1_dims = fcl1_dims
        self.fcl2_dims = fcl2_dims
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+"_sac")
        self.name = name
        
        #Neural net
        self.fcl1 =nn.Linear(*self.inp_dims  ,self.fcl1_dims)
        self.fcl2 = nn.Linear(self.fcl1_dims,self.fcl2_dims)
        self.v = nn.Linear(self.fcl2_dims,1)
        self.optimizer = optim.Adam(self.parameters(),lr=beta)

        self.device  = T.device("cuda:0" if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self,state):
        state_val=self.fcl1(state.float())
        state_val= F.relu(state_val)
        state_val=self.fcl2(state_val)
        state_val= F.relu(state_val)
        v=self.v(state_val)
       


        return v


    def checkpoint_save(self):
        print("................Saving the checkpoint............")
        T.save(self.state_dict(),self.checkpoint_file)

    def checkpoint_load(self):
        print("................Loading the checkpoint............")
        self.load_state_dict(T.load(self.checkpoint_file))


class actor(nn.Module):
    
    def __init__(self,alpha,inp_dims,max_act,fcl1_dims=256,fcl2_dims=256,n_actions=2,name="Actor",checkpoint_dir = "temp/sac"):


        super(actor,self).__init__()

        self.inp_dims = inp_dims
        self.fcl1_dims = fcl1_dims
        self.fcl2_dims = fcl2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+"_sac")
        self.name = name
        self.max_act=max_act
        self.reparam_noise = 1e-6
        

        self.fcl1 =nn.Linear(*self.inp_dims  ,self.fcl1_dims)
        self.fcl2 = nn.Linear(self.fcl1_dims,self.fcl2_dims)
        self.mu = nn.Linear(self.fcl2_dims,self.n_actions)
        self.sigma=nn.Linear(self.fcl2_dims,self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)

        self.device  = T.device("cuda:0" if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self,state):
        prb=self.fcl1(state.float())
        prb= F.relu(prb)
        prb=self.fcl2(prb)
        prb= F.relu(prb)
        mu=self.mu(prb)

        sigma=self.sigma(prb)
        #Clamping the value of sigma(in the SAC paper they have used 'min=-20' and 'max= +2')
        sigma = T.clamp(sigma,min=self.reparam_noise,max=1)     #Computationally faster(clamp) than using a sigmoid

        return mu,sigma

    #Gaussian distribution for continuous action space
    def sampling_normal(self,state,reparameterize=True):
        mu,sigma=self.forward(state)
        probs =Normal(mu,sigma)

        if reparameterize:
            actions  = probs.rsample()
        else:

            actions = probs.sample()

        action =T.tanh(actions)*T.tensor(self.max_act).to(self.device)
        log_probs = probs.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1,keepdim=True)


        return action,log_probs

    def checkpoint_save(self):
        print("................Saving the checkpoint............")
        T.save(self.state_dict(),self.checkpoint_file)

    def checkpoint_load(self):
        print("................Loading the checkpoint............")
        self.load_state_dict(T.load(self.checkpoint_file))