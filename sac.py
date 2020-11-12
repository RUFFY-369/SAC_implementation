import numpy as np
import os
import torch as T
import torch.nn.functional as F
from replay_buffer import replay_buffer
from nets import actor,critic,value


# Implementation of Soft Actor Critic paper
# :https://arxiv.org/abs/1801.01290



class agent():
    
    def __init__(self,alpha=0.0003,beta=0.0003,inp_dims=[8],env=None,gamma=0.99,n_actions= 2,
                 max_size = 1000000,tau=0.005,l1_size = 256,l2_size = 256,batch_size =256,reward_scale=2):
        

        self.tau =tau
        self.gamma = gamma
        
        self.mem = replay_buffer(max_size,inp_dims,n_actions)

        self.batch_size = batch_size
        self.n_actions = n_actions
        



        self.actor = actor(alpha,inp_dims,n_actions= n_actions,name="Actor",
                           max_act=env.action_space.high)



        self.critic1 = critic(beta,inp_dims,n_actions= n_actions,name="Critic1")
        self.critic2 = critic(beta,inp_dims,n_actions= n_actions,name="Critic2")

        self.value =value(beta,inp_dims,name="Value")
        self.target_val =  value(beta,inp_dims,name="Target_value")

        self.scale  = reward_scale


        self.update_net_params(tau=1)

    def update_net_params(self,tau=None):
        if tau is None:
            tau =self.tau


        target_val_params = self.target_val.named_parameters()
        val_params = self.value.named_parameters()

        target_val_state_dict = dict(target_val_params)
        val_state_dict = dict(val_params)

        for name in val_state_dict:
            val_state_dict[name] =tau*val_state_dict[name].clone() + (1-tau)*target_val_state_dict[name].clone()

        self.target_val.load_state_dict(val_state_dict)



    def action_choose(self,obsv):

        state = T.tensor([obsv]).to(self.actor.device)
        actions, _ =self.actor.sampling_normal(state,reparameterize=False)
        
        return actions.cpu().detach().numpy()[0]
       
        

    def rem_transition(self,state,action,reward,nw_state,done):
        self.mem.transition_store(state,action,reward,nw_state,done)

    def learning(self):

        if self.mem.count_mem < self.batch_size:
            return

        state,action,reward,new_state,done = self.mem.sample_buffer(self.batch_size)
        reward=T.tensor(reward,dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        nw_state = T.tensor (new_state,dtype=T.float).to(self.actor.device)
        state = T.tensor (state,dtype=T.float).to(self.actor.device)
        action= T.tensor (action,dtype=T.float).to(self.actor.device)


        val = self.value(state).view(-1)
        nw_val = self.target_val(nw_state).view(-1)
        nw_val[done]  = 0.0

        actions,log_probs = self.actor.sampling_normal(state,reparameterize = False)
        log_probs  =log_probs.view(-1)
        q1_new_pol = self.critic1.forward(state,actions)
        q2_new_pol = self.critic2.forward(state,actions)
        critic_val = T.min(q1_new_pol,q2_new_pol)
        critic_val = critic_val.view(-1)

        self.value.optimizer.zero_grad()
        val_target = critic_val- log_probs
        val_loss = 0.5* F.mse_loss(val,val_target)
        val_loss.backward(retain_graph = True)
        self.value.optimizer.step()

        actions,log_probs = self.actor.sampling_normal(state,reparameterize=True)
        log_probs  =log_probs.view(-1)
        q1_new_pol = self.critic1.forward(state,actions)
        q2_new_pol = self.critic2.forward(state,actions)
        critic_val = T.min(q1_new_pol,q2_new_pol)
        critic_val = critic_val.view(-1)

        actor_loss = log_probs -  critic_val
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_cap = self.scale*reward + self.gamma*nw_val
        q1_old_pol = self.critic1.forward(state,action).view(-1)
        q2_old_pol = self.critic2.forward(state,action).view(-1)
        critic1_loss = 0.5*F.mse_loss(q1_old_pol,q_cap)
        critic2_loss = 0.5*F.mse_loss(q2_old_pol,q_cap)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_net_params()


    

    def model_load(self):
        print(".........Loading the model..........")
        self.actor.checkpoint_load()
        self.value.checkpoint_load()
        self.critic1.checkpoint_load()
        self.critic2.checkpoint_load()
        self.target_val.checkpoint_load()
 
    def model_save(self):
        print(".............Saving the model............")
        self.actor.checkpoint_save
        self.value.checkpoint_save()
        self.critic1.checkpoint_save()
        self.critic2.checkpoint_save()
        self.target_val.checkpoint_save()

 
