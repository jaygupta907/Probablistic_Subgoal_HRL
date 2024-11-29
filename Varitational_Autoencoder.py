import  Arguments 
import torch
import torch.nn as nn
import torch.optim as optim
from Loss import KL_Divergence

class VAE(nn.Module):
    def __init__(self,hidden_dim_1,hidden_dim_2,hidden_dim_3,input_dim,latent_dim,out_dim,device):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=hidden_dim_1),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim_1,out_features=hidden_dim_2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim_2,out_features=hidden_dim_3),
            nn.LeakyReLU()

        ).double()

        self.mean = nn.Sequential(nn.Linear(in_features=hidden_dim_3,out_features=latent_dim),
                                    nn.Tanh()).double()
        self.log_std = nn.Sequential(nn.Linear(in_features=hidden_dim_3,out_features=latent_dim),
                                    nn.Tanh()).double()

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim,out_features=hidden_dim_3),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim_3,out_features=hidden_dim_2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim_2,out_features=hidden_dim_3),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim_3,out_features=out_dim),           
        ).double()
        self.device = device
        self.LOG_STD_MAX =  1
        self.LOG_STD_MIN = 0
        self.mean_min = -2
        self.mean_max = 2

    def reparameterization(self,mean,std):
        epsilon = torch.randn_like(mean).to(self.device)
        z= mean + std*epsilon
        return z
    
    def forward(self,state):
        hidden_representation = self.encoder(state)
        mean = self.mean(hidden_representation)
        log_std  =self.log_std(hidden_representation)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std+1)
        mean = self.mean_min + 0.5 * (self.mean_max - self.mean_min) * (mean+1)
        std = torch.exp(log_std)
        z = self.reparameterization(mean=mean,std=std)
        output = self.decoder(z)

        return output,z,{'mean':mean,'std':std}
    
class VAE_representation_network():
    def __init__(self,env,args,lower_agent,higher_agent,device,writer):
        self.args = args
        self.lower_agent = lower_agent
        self.higher_agent = higher_agent
        self.env = env
        self.device = device
        self.VAE_network = VAE(self.args.hidden_dim_1,
                               self.args.hidden_dim_2,
                               self.args.hidden_dim_3,
                               self.env.observation_space['observation'].shape[0],
                               self.args.latent_dim,
                               self.env.observation_space['observation'].shape[0],
                               self.args.device).to(self.device)
        self.optimizer = optim.Adam(list(self.VAE_network.parameters()), lr=self.args.lr)
        self.writer = writer
        

    def get_distribution(self,state):
        return self.VAE_network(state)

    def update(self,level,timestep):
        if level=='higher':
            observations,_, _, next_observations,_,_ = self.higher_agent.replay_buffer.sample(self.args.batch_size)
            state_1 = observations
            state_2 = next_observations
        else:
            observations, _, _, next_observations,_,_ = self.lower_agent.replay_buffer.sample(self.args.batch_size)
            state_1 = observations
            state_2 = next_observations  
        representation_1 = self.VAE_network(state_1)
        representation_2 = self.VAE_network(state_2)
        representation_state_1 =representation_1[-1]
        representation_state_2 = representation_2[-1]
        output_1 = representation_1[0]
        output_2 = representation_2[0]
        encoding_1 = representation_1[1]
        encoding_2 = representation_2[1]

        reconstruction_loss = torch.norm(state_1-output_1,dim=1)**2+torch.norm(state_2-output_2,dim=1)**2
        Distribution_1 = {'mean':representation_state_1['mean'],'std':representation_state_1['std']}
        Distribution_2 = {'mean':representation_state_2['mean'],'std':representation_state_2['std']}
        Latent_loss = torch.norm(encoding_1-encoding_2)**2
        if level=='higher':
            Loss = torch.max(torch.zeros_like(Latent_loss).to(self.device),self.args.m-Latent_loss)+reconstruction_loss
        else :
            Loss =Latent_loss+reconstruction_loss
        self.writer.add_scalar("data/Autoencoder_Loss",Loss.mean(), timestep)
        self.optimizer.zero_grad()
        Loss.mean().backward()
        self.optimizer.step()

        