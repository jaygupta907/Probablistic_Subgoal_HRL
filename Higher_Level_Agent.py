import torch 
from SAC import SAC_Agent
import numpy
import Arguments

higher_agent_args = Arguments.Higher_level_args()
VAE_args = Arguments.VAE_args()
class Higher_Agent():
    def __init__(self,env,writer):
        self.env = env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.latent_dim_high = 1
        self.latent_dim_low = -1
        self.action_scale_high = torch.tensor((self.latent_dim_high - self.latent_dim_low) / 2.0, dtype=torch.float32)
        self.action_bias_high = torch.tensor((self.latent_dim_high + self.latent_dim_low) / 2.0, dtype=torch.float32)
        self.obs_dim =env.observation_space['observation'].shape[0]
        self.action_dim=VAE_args.latent_dim
        self.goal_dim =  env.observation_space['desired_goal'].shape[0]
        self.writer = writer


    def init_agent(self):
        higher_agent = SAC_Agent(Actor_args={'obs_dim': self.obs_dim,
                                            'goal_dim': self.goal_dim,
                                            'action_dim': self.action_dim,
                                            'action_scale': self.action_scale_high, 'action_bias': self.action_bias_high,
                                            'level':'higher'},
                                QNetwork_args={'obs_dim': self.obs_dim,
                                            'goal_dim': self.goal_dim,
                                            'action_dim':  self.action_dim},
                                device=self.device, q_lr=higher_agent_args.q_lr,
                                policy_lr=higher_agent_args.policy_lr,
                                autotune=higher_agent_args.autotune,
                                alpha=higher_agent_args.alpha,
                                gamma=higher_agent_args.gamma,
                                target_network_frequency=higher_agent_args.target_network_frequency,
                                tau=higher_agent_args.tau,
                                action_space=self.env.action_space,
                                batch_size =higher_agent_args.batch_size,
                                writer=self.writer)
        return higher_agent