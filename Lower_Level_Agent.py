import torch 
from SAC import SAC_Agent
import numpy
import Arguments

lower_agent_args = Arguments.Lower_level_args()
VAE_args = Arguments.VAE_args()

class Lower_Agent():
    def __init__(self,env,writer):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_scale_low = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        self.action_bias_low = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        self.obs_dim =env.observation_space['observation'].shape[0]
        self.action_dim=env.action_space.shape[0]
        self.goal_dim = VAE_args.latent_dim
        self.writer = writer


    def init_agent(self):
        lower_agent = SAC_Agent(Actor_args={'obs_dim': self.obs_dim,
                                            'goal_dim': self.goal_dim,
                                            'action_dim': self.action_dim,
                                            'action_scale': self.action_scale_low, 'action_bias': self.action_bias_low,
                                            'level':'lower'},
                                QNetwork_args={'obs_dim': self.obs_dim,
                                            'goal_dim': self.goal_dim,
                                            'action_dim':  self.action_dim},
                                device=self.device, q_lr=lower_agent_args.q_lr,
                                policy_lr=lower_agent_args.policy_lr,
                                autotune=lower_agent_args.autotune,
                                alpha=lower_agent_args.alpha,
                                gamma=lower_agent_args.gamma,
                                target_network_frequency=lower_agent_args.target_network_frequency,
                                tau=lower_agent_args.tau,
                                action_space=self.env.action_space,
                                batch_size =lower_agent_args.batch_size,
                                writer = self.writer)
        return lower_agent