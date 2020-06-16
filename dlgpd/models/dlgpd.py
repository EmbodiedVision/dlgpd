from torch import nn

from ..utils.logging import log_gp_hyperparameters


class DLGPDModel(nn.Module):
    def __init__(self, vae, gp, reward_model):
        super(DLGPDModel, self).__init__()
        self.vae = vae
        self.gp = gp
        self.reward_model = reward_model

    def log_hyperparameters(self, summary_writer, step):
        log_gp_hyperparameters(self.gp, summary_writer, step)
        reward_gp = self.reward_model.gp
        for k in range(len(reward_gp.inner_kernel.lengthscale[0])):
            ls = reward_gp.inner_kernel.lengthscale[0, k].item()
            summary_writer.add_scalar(f"reward_model/lengthscale_{k}", ls, step)
        noise = reward_gp.likelihood.noise.item()
        summary_writer.add_scalar(f"reward_model/noise", noise, step)
        outputscale = reward_gp.scale_kernel.outputscale.item()
        summary_writer.add_scalar(f"reward_model/outputscale", outputscale, step)


class DLGPDOptimizer(object):
    def __init__(self, vae_optim, gp_optim, reward_model_optim):
        self.vae_optim = vae_optim
        self.gp_optim = gp_optim
        self.reward_model_optim = reward_model_optim
        self._all = [vae_optim, gp_optim, reward_model_optim]
        self._all_except_reward = [vae_optim, gp_optim]

    def zero_grad(self, except_reward=False):
        if except_reward:
            [o.zero_grad() for o in self._all_except_reward]
        else:
            [o.zero_grad() for o in self._all]

    def step(self):
        [o.step() for o in self._all]

    def state_dict(self):
        return {
            "vae_optim": self.vae_optim.state_dict(),
            "gp_optim": self.gp_optim.state_dict(),
            "reward_model_optim": self.reward_model_optim.state_dict(),
        }
