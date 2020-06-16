import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred import Ingredient

from . import Belief

vae_ingredient = Ingredient("vae")


def sample(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z


# Define Encoder, Decoder, and combine for VAE
class Encoder(nn.Module):
    def __init__(
        self,
        latent_size,
        img_channels=2 * 3,
        deterministic=False,
        batchnorm=False,
        skip_last_fc=False,
    ):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        # self.img_size = img_size
        self.img_channels = img_channels
        self.deterministic = deterministic
        self.batchnorm = batchnorm
        self.skip_last_fc = skip_last_fc

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        if not self.skip_last_fc:
            self.fc_mu = nn.Linear(2 * 2 * 256, latent_size)
            if self.batchnorm:
                self.fc_batchnorm = nn.BatchNorm1d(latent_size, affine=False)
            self.fc_logsigma = nn.Linear(2 * 2 * 256, latent_size)
        else:
            assert self.latent_size == 1024
            assert self.deterministic
            assert not self.batchnorm

    def forward(self, x):
        *bs, c, h, w = x.shape
        x = x.view(np.prod(bs), c, h, w)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        if self.skip_last_fc:
            # assertions for self.deterministic, self.batchnorm handled in constructor
            return x.view(*bs, -1)
        else:
            mu = self.fc_mu(x).view(*bs, -1)
            if self.batchnorm:
                mu = self.fc_batchnorm(mu)
            logsigma = self.fc_logsigma(x).view(*bs, -1)

            if self.deterministic:
                return mu, torch.ones_like(mu).mul(1e-8).log()
            else:
                return mu, logsigma


class Decoder(nn.Module):
    def __init__(self, latent_size, img_channels=2 * 3):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x, return_logits=False):
        *bs, latent_size = x.shape
        x = x.view(np.prod(bs), latent_size)
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        if return_logits:
            reconstruction = x
        else:
            reconstruction = torch.sigmoid(x)
        reconstruction = reconstruction.view(*bs, *reconstruction.shape[1:])
        return reconstruction


class StackedImageFilter(nn.Module):
    def __init__(
        self,
        latent_size,
        image_channels,
        n_images,
        deterministic=True,
        batchnorm_encoder=False,
        single_image_reconstruction_loss=False,
    ):
        super(StackedImageFilter, self).__init__()
        self.latent_size = latent_size
        # Number of image channels - without stacking!
        self.image_channels = image_channels
        # Number of images to stack
        self.n_images = n_images
        self.deterministic = deterministic
        self.single_image_reconstruction_loss = single_image_reconstruction_loss
        self.image_encoder = Encoder(
            latent_size, n_images * image_channels, deterministic, batchnorm_encoder
        )
        self.image_decoder = Decoder(latent_size, n_images * image_channels)

    @property
    def warmup(self):
        # Number of observations at the beginning of an observation window
        # for which no state can be inferred. For this StackedImageFilter,
        # we need 'n_images' observations for the first state to be inferred.
        # For the first 'n_images - 1' images, we cannot infer a state.
        return self.n_images - 1

    @property
    def reconstruction_loss_observations(self):
        # Number of observations required for computing the reconstruction loss
        # for a single state. For a StackedImageFilter, the reconstruction loss
        # for a single state is computed from 'n_images'.
        return self.n_images

    def _chunk_sequence(self, image_sequence):
        # image_sequence: T x bs x <image_dims>
        image_chunks = image_sequence.unfold(dimension=0, size=self.n_images, step=1)
        # image_chunks: n_windows x bs x <image_dims> x 2
        image_chunks = image_chunks.transpose(-1, -2)
        image_chunks = image_chunks.transpose(-2, -3)
        image_chunks = image_chunks.transpose(-3, -4)
        # image_chunks: n_windows x bs x 2 x <image_dims>
        n_windows, bs, n_images, c, h, w = image_chunks.shape
        image_chunks = image_chunks.reshape(n_windows, bs, n_images * c, h, w)
        # image_chunks: n_windows x bs x n_images*c x h x w
        return image_chunks

    def encode_sequence(self, image_sequence):
        # image_sequence: T x bs x <image_dims>
        image_sequence = image_sequence.cuda()
        image_chunks = self._chunk_sequence(image_sequence)
        # image_chunks: n_windows x bs x n_images*c x h x w
        latents = self.image_encoder(image_chunks)
        # latents: tuple of (n_windows x bs x latent_size)
        if self.deterministic:
            state_mean = latents[0]
            state_std = torch.zeros_like(state_mean)
            state_sample = latents[0]
        else:
            state_mean = latents[0]
            state_std = F.softplus(latents[1] + 0.55) + 0.01
            state_sample = sample(state_mean, sigma=state_std)
        return Belief(state_mean, state_std, state_sample)

    def decode_sequence(self, latents):
        # latents: n_windows x bs x latent_size
        decoded_images = self.image_decoder(latents)
        # decoded_images: n_windows x bs x n_images*c x h x w
        n_windows, bs, _, h, w = decoded_images.shape
        decoded_images = decoded_images.view(
            n_windows, bs, self.n_images, self.image_channels, h, w
        )
        # take last image as decoding
        decoded_images = decoded_images[..., -1, :, :, :]
        return decoded_images

    def reconstruction_loss(self, latents, observation_history, loss_type):
        # latents: bs x latent_size
        # observation_history: n_images x bs x <image_dims>
        assert loss_type in ["bce", "mse"]
        decoded_images = self.image_decoder(latents, return_logits=(loss_type == "bce"))
        bs, _, h, w = decoded_images.shape
        decoded_images = decoded_images.view(
            bs, self.n_images, self.image_channels, h, w
        )
        decoded_images = decoded_images.transpose(0, 1)
        loss_fcn = {"mse": F.mse_loss, "bce": F.binary_cross_entropy_with_logits}[
            loss_type
        ]
        # decoded_images: n_windows x bs x n_images*c x h x w
        return loss_fcn(decoded_images, observation_history, reduction="none")


@vae_ingredient.capture
def build_vae(class_name, latent_size, image_channels, vae_kwargs):
    cls = globals()[class_name]
    obj = cls(latent_size, image_channels, **vae_kwargs)
    return obj
