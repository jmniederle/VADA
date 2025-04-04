from typing import Union, Optional
import random

import torch.nn as nn
import torch
import torch.distributions as D
import numpy as np

from src.models import flow


class e_y(nn.Module):
    """
    Process kmers in Nanopore window by embedding each kmer and returning weighted sum of kmer embeddings
    """

    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_embeddings=4 ** 5 + 1, embedding_dim=emb_dim, padding_idx=0)

    def forward(self, y):
        """
        Forward for y conditioning embedding

        Args:
            y: tensor with idx's of one-hot-encoded kmers, shape [batch_size, window_len]

        Returns:
            aggregated conditioning tensor, shape [batch_size, emb_dim]

        """
        embedded = self.embedding(y)
        return torch.mean(embedded, dim=1)


# Encoder for Z_y q(Z_y,t|x_t)
class q_z_y(nn.Module):
    """
    h_dim --> refers to the MAXIMUM number of channels in conv
    """
    def __init__(self, h_dim=64, z_dim=64,
                 activation: nn.Module = nn.ReLU,
                 norm: bool = True,
                 ch_mults=(2, 2, 2, 2),
                 n_blocks: int = 2,
                 use_s_1_filter: bool = True,
                 ):
        super().__init__()

        activation = getattr(nn, activation) if isinstance(activation, str) else activation

        # Number of resolutions
        n_resolutions = len(ch_mults)

        if h_dim % np.prod(ch_mults) != 0:
            raise ValueError(f"h_dim ({h_dim}) is not divisible by product of channel multipliers, "
                             f"choose different value.")

        if n_blocks == 0:
            raise ValueError("n_blocks of encoder must be at least 1 for increasing channels")

        n_channels_initial = int(h_dim / np.prod(ch_mults))

        self.pred_size = np.prod([2 for _ in range(n_resolutions)])

        # Number of channels
        out_channels = in_channels = n_channels_initial

        # Number of channels
        in_channels = out_channels

        # Start with one conv to ensure channels match if needed
        if in_channels != 1:
            if use_s_1_filter:
                self.initial = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=1, padding=0)

            else:
                self.initial = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)

        else:
            self.initial = nn.Identity()

        # Create sequential object with layers, decreasing resolution
        self.res_down_sample_net = nn.Sequential()

        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                self.res_down_sample_net.append(
                    ResidualBlock(
                        in_channels,
                        out_channels,
                        activation=activation(),
                        norm=norm
                    ))
                in_channels = out_channels

            # # Down sample at all resolutions except the last
            # if i < n_resolutions - 1:
            #     self.encoder_net.append(Downsample(in_channels))

            # Down sample ALL resolutions, therefore ending up with spatial dim=1 at the end
            self.res_down_sample_net.append(Downsample(in_channels))

        n_dim_flat = int(self.pred_size / np.prod(ch_mults[:-1]))

        self.mu_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            activation(),
            nn.Linear(h_dim, z_dim))

        self.sigma_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            activation(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

    def forward(self, x_t):
        """
        Process window of x_t to produce latent Z_y
        Args:
            x_t: window of raw signal, shape [batch_size, window_length]

        Returns:

        """

        x_t = x_t.unsqueeze(1)

        h = self.initial(x_t)  # Shape h: [batch_size, min_channels, pred_size]

        h = self.res_down_sample_net(h)

        h = torch.flatten(h, start_dim=1)

        mu = self.mu_head(h)
        sigma = self.sigma_head(h)

        # Add a small constant to avoid diagonal becoming zero
        sigma = sigma + 1e-7

        return D.MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(sigma))


# Conditional prior for z_y P(z_y,t|y_t)
class p_z_y(nn.Module):
    def __init__(self, y_dim=64, h_dim=64, z_dim=64, n_layers=2, activation=nn.ReLU):
        super().__init__()

        activation = getattr(nn, activation) if isinstance(activation, str) else activation

        self.fc_net = nn.Sequential(
            nn.Linear(y_dim, h_dim),
            activation())

        for _ in range(n_layers-1):
            self.fc_net.append(nn.Linear(h_dim, h_dim))
            self.fc_net.append(activation())

        self.mu_head = nn.Linear(h_dim, z_dim)
        self.sigma_head = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

    def forward(self, y_t):
        """
        Process aggregated k-mers embedding and output conditional prior
        Args:
            y_t: aggregated k-mers, shape [batch_size, embedding_dim]

        Returns:
        """

        h = self.fc_net(y_t)
        mu = self.mu_head(h)
        sigma = self.sigma_head(h)
        return D.MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(sigma))
    
class p_z_y_flow(nn.Module):
    """ 
    Conditional prior which uses real NVP flow
    """
    def __init__(
            self,
            y_dim:int, 
            h_dim:int, 
            z_dim:int, 
            n_embedding_layers:int,
            n_flow_layers:int,
            n_mlp_layers:int,
            flow_hidden_size:int,
            normalize_mlp_inputs:bool=True,
            activation:Union[str, type[nn.Module]]=nn.CELU
            ):
        """ 
        Args:
            y_dim: number of dimensions of the embedding y
            h_dim: hidden size of the first MLP that processes y
            z_dim: dimensionality of the latent space
            n_embedding_layers: number of layers used in the MLP that processes y
            n_flow_layers: number of bijective layers used in the real NVP flow
            n_mlp_layers: number of layers used in the mlps for the t- and s-nets in the NVP flow
            flow_hidden_size: hidden size of the t- and s-nets
            normalize_mlp_inputs: whether the input to the t- and s-net should be normalized using LayerNorm
        """
        super().__init__()
        self.embedding_net = flow.get_mlp(
            in_size=y_dim,
            out_size=h_dim,
            hidden_size=h_dim,
            number_of_hidden_layers=n_embedding_layers-2,  # this is the convention in p_z_y, so we stick to it
            activation=activation
        )
        self.flow_net = flow.ConditionalFlow.get_MLP_flow(
            num_latent_dims=z_dim,
            context_size=h_dim,
            num_flow_layers=n_flow_layers,
            num_hidden_layers=n_mlp_layers-2,  # this is the convention above, so we stick to it
            hidden_size=flow_hidden_size,
            normalize_mlp_inputs=normalize_mlp_inputs,
            activation=activation
        )
    
    def forward(self, y_t:torch.Tensor)->flow.ConditionedFlow:
        context = self.embedding_net(y_t)
        return self.flow_net(context)


class ResidualBlock(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (nn.Module): Activation function to use.
        norm (bool): Whether to use normalization.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: nn.Module = nn.ReLU(),
            norm: bool = True,
            **kwargs
    ):
        super().__init__()
        self.activation: nn.Module = activation

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, **kwargs)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, **kwargs)

        self.shortcut = nn.Identity()

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection, this happens when reducing channels in last conv of block
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, **kwargs)
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.BatchNorm1d(in_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        # Second convolution layer
        h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$
    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2, padding=1,
                              **kwargs)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$
    Args:
        n_channels (int): Number of channels in the input and output.
        n_channels_out (int, None): Number of channels to output, if None, return same amount of channels

    """

    def __init__(self, n_channels: int, n_channels_out: Union[int, None] = None, **kwargs):
        super().__init__()
        if n_channels_out is None:
            n_channels_out = n_channels

        self.conv = nn.ConvTranspose1d(in_channels=n_channels, out_channels=n_channels_out, kernel_size=2, stride=2,
                                       padding=0, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


# Decoder
class p_x_z(nn.Module):
    def __init__(self,
                 z_dim: int = 64,
                 activation: type[nn.Module] = nn.ReLU,
                 scale: float = 1 / np.sqrt(2),
                 h_dim: int = 64,
                 norm: bool = True,
                 ch_mults=(2, 2, 2, 2),
                 n_blocks: int = 2,
                 use_s_1_filter: bool = True,
                 verbose: bool = False
                 ):
        super().__init__()

        activation = getattr(nn, activation) if isinstance(activation, str) else activation

        self.verbose = verbose

        # Number of resolutions
        n_resolutions = len(ch_mults)
        n_channels = h_dim

        self.pred_size = np.prod([2 for _ in range(n_resolutions)])

        self.fc_in = nn.Sequential(
            nn.Linear(z_dim + self.pred_size, h_dim),
            activation()
        )

        # Number of channels
        out_channels = in_channels = n_channels

        # Create list of layers, increasing resolution
        # decoder_net = []
        self.decoder_net = nn.Sequential()

        # Number of channels
        in_channels = out_channels

        self.decoder_net.append(Upsample(h_dim))

        # For each resolution
        for i in reversed(range(n_resolutions)):
            # n_blocks with the same resolution
            out_channels = in_channels
            for _ in range(max(n_blocks-1, 0)):
                self.decoder_net.append(ResidualBlock(
                    in_channels,
                    out_channels,
                    activation=activation(),
                    norm=norm
                ))

            if n_blocks > 0:
                # Final at each resolution block reduces number of channel
                out_channels = in_channels // ch_mults[i]
                self.decoder_net.append(ResidualBlock(in_channels, out_channels, activation=activation(), norm=norm))

                in_channels = out_channels

                # Up sample all resolution but the last
                if i > 0:
                    self.decoder_net.append(Upsample(in_channels))

            # If n_blocks is 0, use no residual blocks but only use Upsampling TConvs, use these to decrease channels
            else:
                out_channels = in_channels // ch_mults[i]
                if i > 0:
                    self.decoder_net.append(Upsample(n_channels=in_channels, n_channels_out=out_channels))
                    in_channels = out_channels

        # Combine modules
        # self.decoder_net = nn.ModuleList(decoder_net)

        if use_s_1_filter:
            self.final = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

        else:
            self.final = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)

        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, z_y, x_prev):
        h_in = torch.cat((z_y, x_prev), dim=1)
        h = self.fc_in(h_in)
        h = h.unsqueeze(2)
        #
        # for m in self.decoder_net:
        #     h = m(h)
        h = self.decoder_net(h)

            # if self.verbose:
            # print(f"Module: {m.__class__.__name__}, resulting shape: {h.shape}\n")

        h = self.final(h)

        # if self.verbose:
        # print(f"{self.final}\n Final output shape: {h.shape})")

        x_t_mu = h.squeeze(1)
        return D.Normal(loc=x_t_mu, scale=self.scale)


class r_y(nn.Module):
    def __init__(self, z_y_dim, y_dim, h_dim=64, n_layers=2, activation=nn.ReLU):
        super().__init__()

        activation = getattr(nn, activation) if isinstance(activation, str) else activation

        self.fc_net = nn.Sequential(
            nn.Linear(z_y_dim, h_dim),
            activation()
        )

        for _ in range(n_layers - 1):
            self.fc_net.append(nn.Linear(h_dim, h_dim))
            self.fc_net.append(activation())

        self.final = nn.Linear(h_dim, y_dim)

    def forward(self, z_y_sample):
        h = self.fc_net(z_y_sample)
        out = self.final(h)
        return out


class VADA(nn.Module):
    """
    Variational Autoregressive DNA-conditioned Autoencoder

    Args:
        y_emb_dim: number of dimensions for embedding y
        x_pred_size: size of prediction window
        h_dim: number of dimensions for linear layers and max channels for convolutions in encoder/decoder
        z_dim: number of dimensions for latent variable z
        teacher_forcing_ratio: probability of teacher forcing, default 100%
        beta_kl_y: coefficient to use for KL loss term
        beta_reconstr: coefficient to use for reconstruction loss term
        beta_aux: coefficient to use for auxiliary loss term
        device: device model is on
        activation: activation function to use
        n_layers_prior: number of layers for the conditional prior
        n_blocks_encoder: number of residual blocks for the encoder per resolution
        n_blocks_decoder: number of residual blocks for the decoder per resolution
        n_layers_auxiliary: number of layers for the auxiliary regressor
        use_s_1_filter_decoder: use length 1 or length 3 (if False) conv filter to get correct number of out channels
        use_s_1_filter_encoder: use length 1 or length 3 (if False) conv filter to get correct number of in channels
        norm:
        use_flow: whether to use flow for the p_z_y conditional prior
        flow_kwargs: kwargs to be passed to p_z_y_flow if use_flow is True

    """
    def __init__(
            self,
            y_emb_dim: int,
            x_pred_size: int,
            h_dim: int, z_dim: int,
            teacher_forcing_ratio: float = 1.0,
            beta_kl_y: float = 1.0,
            beta_reconstr: float = 1.0,
            beta_aux: float = 1.0,
            device: Union[str, torch.device] = 'cpu',
            activation: Union[str, nn.Module] = nn.ReLU,
            n_layers_prior: int = 2,
            n_blocks_encoder: int = 2,
            n_blocks_decoder: int = 2,
            n_layers_auxiliary: int = 2,
            use_s_1_filter_decoder: bool = True,
            use_s_1_filter_encoder: bool = True,
            norm: bool = True,
            use_flow: bool = False,
            flow_kwargs: Optional[dict] = None,
            ):


        # Embedding layer conditioning
        super().__init__()

        assert x_pred_size % 2 == 0

        self.e_y = e_y(emb_dim=y_emb_dim)

        # Decide channel multipliers according two x_pred_size
        print(f"Number of resolutions in encoder and decoder: {int(np.log2(x_pred_size))}")
        mults = [2 for _ in range(int(np.log2(x_pred_size)))]

        # Encoders
        self.q_z_y = q_z_y(
            h_dim=h_dim, 
            z_dim=z_dim, 
            norm=norm, 
            n_blocks=n_blocks_encoder,
            use_s_1_filter=use_s_1_filter_encoder, 
            activation=activation, 
            ch_mults=tuple(mults)
            )

        # Priors
        self.use_flow = use_flow
        if use_flow:
            self.p_z_y = p_z_y_flow(
                y_dim=y_emb_dim,
                h_dim=h_dim,
                z_dim=z_dim,
                n_embedding_layers=n_layers_prior,
                activation=activation,
                **flow_kwargs
            )
        else:
            self.p_z_y = p_z_y(y_dim=y_emb_dim, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers_prior, activation=activation)


        self.p_x_z = p_x_z(
            z_dim=z_dim, 
            h_dim=h_dim, 
            ch_mults=tuple(mults), 
            norm=norm,
            n_blocks=n_blocks_decoder, 
            use_s_1_filter=use_s_1_filter_decoder, 
            activation=activation
            )

        # Auxiliary Regressor
        self.r_y = r_y(z_y_dim=z_dim, y_dim=y_emb_dim, h_dim=h_dim, n_layers=n_layers_auxiliary, activation=activation)

        # Loss components
        self.bce_loss = nn.BCELoss()
        self.mse = nn.MSELoss(reduction='none')

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.x_pred_size = x_pred_size

        self.device = device

        # Weights for loss terms
        self.beta_kl_y = beta_kl_y
        self.beta_reconstr = beta_reconstr
        self.beta_aux = beta_aux

    def encode(self, x_t):
        """
        Encode batch of Nanopore signals for timepoint t, x_t

        Args:
            x_t: batch of x_t

        Returns:
            Latent distributions for z_y, and z_eps

        """
        z_y_encoding_dist = self.q_z_y(x_t)
        return z_y_encoding_dist

    def decode(self, z_y_sample, x_prev):
        """
        Decode a z_y and z_eps SAMPLE and the previous predicted Nanopore signal x_prev into next predicted
        Nanopore signal window

        Args:
            z_y_sample: batch of z_y samples
            x_prev: previously predicted window

        Returns:

        """
        x_t_pred = self.p_x_z(z_y_sample, x_prev)
        return x_t_pred

    def forward(self, x_t, x_prev, y_t):
        """
        Forward pass of model

        Args:
            y_t: tensor with idx's of one-hot-encoded kmers, shape [batch_size, window_len]
            x_t: reference Nanopore signal for current prediction window
            x_prev: previous window of Nanopore signal

        Returns:
            p(x_t|z_y_t, z_eps_t),
            (q(z_y_t|x_t), q(z_eps_t|x_t)),
            (z_y_t, z_eps_t)
            (p(z_y_t|y_t), p(z_eps_t))

        """
        # Encode
        z_y_encoding_dist = self.encode(x_t)  # q(z_y|x), q(z_eps|x)

        # Re-parameterization trick
        z_y_sample = z_y_encoding_dist.rsample()

        # Decode
        x_t_decoding_dist = self.decode(z_y_sample, x_prev)

        # Embed conditioning
        y_t_prime = self.e_y(y_t)

        # Prior
        z_y_prior = self.p_z_y(y_t_prime)

        # y_hat = self.classify(z_y_sample)
        y_t_prime_hat = self.r_y(z_y_sample)

        return x_t_decoding_dist, z_y_encoding_dist, z_y_sample, z_y_prior, y_t_prime, y_t_prime_hat

    def compute_losses_timestep(self, x_t, x_t_decoding_dist, z_y_encoding_dist,
                                z_y_sample, z_y_prior, y_t_prime, y_t_prime_hat):
        """
        Compute the losses for a window of Nanopore signal and k-mer conditioning

        Args:
            x_t:
            y_t:

        Returns:

        """
        # KL loss
        if self.use_flow:
            dkl_z_y = z_y_encoding_dist.log_prob(z_y_sample) - z_y_prior.log_prob(z_y_sample)
        else:
            dkl_z_y = D.kl.kl_divergence(z_y_encoding_dist, z_y_prior)

        # Reconstruction
        reconstruction_loss = -x_t_decoding_dist.log_prob(x_t).mean(1)  # mean over prediction window

        # Auxiliary Regression Loss
        aux_loss = self.mse(y_t_prime_hat, y_t_prime).mean(1)

        return x_t_decoding_dist, reconstruction_loss, dkl_z_y, aux_loss

    def compute_loss_full_seq(self, x, y, aggr_fn=torch.mean):
        seq_len = x.shape[1]
        batch_size = x.shape[0]

        reconstruction_losses = []
        dkl_z_y_losses = []
        aux_losses = []
        mask_non_zero = []

        x_prev = torch.zeros((batch_size, self.x_pred_size)).to(self.device)

        max_t = int(seq_len / self.x_pred_size) * self.x_pred_size  # Only include full windows

        for t in range(0, max_t, self.x_pred_size):
            x_t = x[:, t:t + self.x_pred_size]
            y_t = y[:, t:t + self.x_pred_size]

            x_t_decoding_dist, reconstruction_loss, dkl_z_y, aux_loss \
                = self.compute_losses_timestep(x_t, *self(x_t, x_prev, y_t))

            reconstruction_losses.append(reconstruction_loss)
            dkl_z_y_losses.append(dkl_z_y)
            aux_losses.append(aux_loss)

            # Create mask to only include timepoints where at least half of the window contained a complete k-mer
            non_zero = torch.sum(y_t == 0, dim=1) <= 0.5 * self.x_pred_size
            mask_non_zero.append(non_zero)

            x_t_pred = x_t_decoding_dist.rsample()

            teacher_force = (random.random() < self.teacher_forcing_ratio)
            x_prev = x_t if teacher_force else x_t_pred

        non_zero = torch.stack(mask_non_zero)
        reconstr = aggr_fn(torch.masked_select(torch.stack(reconstruction_losses), non_zero))
        kl_z_y = aggr_fn(torch.masked_select(torch.stack(dkl_z_y_losses), non_zero))
        aux = aggr_fn(torch.masked_select(torch.stack(aux_losses), non_zero))

        total_loss = self.beta_reconstr * reconstr \
                     + self.beta_kl_y * kl_z_y \
                     + self.beta_aux * aux

        return total_loss, reconstr, kl_z_y, aux

    def compute_loss_parallel(self, x, y, aggr_fn=torch.mean):
        """
        Compute loss using only vectorized operations and no for-loops, this means 100% teacher forcing is applied

        Args:
            x:
            y:
            aggr_fn:

        Returns:

        """
        x_prev = torch.roll(x, shifts=self.x_pred_size, dims=1)
        x_prev[:, 0:self.x_pred_size] = 0

        x_prev_chunks = torch.split(x_prev, self.x_pred_size, dim=1)
        x_chunks = torch.split(x, self.x_pred_size, dim=1)
        y_chunks = torch.split(y, self.x_pred_size, dim=1)

        # Drop the last window if it's not completely filled
        if x_chunks[-1].shape[1] != self.x_pred_size:
            x_prev_chunks = x_prev_chunks[:-1]
            x_chunks = x_chunks[:-1]
            y_chunks = y_chunks[:-1]

        # Shape: [batch_size, num_chunks, pred_size]
        x_prev_w = torch.stack(x_prev_chunks).swapaxes(0, 1)
        x_t_w = torch.stack(x_chunks).swapaxes(0, 1)
        y_t_w = torch.stack(y_chunks).swapaxes(0, 1)

        batch_size, num_chunks = x_t_w.shape[0], x_t_w.shape[1]

        # Reshape tensors to [batch_size * num_chunks, pred_size], such that conv decoder_net can deal with the input
        x_prev_w = x_prev_w.reshape(batch_size * num_chunks, self.x_pred_size)
        x_t_w = x_t_w.reshape(batch_size * num_chunks, self.x_pred_size)
        y_t_w = y_t_w.reshape(batch_size * num_chunks, self.x_pred_size)

        x_t_decoding_dist, reconstruction_loss, dkl_z_y, aux_loss \
            = self.compute_losses_timestep(x_t_w, *self(x_t_w, x_prev_w, y_t_w))

        # Create mask to only include timepoints where at least half of the window contained a complete k-mer
        non_zero = torch.sum(y_t_w == 0, dim=1) <= 0.5 * self.x_pred_size

        reconstr = aggr_fn(torch.masked_select(reconstruction_loss, non_zero))
        kl_z_y = aggr_fn(torch.masked_select(dkl_z_y, non_zero))
        aux = aggr_fn(torch.masked_select(aux_loss, non_zero))

        total_loss = self.beta_reconstr * reconstr \
                     + self.beta_kl_y * kl_z_y \
                     + self.beta_aux * aux

        return total_loss, reconstr, kl_z_y, aux

    def encode_and_split_seq(self, x, y=None):
        x_prev = torch.roll(x, shifts=self.x_pred_size, dims=1)
        x_prev[:, 0:self.x_pred_size] = 0

        x_chunks = torch.split(x, self.x_pred_size, dim=1)

        # Drop the last window if it's not completely filled
        if x_chunks[-1].shape[1] != self.x_pred_size:
            x_chunks = x_chunks[:-1] 

        # Shape: [batch_size, num_chunks, pred_size]
        x_t_w = torch.stack(x_chunks).swapaxes(0, 1)

        batch_size, num_chunks = x_t_w.shape[0], x_t_w.shape[1]

        # Reshape tensors to [batch_size * num_chunks, pred_size], such that conv decoder_net can deal with the input
        x_t_w = x_t_w.reshape(batch_size * num_chunks, self.x_pred_size)
        

        z_y_dist = self.encode(x_t_w)
        z_y = z_y_dist.sample()
        # z_y.reshape(batch_size, num_chunks, self.x_pred_size)

        y_t_w = None
        if y is not None:
            y_chunks = torch.split(y, self.x_pred_size, dim=1)
            y_chunks = y_chunks[:-1]
            y_t_w = torch.stack(y_chunks).swapaxes(0, 1)
            y_t_w = y_t_w.reshape(batch_size * num_chunks, self.x_pred_size)
        return z_y, y_t_w

    def sample(self, y):
        batch_size = y.shape[0]
        seq_len = y.shape[1]
        x_prev = torch.zeros((batch_size, self.x_pred_size)).to(self.device)

        max_t = int(seq_len / self.x_pred_size) * self.x_pred_size  # Only include full windows
        x_output = torch.zeros((batch_size, max_t)).to(self.device)

        for t in range(0, max_t, self.x_pred_size):
            y_t = y[:, t:t + self.x_pred_size]

            # Embed conditioning
            y_t_prime = self.e_y(y_t)

            z_y_prior = self.p_z_y(y_t_prime)
            z_y_sample = z_y_prior.sample()

            # Decode
            x_t_decoding_dist = self.decode(z_y_sample, x_prev)

            x_t = x_t_decoding_dist.loc

            x_output[:, t:t + self.x_pred_size] = x_t

            x_prev = x_t

        return x_output
