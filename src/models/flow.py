from typing import Union

import torch
from torch import nn
from torch import distributions as D

import wandb

EPS = 1e-6

def get_mlp(in_size:int, out_size:int, hidden_size:int, number_of_hidden_layers:int, normalize_input:bool=False, activation:Union[str, type[nn.Module]]=nn.CELU):
    """
    Get an MLP
    :parameter in_size: number of input dimensions
    :parameter out_size: number of output dimensions
    :parameter hidden_size: number of hidden dimensions
    :parameter number_of_hidden_layers: number of layers excluding input and output layer
    """
    activation = getattr(nn, activation) if isinstance(activation, str) else activation
    layers = []
    if normalize_input:
        layers.append(nn.LayerNorm(in_size))  # maybe set bias to False
    layers.append(nn.Linear(in_size, hidden_size))
    layers.append(activation())
    for _ in range(number_of_hidden_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation())
    layers.append(nn.Linear(hidden_size, out_size))
    return nn.Sequential(*layers)


class _InverseConditionalRealNVPLayer(nn.Module):
    def __init__(self, conditional_real_nvp_layer):
        super().__init__()
        self._forward_layer = [
            conditional_real_nvp_layer
            ]  # put it in a list to prevent recursion errors when moving to gpu

    @property
    def forward_layer(self):
        return self._forward_layer[0]

    def transform(self, z_new, h):
        """
        Transform z_new to z_old conditionally on h
        NB this is the transformation in the inverse flow
        """
        total_block_size = z_new.shape[-1]
        if self.forward_layer.constant_block_size is None:
            constant_block_size = (total_block_size // 2) + (total_block_size % 2)
        else:
            constant_block_size = self.forward_layer.constant_block_size

        z_keep, z_new_ = z_new[..., :constant_block_size], z_new[..., constant_block_size:]

        # s = self.forward_layer.s_net(torch.cat([z_keep, h], dim=-1))
        # t = self.forward_layer.t_net(torch.cat([z_keep, h], dim=-1))
        t, s = self.forward_layer.t_and_s_net(z_keep, h)

        z_change = (z_new_ - t) / self.forward_layer.s_activation(s)
        return torch.cat([z_keep, z_change], dim=-1)

    def log_det_jacobian(self, z_new, h):
        """
        Compute the log_det_jacobian of this (inverse flow) transformation at z_new conditionally on h
        """
        return - self.forward_layer.log_det_jacobian(z_new, h)  # only difference is the sign

    def forward(self, z_new, h):
        """
        Compute the transformed version of z_new and the value of the log det jacobian of this transformation at z_new conditionally on h.
        NB this is for the transformation in the inverse flow.
        """
        total_block_size = z_new.shape[-1]
        if self.forward_layer.constant_block_size is None:
            constant_block_size = (total_block_size // 2) + (total_block_size % 2)
        else:
            constant_block_size = self.forward_layer.constant_block_size

        z_keep, z_new_ = z_new[..., :constant_block_size], z_new[..., constant_block_size:]

        # s = self.forward_layer.s_net(torch.cat([z_keep, h], dim=-1))
        # t = self.forward_layer.t_net(torch.cat([z_keep, h], dim=-1))
        t, s = self.forward_layer.t_and_s_net(z_keep, h)

        z_change = (z_new_ - t) / self.forward_layer.s_activation(s)

        if self.forward_layer.use_softplus:
            s = torch.log(self.forward_layer.s_activation(s))
        # log exp is identity
        log_det_jacobian = -torch.sum(s, dim=tuple(range(-self.forward_layer.num_event_dims, 0,
                                                         1)))  # minus sign because we divide by s_activation(s) in the inverse transform
        return torch.cat([z_keep, z_change], dim=-1), log_det_jacobian


class ConditionalRealNVPLayer(nn.Module):
    """ 
    implements RealNVP using channel splitting
    """
    def __init__(
            self, 
            s_net: Union[nn.Module, None] = None, 
            t_net: Union[nn.Module, None] = None,
            constant_block_size=None, 
            use_softplus=False, 
            EPS=EPS, 
            num_event_dims=1,
            t_and_s_net: Union[nn.Module, None] = None
            ):
        super().__init__()
        if t_and_s_net is not None and (s_net is not None or t_net is not None):
            raise ValueError("Cannot specify both t_and_s_net and s_net/t_net")
        if t_and_s_net is None and (s_net is None or t_net is None):
            raise ValueError("Must specify either t_and_s_net or s_net/t_net")
        self._s_net = s_net
        self.t_net = t_net
        self._t_and_s_net = t_and_s_net
        self.use_softplus = use_softplus
        self.s_activation = self._softplus if use_softplus else self._exp
        self.constant_block_size = constant_block_size
        self._EPS = EPS
        self.num_event_dims = num_event_dims
        self.inverse = _InverseConditionalRealNVPLayer(self)

    def t_and_s_net(self, z_keep, h):
        zh = torch.cat([z_keep, h], dim=-1)
        if self._t_and_s_net is not None:
            ts = self._t_and_s_net(zh)
            t, s = ts.chunk(2, dim=-1)
        else:
            t = self.t_net(zh)
            s = self._s_net(zh)
        return t, s

    def s_net(self, z_keep, h):
        zh = torch.cat([z_keep, h], dim=-1)
        if self._t_and_s_net is not None:
            ts = self._t_and_s_net(zh)
            _, s = ts.chunk(2, dim=-1)
        else:
            s = self._s_net(zh)
        return s

    def set_num_event_dims(self, num_event_dims):
        self.num_event_dims = num_event_dims

    def _softplus(self, s):
        return nn.functional.softplus(s) + self._EPS

    def _exp(self, s):
        return torch.exp(s) + self._EPS

    def transform(self, z, h):
        """
        Transform z conditionally on h without computing the log_det_jacobian
        NB this is the transformation in the forward flow
        """
        total_block_size = z.shape[-1]
        if self.constant_block_size is None:
            constant_block_size = (total_block_size // 2) + (total_block_size % 2)
        else:
            constant_block_size = self.constant_block_size

        z_keep, z_change = z[..., :constant_block_size], z[..., constant_block_size:]

        # s = self.s_net(torch.cat([z_keep, h], dim=-1))
        # t = self.t_net(torch.cat([z_keep, h], dim=-1))
        t, s = self.t_and_s_net(z_keep, h)
        z_new = self.s_activation(s) * z_change + t
        return torch.cat([z_keep, z_new], dim=-1)

    def log_det_jacobian(self, z_old, h):
        """
        Compute the log_det_jacobian of this layer at z_old conditionally on h
        """
        total_block_size = z_old.shape[-1]
        if self.constant_block_size is None:
            constant_block_size = (total_block_size // 2) + (total_block_size % 2)
        else:
            constant_block_size = self.constant_block_size
        z_keep = z_old[..., :constant_block_size]
        s = self.s_net(z_keep, h)
        if self.use_softplus:
            s = torch.log(self.s_activation(s))
        # log exp is identity
        return torch.sum(s, dim=tuple(range(-self.num_event_dims, 0, 1)))

    def forward(self, z, h):
        """
        Compute the transformed version of z and the value of the log det jacobian of the transformation at z conditionally on h.
        """
        total_block_size = z.shape[-1]
        if self.constant_block_size is None:
            constant_block_size = (total_block_size // 2) + (total_block_size % 2)
        else:
            constant_block_size = self.constant_block_size

        z_keep, z_change = z[..., :constant_block_size], z[..., constant_block_size:]

        t, s = self.t_and_s_net(z_keep, h)
        z_new = self.s_activation(s) * z_change + t

        if self.use_softplus:
            s = torch.log(self.s_activation(s))
        log_det_jacobian = torch.sum(s, dim=tuple(range(-self.num_event_dims, 0, 1)))
        return torch.cat([z_keep, z_new], dim=-1), log_det_jacobian


class ConditionalFlow(nn.Module):
    def __init__(self, num_event_dims, initial_loc_layer, initial_pre_scale_layer,
                 *layers:ConditionalRealNVPLayer, EPS=EPS):
        super().__init__()
        self.num_event_dims = num_event_dims
        self.layers = nn.ModuleList(layers)
        for layer in self.layers:
            layer.set_num_event_dims(self.num_event_dims)
        self.initial_loc_layer = initial_loc_layer
        self.initial_pre_scale_layer = initial_pre_scale_layer
        self.EPS = EPS


    def forward(self, h:torch.Tensor)->"ConditionedFlow":

        if wandb.run is not None:
            wandb.log({
                'debug/h_norm': h.norm().detach().cpu().item(), 
                'debug/h_max': h.max().detach().cpu().item(), 
                'debug/h_min': h.min().detach().cpu().item()
                }, step=wandb.run.step, commit=False)
        
        initial_loc = self.initial_loc_layer(h)
        initial_scale = nn.functional.softplus(self.initial_pre_scale_layer(h)) + self.EPS
        try:
            initial_distribution = D.Independent(
                D.Normal(initial_loc, initial_scale),
                reinterpreted_batch_ndims=self.num_event_dims
            )
        except ValueError as e:
            print("A ValueError was raised during the forward pass of ConditionalFlow")
            print(f"{torch.any(torch.isnan(initial_loc)).item()=}")
            print(f"{torch.any(torch.isnan(initial_scale)).item()=}")
            print(f"{torch.any(torch.isnan(h)).item()=}")
            print("Parameters of ConditionalFlow layer with nans:")
            for name, param in self.named_parameters():
                if torch.any(torch.isnan(param.data)).item():
                    print("    ", name)
            raise e
        return ConditionedFlow(self, h, initial_distribution)

    @classmethod
    def get_MLP_flow(
            cls,
            num_latent_dims,
            context_size,
            num_flow_layers,
            num_hidden_layers,
            hidden_size,
            use_soft_plus=False,
            normalize_mlp_inputs=False, 
            activation:Union[str, type[nn.Module]]=nn.CELU
    ):
        constant_block_size = num_latent_dims // 2 + (num_latent_dims % 2)
        mlp_out_size = num_latent_dims - constant_block_size
        mlp_in_size = constant_block_size + context_size

        initial_loc_layer = nn.Linear(context_size, num_latent_dims)
        initial_pre_scale_layer = nn.Linear(context_size, num_latent_dims)
        flow_layers = []
        for _ in range(num_flow_layers):
            s_net = get_mlp(mlp_in_size, mlp_out_size, hidden_size, num_hidden_layers,
                            normalize_input=normalize_mlp_inputs, activation=activation)
            t_net = get_mlp(mlp_in_size, mlp_out_size, hidden_size, num_hidden_layers,
                            normalize_input=normalize_mlp_inputs, activation=activation)
            flow_layers.append(ConditionalRealNVPLayer(s_net, t_net, constant_block_size, use_softplus=use_soft_plus))
        return cls(1, initial_loc_layer, initial_pre_scale_layer, *flow_layers)


class ConditionedFlow(D.Distribution):  # TODO either modify this or duplicate this for the layers that work with masking.

    def __init__(
            self, 
            conditional_flow: ConditionalFlow, 
            conditioning: torch.Tensor,
            initial_distribution: D.Distribution
            ):
        super().__init__()
        self.initial_distribution = initial_distribution
        self.conditional_flow = conditional_flow
        self.conditioning = conditioning  # h
        self._event_shape = initial_distribution.event_shape
        self._batch_shape = initial_distribution.batch_shape

    @property
    def arg_constraints(self):
        return {}

    def transform_initial_sample(self, z0):
        z = z0
        default_constant_block_size = (z.shape[-1] // 2) + (z.shape[-1] % 2)

        layers: list[ConditionalRealNVPLayer] = self.conditional_flow.layers

        for layer in layers:
            z = layer.transform(z, self.conditioning)
            # now swap the two blocks
            constant_block_size = layer.constant_block_size or default_constant_block_size  # NB constant_block_size should never be 0 anyway
            z1, z2 = z[..., :constant_block_size], z[..., constant_block_size:]
            z = torch.cat([z2, z1], dim=-1)
        # self.conditional_flow.DEBUG.z_norm_sampled.append(z.norm().detach().cpu().item())
        return z

    @torch.no_grad
    def sample(self, sample_shape=torch.Size([])):
        z0 = self.initial_distribution.sample(sample_shape)
        return self.transform_initial_sample(z0)

    def rsample(self, sample_shape=torch.Size([])):
        z0 = self.initial_distribution.rsample(sample_shape)
        return self.transform_initial_sample(z0)

    def log_prob(self, z):
        return_value = 0
        default_constant_block_size = (z.shape[-1] // 2) + (z.shape[-1] % 2)
        _debug_index = len(self.conditional_flow.layers) - 1
        _debug_info = {}

        layers: list[ConditionalRealNVPLayer] = self.conditional_flow.layers

        for layer in reversed(layers):

            _debug_info[f'debug/z_norm/layer_{_debug_index}_in'] = z.norm().item()
            _debug_info[f"debug/z_max/layer_{_debug_index}_in"] = z.max().item()
            _debug_info[f"debug/z_min/layer_{_debug_index}_in"] = z.min().item()
            _debug_index -= 1

            # undo swapping
            constant_block_size = layer.constant_block_size or default_constant_block_size
            z1, z2 = z[..., -constant_block_size:], z[..., :-constant_block_size]
            z = torch.cat([z1, z2], dim=-1)

            inverse_layer = layer.inverse
            z, delta_return_value = inverse_layer(z, self.conditioning)
            return_value += delta_return_value
            # return_value += inverse_layer.log_det_jacobian(z, self.conditioning)
            # z = inverse_layer.transform(z, self.conditioning)
        return_value += self.initial_distribution.log_prob(z)
        if wandb.run is not None:
            wandb.log(_debug_info, step=wandb.run.step, commit=False)
        return return_value

    def inverse_flow(self, z):
        default_constant_block_size = (z.shape[-1] // 2) + (z.shape[-1] % 2)

        layers: list[ConditionalRealNVPLayer] = self.conditional_flow.layers

        for layer in reversed(layers):
            # undo swapping
            constant_block_size = layer.constant_block_size or default_constant_block_size
            z1, z2 = z[..., -constant_block_size:], z[..., :-constant_block_size]
            z = torch.cat([z1, z2], dim=-1)

            inverse_layer = layer.inverse
            z = inverse_layer.transform(z, self.conditioning)
        return z
