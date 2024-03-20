# Code reference:
# This script is based on invertible-resnet by jhjacobsen
# Source: https://github.com/jhjacobsen/invertible-resnet.git

"""
Soft Spectral Normalization (use for fc-layers)

Adpated from:
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch


class SpectralNorm(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.

    def __init__(self, net, coeff, name="weight", heads=1):
        self.net = net
        self.coeff = coeff
        self.name = name
        self.heads = heads

    def compute_weight(self, module):
        if self.net.endswith("resgcn"):
            weight = getattr(module.lin, self.name + "_orig")
        elif self.net.endswith("resgat"):
            weight = getattr(module.lin_src, self.name + "_orig")
        lip_log = getattr(module, self.name + "_lip")  # for logging

        if self.heads == 1:
            lip_con_approx = torch.norm(weight)
        else:
            lip_con_approx = torch.norm(weight.flatten().view(self.heads, -1),dim=-1).mean()

        # soft normalization: only when lip_const larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), lip_con_approx / self.coeff)
        weight = weight / factor

        # for logging
        lip_log.copy_(lip_con_approx.detach())

        return weight

    def __call__(self, module, inputs):
        if self.net.endswith("resgcn"):
            setattr(
                module.lin,
                self.name,
                self.compute_weight(module),
            )
        elif self.net.endswith("resgat"):
            setattr(
                module.lin_src,
                self.name,
                self.compute_weight(module),
            )

    @staticmethod
    def apply(module, net, coeff, name, heads):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNorm(net, coeff, name, heads)
        if net.endswith("resgcn"):
            weight = module.lin._parameters[name]
            delattr(module.lin, fn.name)

            module.lin.register_parameter(fn.name + "_orig", weight)
            # We still need to assign weight back as fn.name because all sorts of
            # things may assume that it exists, e.g., when initializing weights.
            # However, we can't directly assign as it could be an nn.Parameter and
            # gets added as a parameter. Instead, we register weight.data as a plain
            # attribute.
            setattr(module.lin, fn.name, weight.data)
        elif net.endswith("resgat"):
            weight = module.lin_src._parameters[name]
            delattr(module.lin_src, fn.name)

            module.lin_src.register_parameter(fn.name + "_orig", weight)
            # We still need to assign weight back as fn.name because all sorts of
            # things may assume that it exists, e.g., when initializing weights.
            # However, we can't directly assign as it could be an nn.Parameter and
            # gets added as a parameter. Instead, we register weight.data as a plain
            # attribute.
            setattr(module.lin_src, fn.name, weight.data)

        module.register_buffer(fn.name + "_lip", torch.ones(1).to(weight.device))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        fn = self.fn
        version = local_metadata.get("spectral_norm", {}).get(
            fn.name + ".version", None
        )
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + "_orig"]
                weight = state_dict.pop(prefix + fn.name)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + "_u"]
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[prefix + fn.name + "_v"] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if "spectral_norm" not in local_metadata:
            local_metadata["spectral_norm"] = {}
        key = self.fn.name + ".version"
        if key in local_metadata["spectral_norm"]:
            raise RuntimeError(
                "Unexpected key in metadata['spectral_norm']: {}".format(key)
            )
        local_metadata["spectral_norm"][key] = self.fn._version


def spectral_norm_gnn(module, net, coeff, name="weight", heads=1):
    SpectralNorm.apply(module, net, coeff, name, heads)
    return module


def remove_spectral_norm(module, name="weight"):
    r"""Removes the spectral normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))
