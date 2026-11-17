# Code adapted from the official NECTR implementation https://github.com/arghyasinha/nectr
from __future__ import annotations
from abc import abstractmethod
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Denoiser
from .unet import UNet


class NECTR_Base(Denoiser):
    r"""
    Base class for the NECTR denoisers.

    Subclasses define the feature map :meth:`pre_activation` and the weight network :meth:`N_theta`.

    :param str nlm_type: filter applied by :meth:`forward`, one of ``"non_symm"``, ``"symm_cached"`` or
        ``"symm_uncached"``.
    :param torch.device, str device: device to run the model on.
    """

    NLM_TYPES = ("non_symm", "symm_cached", "symm_uncached")

    def __init__(self, nlm_type: str, device: torch.device | str = None, blind: bool = False, 
                 reference: torch.Tensor | None = None, window_rad: int = 7, in_channel: int = 3):
        super().__init__()

        self.sigma = 1
        self.blind = blind
        self.hold_cache = {}
        if nlm_type.lower() not in self.NLM_TYPES:
            warnings.warn(
                f"Unknown nlm_type {nlm_type!r}, expected one of {self.NLM_TYPES}. "
                f"Falling back to nlm_type='non_symm'."
            )
            nlm_type = "non_symm"
        self.reference = reference

        self.window_rad = window_rad
        self.in_channel = in_channel
        self.nlm_type = nlm_type.lower()
        self.device = device

    @staticmethod
    def _check_output_activation(output_activation, supported):
        r"""
        Return ``output_activation`` if supported, otherwise warn and fall back to ``"control_sigmoid"``.

        :param str output_activation: requested activation.
        :param tuple supported: supported activations.
        :return: (str) the activation to use.
        """
        if output_activation not in supported:
            warnings.warn(
                f"Unknown output_activation {output_activation!r}, expected one of {supported}. "
                f"Falling back to output_activation='control_sigmoid'."
            )
            return "control_sigmoid"
        return output_activation

    def _get_reference(self, x, reference=None):
        r"""
        Return the guide image: ``reference`` if given, else ``self.reference``, else ``x``.

        :param torch.Tensor x: image being filtered.
        :param torch.Tensor reference: explicit guide image, or ``None``.
        :return: (:class:`torch.Tensor`) the guide image.
        """
        if reference is not None:
            return reference
        if self.reference is not None:
            return self.reference
        return x

    def _append_sigma(self, x, sigma=None):
        r"""
        Append a constant noise level map to ``x`` as an extra channel (non-blind models only).

        :param torch.Tensor x: image of shape ``(B, C, H, W)``.
        :param float, list, torch.Tensor sigma: noise level, scalar or one value per image. Required when
            ``blind=False``.
        :return: (:class:`torch.Tensor`) ``x`` if blind, else a tensor of shape ``(B, C + 1, H, W)``.
        """
        if self.blind:
            return x
        if sigma is None:
            raise ValueError("sigma is None, but blind is False")
        sigma = self._handle_sigma(
            sigma, batch_size=x.shape[0], ndim=x.ndim, device=x.device, dtype=x.dtype
        )  # (B, 1, 1, 1)
        sigma_map = sigma.expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat((x, sigma_map), dim=1)

    @abstractmethod
    def pre_activation(self, x, sigma=None):
        r"""Compute the features of the guide image used by :meth:`N_theta`."""
        raise NotImplementedError

    @abstractmethod
    def N_theta(self, x, x_s):
        r"""Compute the non-negative weights between the features ``x`` and their shifted copy ``x_s``."""
        raise NotImplementedError

    def controlled_sigmoid(self, x, a, b):
        r"""Learned positive activation :math:`\mathrm{softplus}(a) / (1 + \exp(-\mathrm{sigmoid}(b)\,x))`."""
        return F.softplus(a) / (1 + torch.exp(-F.sigmoid(b) * x))

    def _shift(self, t, dx, dy, H, W):
        r"""Crop the window-padded tensor ``t`` at offset ``(dx, dy)``."""
        r = self.window_rad
        return t[:, :, r + dx : r + dx + H, r + dy : r + dy + W]

    def nlm(
        self, x, reference=None, sigma=None, store_cache=False, accum=False
    ) -> torch.Tensor:
        r"""
        Apply the normalized non-local filter :math:`\hat{x} = D^{-1} W x`, with :math:`D = \mathrm{diag}(W\mathbf{1})`.

        Only half of the search window is evaluated by :meth:`N_theta`. The weights for the opposite offsets are
        recovered by shifting, so :math:`W` is symmetric. If ``self.hold_cache`` is not empty, the cached weights are
        used instead of evaluating :meth:`N_theta`.

        :param torch.Tensor x: image to filter, of shape ``(B, C, H, W)``.
        :param torch.Tensor reference: guide image used to compute the weights. If ``None``, see
            :meth:`_get_reference`.
        :param float, torch.Tensor sigma: noise level. Required when ``blind=False``.
        :param bool store_cache: if ``True``, store the computed weights in ``self.hold_cache``.
        :param bool accum: if ``True``, also return the normalization constants :math:`W\mathbf{1}`.
        :return: ``x_hat``, or ``(x_hat, Z)`` if ``accum=True``, both of shape ``(B, C, H, W)``.
        """
        r = self.window_rad
        H, W = x.shape[2], x.shape[3]
        pad = (r, r, r, r)
        U = torch.zeros_like(x)
        Z = torch.zeros_like(x)

        if not self.hold_cache:
            reference = self.pre_activation(
                self._get_reference(x, reference), sigma=sigma
            )
            padded_reference = F.pad(reference, pad, mode="circular")
        padded_img = F.pad(x, pad, mode="circular")
        box = F.pad(torch.ones_like(x), pad, mode="constant", value=0)

        cache = {}
        # half window: dy > 0 for all dx, plus dy = 0 for dx >= 0
        offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(1, r + 1)]
        offsets += [(dx, 0) for dx in range(0, r + 1)]
        for dx, dy in offsets:
            if not self.hold_cache:
                reference_shifted = self._shift(padded_reference, dx, dy, H, W)
                weight = self.N_theta(reference, reference_shifted)
                weight = weight * self._shift(box, dx, dy, H, W)
                if store_cache:
                    cache[(dx, dy)] = weight
            else:
                weight = self.hold_cache[(dx, dy)]

            U = U + weight * self._shift(padded_img, dx, dy, H, W)
            Z = Z + weight

            if dx == 0 and dy == 0:
                continue

            # weight for the opposite offset (-dx, -dy)
            weight = self._shift(F.pad(weight, pad, mode="circular"), -dx, -dy, H, W)
            weight = weight * self._shift(box, -dx, -dy, H, W)
            U = U + weight * self._shift(padded_img, -dx, -dy, H, W)
            Z = Z + weight

        if store_cache:
            self.hold_cache = cache
        U = U / Z
        return (U, Z) if accum else U

    def forward_symm(self, x, reference=None, sigma=None, accum=False):
        r"""
        Apply a symmetric, row-stochastic version of the non-local filter.

        With :math:`S = D^{-1/2} W D^{-1/2}` and :math:`\alpha = \max_i (S\mathbf{1})_i`, this returns

        .. math::

            \hat{x} = \frac{1}{\alpha} S x + \mathrm{diag}\Big(\mathbf{1} - \frac{1}{\alpha} S\mathbf{1}\Big) x.

        The weights are computed once and cached for the three filtering steps.

        :param torch.Tensor x: image to filter, of shape ``(B, C, H, W)``.
        :param torch.Tensor reference: guide image used to compute the weights. If ``None``, see
            :meth:`_get_reference`.
        :param float, torch.Tensor sigma: noise level. Required when ``blind=False``.
        :param bool accum: if ``True``, also return :math:`D`.
        :return: ``x_hat``, or ``(x_hat, D)`` if ``accum=True``, both of shape ``(B, C, H, W)``.
        """
        # fix the guide image now, so the weights are not computed from the all-ones image below
        reference = self._get_reference(x, reference)
        self.hold_cache = {}
        try:
            U, D = self._symm(x, reference, sigma, store_cache=True)
        finally:
            del self.hold_cache
            self.hold_cache = {}
        return (U, D) if accum else U

    def forward_symm_naive(self, x, reference=None, sigma=None, accum=False):
        r"""
        Same as :meth:`forward_symm`, but without the weight cache (the weight network is evaluated three times).

        :param torch.Tensor x: image to filter, of shape ``(B, C, H, W)``.
        :param torch.Tensor reference: guide image used to compute the weights. If ``None``, see
            :meth:`_get_reference`.
        :param float, torch.Tensor sigma: noise level. Required when ``blind=False``.
        :param bool accum: if ``True``, also return :math:`D`.
        :return: ``x_hat``, or ``(x_hat, D)`` if ``accum=True``, both of shape ``(B, C, H, W)``.
        """
        reference = self._get_reference(x, reference)
        U, D = self._symm(x, reference, sigma, store_cache=False)
        return (U, D) if accum else U

    def _symm(self, x, reference, sigma, store_cache):
        r"""Shared computation of :meth:`forward_symm` and :meth:`forward_symm_naive`."""
        one = torch.ones_like(x)
        _, D = self.nlm(
            one, reference, sigma=sigma, store_cache=store_cache, accum=True
        )
        sqrt_D = torch.sqrt(D)

        # S 1 and its supremum alpha
        one_hat = sqrt_D * self.nlm(1.0 / sqrt_D, reference, sigma=sigma)
        alpha = torch.amax(one_hat, dim=[2, 3], keepdim=True)

        x1 = sqrt_D * self.nlm(x / sqrt_D, reference, sigma=sigma) / alpha
        x_tilde = (one - one_hat / alpha) * x

        return x1 + x_tilde, D

    def forward(self, x, sigma=None, *args, reference=None, **kwargs):
        r"""
        Apply the NECTR denoiser.

        ``nlm_type="non_symm"`` applies :meth:`nlm`, ``"symm_cached"`` applies :meth:`forward_symm` and
        ``"symm_uncached"`` applies :meth:`forward_symm_naive`.

        :param torch.Tensor x: noisy image of shape ``(B, C, H, W)``.
        :param float, torch.Tensor sigma: noise level. Required when ``blind=False``.
        :param torch.Tensor reference: guide image used to compute the weights. If ``None``, ``self.reference`` or
            ``x`` is used.
        :return: (:class:`torch.Tensor`) denoised image of shape ``(B, C, H, W)``.
        """
        if self.nlm_type == "symm_cached":
            return self.forward_symm(x, reference, sigma)
        elif self.nlm_type == "symm_uncached":
            return self.forward_symm_naive(x, reference, sigma)
        else:
            return self.nlm(x, reference, sigma)



class NECTR(NECTR_Base):
    r"""
    NECTR non-local convolutional kernel denoiser.

    Nonexpansive learned non-local means denoiser introduced by :footcite:t:`sinha2026trainable`, adapted from the
    `official implementation <https://github.com/arghyasinha/nectr>`_ (Model 1, lightweight CNN). A CNN predicts
    non-negative pairwise weights over a search window, and the output is the weighted average of the input pixels.

    :param torch.device, str device: device to run the model on.
    :param str model_type: ``"concat"`` concatenates the features of the pixel pair before :math:`N_\theta`.
        ``"rbf"`` only builds the feature extractor.
    :param int residual_depth: number of residual blocks in the feature extractor.
    :param int proj_depth: number of pointwise projection layers in :math:`N_\theta`, each halving the channels.
    :param int latent_dim: number of channels of the latent features.
    :param int in_channel: number of channels of the input image.
    :param int patch_rad: radius of the first convolution kernel.
    :param int window_rad: radius of the non-local search window.
    :param str output_activation: positive activation of :math:`N_\theta`, one of ``"control_sigmoid"``,
        ``"exp"`` or ``"sig"``.
    :param bool blind: if ``False``, the noise level is given to the network and ``sigma`` is required.
    :param torch.Tensor reference: default guide image used to compute the weights. If ``None``, the input is used.
    :param str nlm_type: filter applied by :meth:`forward`, one of ``"non_symm"``, ``"symm_cached"`` or
        ``"symm_uncached"``.

    |sep|

    :Examples:

        >>> import deepinv as dinv
        >>> import torch
        >>> model = dinv.models.NECTR(in_channel=3, latent_dim=32, window_rad=2)
        >>> y = torch.rand(1, 3, 16, 16)
        >>> with torch.no_grad():
        ...     x_hat = model(y, sigma=0.1)
        >>> x_hat.shape
        torch.Size([1, 3, 16, 16])

    """

    def __init__(
        self,
        device: torch.device | str = None,
        model_type: str = "concat",
        residual_depth: int = 3,
        proj_depth: int = 3,
        latent_dim: int = 128,
        in_channel: int = 3,
        patch_rad: int = 3,
        window_rad: int = 7,
        output_activation: str = "control_sigmoid",
        blind: bool = False,
        reference: torch.Tensor | None = None,
        nlm_type: str = "non_symm",
    ):
        super(NECTR, self).__init__(nlm_type, device, blind, reference, window_rad, in_channel)
        self.model_type = model_type

        self.residual_depth = residual_depth
        self.proj_depth = proj_depth
        self.latent_dim = latent_dim
        self.patch_size = 2 * patch_rad + 1
        self.output_activation = self._check_output_activation(
            output_activation, ("control_sigmoid", "exp", "sig")
        )

        in_dim = self.in_channel if blind else self.in_channel + 1

        self.in_layer = nn.Conv2d(
            in_dim,
            latent_dim,
            kernel_size=self.patch_size,
            padding=self.patch_size // 2,
        )

        self.in_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(),
                )
                for _ in range(self.residual_depth - 1)
            ]
        )
        self.in_proj.append(nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1))

        self.in_residual = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim * 4, kernel_size=1, padding=0),
                    nn.LeakyReLU(),
                    nn.Conv2d(latent_dim * 4, latent_dim, kernel_size=1, padding=0),
                )
                for _ in range(self.residual_depth)
            ]
        )

        self.in_norm = nn.ModuleList(
            [nn.LayerNorm(self.latent_dim) for _ in range(self.residual_depth)]
        )

        if self.model_type == "rbf":
            return

        self.proj = nn.ModuleList([])
        in_dim = 2 * latent_dim if self.model_type == "concat" else latent_dim
        for i in range(proj_depth):
            self.proj.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_dim // (2**i), in_dim // (2 ** (i + 1)), kernel_size=1
                        ),
                        nn.LayerNorm(in_dim // (2 ** (i + 1))),
                        nn.LeakyReLU(),
                    ]
                )
            )

        in_dim = in_dim // (2**proj_depth)
        out_dim = (
            in_channel * 3
            if self.output_activation == "control_sigmoid"
            else in_channel
        )

        self.out = nn.Conv2d(in_dim, out_dim, bias=False, kernel_size=1)

    def pre_activation(self, x, sigma=None):
        r"""
        Compute the latent features of the guide image.

        :param torch.Tensor x: guide image of shape ``(B, C, H, W)``.
        :param float, torch.Tensor sigma: noise level. Required when ``blind=False``.
        :return: (:class:`torch.Tensor`) features of shape ``(B, latent_dim, H, W)``.
        """
        x = self._append_sigma(x, sigma)
        x = self.in_layer(x)
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return x

    def N_theta(self, x, x_s):
        r"""
        Compute the weights between the features ``x`` and their shifted copy ``x_s``.

        :param torch.Tensor x: features (output of :meth:`pre_activation`).
        :param torch.Tensor x_s: the same features shifted by one offset of the search window.
        :return: (:class:`torch.Tensor`) non-negative weights of shape ``(B, in_channel, H, W)``.
        """
        if self.model_type == "concat":
            x = torch.cat((x, x_s), dim=1)

        for conv, norm, act in self.proj:
            x = conv(x)
            x = norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = act(x)

        x = self.out(x)

        if self.output_activation == "exp":
            x = torch.exp(x / self.sigma**2)
        elif self.output_activation == "control_sigmoid":
            k, a, b = torch.chunk(x, dim=1, chunks=3)
            x = self.controlled_sigmoid(k, a, b) + 1e-8
        elif self.output_activation == "sig":
            x = F.sigmoid(-x / self.sigma**2)

        return x


class NECTRUNet(NECTR_Base):
    r"""
    NECTR non-local kernel denoiser with a U-Net weight network.

    Variant of :class:`deepinv.models.NECTR` (:footcite:t:`sinha2026trainable`) where :math:`N_\theta` is a
    :class:`deepinv.models.UNet` applied to the pair of guide images.

    :param torch.device, str device: device to run the model on.
    :param str model_type: only ``"concat"`` is supported.
    :param int in_channel: number of channels of the input image.
    :param int scales: number of scales of the U-Net. The image size must be divisible by ``2 ** (scales - 1)``.
    :param int window_rad: radius of the non-local search window.
    :param str output_activation: only ``"control_sigmoid"`` is supported.
    :param bool blind: if ``False``, the noise level is given to the network and ``sigma`` is required.
    :param torch.Tensor reference: default guide image used to compute the weights. If ``None``, the input is used.
    :param str nlm_type: filter applied by :meth:`forward`, one of ``"non_symm"``, ``"symm_cached"`` or
        ``"symm_uncached"``.

    |sep|

    :Examples:

        >>> import deepinv as dinv
        >>> import torch
        >>> model = dinv.models.NECTRUNet(in_channel=3, scales=2, window_rad=1)
        >>> y = torch.rand(1, 3, 16, 16)
        >>> with torch.no_grad():
        ...     x_hat = model(y, sigma=0.1)
        >>> x_hat.shape
        torch.Size([1, 3, 16, 16])

    """

    def __init__(
        self,
        device: torch.device | str = None,
        in_channel: int = 3,
        scales: int = 4,
        window_rad: int = 7,
        output_activation: str = "control_sigmoid",
        blind: bool = False,
        reference: torch.Tensor | None = None,
        nlm_type: str = "non_symm",
    ):
        super(NECTRUNet, self).__init__(nlm_type, device, blind)

        self.scales = scales
        self.output_activation = self._check_output_activation(
            output_activation, ("control_sigmoid",)
        )

        in_dim = self.in_channel if blind else self.in_channel + 1
        out_dim = self.in_channel * 3

        self.unet = UNet(
            in_channels=2 * in_dim,
            out_channels=self.in_channel,
            scales=self.scales,
            residual=False,
            cat=True,
            batch_norm=False,
        ).to(device)

        self.layer_norm = nn.LayerNorm(self.in_channel)
        self.out = nn.Conv2d(self.in_channel, out_dim, bias=False, kernel_size=1)

    def pre_activation(self, x, sigma=None):
        r"""
        Append the noise level map to the guide image (non-blind only).

        :param torch.Tensor x: guide image of shape ``(B, C, H, W)``.
        :param float, torch.Tensor sigma: noise level. Required when ``blind=False``.
        :return: (:class:`torch.Tensor`) tensor of shape ``(B, C, H, W)`` if blind, else ``(B, C + 1, H, W)``.
        """
        return self._append_sigma(x, sigma)

    def N_theta(self, x, x_s):
        r"""
        Compute the weights between the guide image ``x`` and its shifted copy ``x_s``.

        :param torch.Tensor x: output of :meth:`pre_activation`.
        :param torch.Tensor x_s: the same tensor shifted by one offset of the search window.
        :return: (:class:`torch.Tensor`) non-negative weights of shape ``(B, in_channel, H, W)``.
        """
        x = torch.cat((x, x_s), dim=1)
        x = self.unet(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.out(x)
        k, a, b = torch.chunk(x, dim=1, chunks=3)
        return self.controlled_sigmoid(k, a, b) + 1e-8
