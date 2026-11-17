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
    NLM_TYPES = ("non_symm", "symm_cached", "symm_uncached")
    def __init__(self, nlm_type, device):
        super().__init__()

        self.hold_cache = {}
        if nlm_type.lower() not in self.NLM_TYPES:
            warnings.warn(
                f"Unknown nlm_type {nlm_type!r}, expected one of {self.NLM_TYPES}. "
                f"Falling back to nlm_type='non_symm'."
            )
            nlm_type = "non_symm"
        self.nlm_type = nlm_type.lower()
        self.device = device

    @staticmethod
    def _check_output_activation(output_activation, supported):
        r"""
        Return ``output_activation`` if it is supported, otherwise warn and fall back to ``"control_sigmoid"``.

        :param str output_activation: requested activation.
        :param tuple supported: activations supported by the model.
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
        Choose the guide image used to compute the weights.

        The explicit ``reference`` argument has priority, then the reference stored in ``self.reference``, then ``x``.

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

        The noise level is converted with :meth:`deepinv.models.Denoiser._handle_sigma`.

        :param torch.Tensor x: image of shape ``(B, C, H, W)``.
        :param float, int, list, torch.Tensor sigma: noise level. A scalar (used for the whole batch), or one value
            per image given as a list or a tensor of shape ``(B,)``, ``(B, 1)`` or ``(B, 1, 1, 1)``. Required when
            ``blind=False``, ignored otherwise.
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
        raise NotImplementedError

    @abstractmethod
    def N_theta(self, x, x_s):
        raise NotImplementedError

    def controlled_sigmoid(self, x, a, b):
        # a(x) = relu/sq(alpha) / ( 1 + exp ( - sigmoid (beta) x)
        return F.softplus(a)/(1+torch.exp(-F.sigmoid(b)*x))

    def nlm(
            self, x, reference=None, sigma=None, store_cache=False, accum=False
        ) -> torch.Tensor:
        r"""
        Apply the normalized non-local filter :math:`\hat{x} = D^{-1} W x`.

        The weights :math:`W` are computed from ``reference`` (or from ``x`` if no reference is given) and applied
        to ``x``. Only half of the search window is evaluated by :meth:`N_theta`. The weights for the opposite offsets
        are recovered by shifting, so :math:`W` is symmetric.

        :param torch.Tensor x: image to filter, of shape ``(B, C, H, W)``.
        :param torch.Tensor reference: guide image used to compute the weights, of shape ``(B, C, H, W)``. If
            ``None``, ``x`` is used.
        :param float, torch.Tensor sig: noise level, passed to :meth:`pre_activation`. Required when ``blind=False``.
        :param dict cache: weights returned by an earlier call with ``store_cache=True``, keyed by the offset
            ``(dx, dy)``. If given, the weight network is not evaluated again. Use this to apply the same filter to
            several images.
        :param bool store_cache: if ``True``, also return the computed weights so that they can be passed back as
            ``cache``.
        :return: ``(x_hat, Z)``, or ``(x_hat, Z, cache)`` if ``store_cache=True``. ``x_hat`` is the filtered image,
            and ``Z`` holds the per-pixel normalization constants :math:`D = \mathrm{diag}(W\mathbf{1})`, both of
            shape ``(B, C, H, W)``.
        """
        U = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        Z = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        # guide image: explicit argument > stored self.reference > x
        if not self.hold_cache:
            reference = self.pre_activation(self._get_reference(x, reference), sigma=sigma)

        padded_img = F.pad(x, (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='circular')
        if not self.hold_cache:
            padded_reference = F.pad(reference, (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='circular')
        box = F.pad(torch.ones_like(x, device=x.device, dtype=x.dtype), (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='constant', value=0)

        if store_cache: cache = {}
        count_real = 0
        count_total = 0
        for dx in range(-self.window_rad, self.window_rad + 1):
            for dy in range(1, self.window_rad + 1):

                if not self.hold_cache:
                    reference_shifted = padded_reference[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                    weight = self.N_theta(reference, reference_shifted)
                    comp_box = box[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                    weight = weight * comp_box
                    if store_cache:
                        key = (dx, dy)
                        cache[key] = weight
                else:
                    key = (dx, dy)
                    weight = self.hold_cache[key]

                v = padded_img[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                U = U + weight * v
                Z = Z + weight
                
                dx_inv = -dx
                dy_inv = -dy
                weight_padded = F.pad(weight, (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='circular')
                weight = weight_padded[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]
                comp_box = box[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]

                weight = weight * comp_box
                if torch.isnan(weight).any():
                    print(f"NaN detected in weight for dx {dx} and dy {dy}")
                v = padded_img[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]
                U = U + weight * v
                Z = Z + weight

        for dx in range(0, self.window_rad + 1):
            for dy in [0]:

                if not self.hold_cache:
                    reference_shifted = padded_reference[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                    weight = self.N_theta(reference, reference_shifted)
                    comp_box = box[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                    weight = weight * comp_box
                    if store_cache:
                        key = (dx, dy)
                        cache[key] = weight
                else:
                    key = (dx, dy)
                    weight = self.hold_cache[key]

                v = padded_img[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                U = U + weight * v
                Z = Z + weight
                

                if dx == 0 and dy ==0:
                    continue

                dx_inv = -dx
                dy_inv = -dy
                
                weight_padded = F.pad(weight, (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='circular')
                weight = weight_padded[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]
                comp_box = box[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]

                weight = weight * comp_box
                
                v = padded_img[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]
                U = U + weight * v
                Z = Z + weight
        if store_cache: self.hold_cache = cache
        U = U / Z
        return (U, Z) if accum else U


    def forward_symm(self, x, reference, sigma=None, accum=False):
        r"""
        Apply a symmetric, row-stochastic version of the non-local filter.

        With :math:`D = \mathrm{diag}(W\mathbf{1})`, let :math:`S = D^{-1/2} W D^{-1/2}` (symmetric) and
        :math:`\alpha = \max_i (S\mathbf{1})_i`. This method returns

        .. math::

            \hat{x} = \frac{1}{\alpha} S x + \mathrm{diag}\Big(\mathbf{1} - \frac{1}{\alpha} S\mathbf{1}\Big) x.

        The resulting operator is symmetric, has non-negative entries and has rows that sum to one. This is useful
        for plug-and-play algorithms that need a symmetric (gradient-type) denoiser.

        The weight network is evaluated only once. Its outputs are cached and reused for the three filtering steps.

        :param torch.Tensor x: image to filter, of shape ``(B, C, H, W)``.
        :param torch.Tensor reference: guide image used to compute the weights (typically the noisy observation),
            of shape ``(B, C, H, W)``.
        :param float, torch.Tensor sig: noise level, passed to :meth:`pre_activation`. Required when ``blind=False``.
        :return: ``(x_hat, D)``: the filtered image and the per-pixel normalization constants
            :math:`W\mathbf{1}`, both of shape ``(B, C, H, W)``.
        """
        # fix the guide image now (argument > self.reference > x), so the weights are not computed from the
        # all-ones image below
        reference = self._get_reference(x, reference)
        self.hold_cache = {}
        # Find D
        one = torch.ones_like(x, device=x.device, dtype=x.dtype)
        _, D = self.nlm(one, reference, sigma=sigma, store_cache=True, accum=True)

        #Find one_hat
        one_hat = 1.0/torch.sqrt(D)
        one_hat = self.nlm(one_hat, reference, sigma=sigma)
        one_hat = torch.sqrt(D) * one_hat

        #find one_hat supremum
        one_hat_sup = torch.amax(one_hat, dim=[2,3], keepdim=True)

        normalized_one_hat = one_hat / one_hat_sup

        #find second component
        x_tilde = (one - normalized_one_hat)*x

        #find first component
        x1 = (1/torch.sqrt(D))*x
        x1 = self.nlm(x1, reference, sigma=sigma)
        x1 = x1 * torch.sqrt(D)
        x1 = x1/one_hat_sup

        U = x1 + x_tilde
        del self.hold_cache
        self.hold_cache = {}
        return (U, D) if accum else U

    def forward_symm_nieve(self, x, reference, sigma=None, accum=False):
        r"""
        Reference implementation of :meth:`forward_symm` that does not use the weight cache.

        It computes the same output, but evaluates the weight network three times. This is slower and uses more
        memory. It is mainly useful for testing.

        :param torch.Tensor x: image to filter, of shape ``(B, C, H, W)``.
        :param torch.Tensor reference: guide image used to compute the weights, of shape ``(B, C, H, W)``.
        :param float, torch.Tensor sig: noise level, passed to :meth:`pre_activation`. Required when ``blind=False``.
        :return: ``(x_hat, D)``: the filtered image and the per-pixel normalization constants, both of shape
            ``(B, C, H, W)``.
        """
        # fix the guide image now (argument > self.reference > x), so the weights are not computed from the
        # intermediate images below
        reference = self._get_reference(x, reference)
        # Find D
        one = torch.ones_like(x, device=x.device, dtype=x.dtype)
        _, D = self.nlm(one, reference, sigma=sigma, accum=True)

        #Find one_hat
        one_hat = 1.0/torch.sqrt(D)
        one_hat = self.nlm(one_hat, reference, sigma=sigma)
        one_hat = torch.sqrt(D) * one_hat

        #find one_hat supremum
        one_hat_sup = torch.amax(one_hat, dim=[2,3], keepdim=True)

        normalized_one_hat = one_hat / one_hat_sup

        #find second component
        x_tilde = (one - normalized_one_hat)*x

        #find first component
        x1 = (1/torch.sqrt(D))*x
        x1 = self.nlm(x1, reference, sigma=sigma)
        x1 = x1 * torch.sqrt(D)
        x1 = x1/one_hat_sup

        U = x1 + x_tilde
        return (U, D) if accum else U

    def forward(self, x, sigma=None, *args, reference=None, **kwargs):
        r"""
        Apply the NECTR denoiser.

        The filter applied depends on ``nlm_type``: ``"non_symm"`` applies :meth:`nlm`, ``"symm_cached"`` applies
        :meth:`forward_symm` and ``"symm_uncached"`` applies :meth:`forward_symm_nieve`.

        :param torch.Tensor x: noisy image of shape ``(B, C, H, W)``.
        :param float, torch.Tensor sigma: noise level. Required when ``blind=False``.
        :param args: extra positional arguments, ignored. They are accepted so that the denoiser can be called by
            deepinv algorithms that pass additional arguments.
        :param torch.Tensor reference: guide image used to compute the weights, of shape ``(B, C, H, W)``. If
            ``None``, ``x`` is used. Keyword-only.
        :param kwargs: extra keyword arguments, ignored.
        :return: (:class:`torch.Tensor`) denoised image of shape ``(B, C, H, W)``.
        """
        if self.nlm_type == "symm_cached":
            return self.forward_symm(x, reference, sigma)
        elif self.nlm_type == "symm_uncached":
            return self.forward_symm_nieve(x, reference, sigma)
        else:
            return self.nlm(x, reference, sigma)


class NECTR(NECTR_Base):
    r"""
    NECTR non-local convolutional kernel denoiser.

    The NECTR (NonExpansive denoising for ConTractive Reconstruction) denoiser was introduced by
    :footcite:t:`sinha2026trainable`. It is globally nonexpansive by design, so it can be used inside plug-and-play
    algorithms to obtain a contractive reconstruction operator. This implementation is adapted from the
    `official NECTR repository <https://github.com/arghyasinha/nectr>`_ (Model 1, lightweight CNN).

    This denoiser is a learned non-local means filter. It does not predict the clean image directly. It predicts
    non-negative pairwise affinities between pixels and returns a weighted average of the input pixels.
    Only half of the search window is evaluated by the network. The weight computed for the offset :math:`d` is reused
    for the offset :math:`-d`.

    :param torch.device, str device: device to run the model on.
    :param str model_type: how the features of the pixel pair are merged before going into :math:`N_\theta`.
        ``"concat"`` concatenates the two feature maps along the channel dimension. ``"rbf"`` only builds the feature
        extractor and skips the projection head.
    :param int residual_depth: number of residual blocks in the feature extractor :math:`\phi`.
    :param int proj_depth: number of pointwise projection layers in :math:`N_\theta`. Each layer halves the number of
        channels, so ``2 * latent_dim`` (for ``"concat"``) must be divisible by ``2 ** proj_depth``.
    :param int latent_dim: number of channels in the latent feature embedding.
    :param int in_channel: number of channels of the input image.
    :param int patch_rad: radius of the patch used by the first convolution of :math:`\phi`. The kernel size is
        ``2 * patch_rad + 1``.
    :param int window_rad: radius of the non-local search window. The window has
        ``(2 * window_rad + 1) ** 2`` pixels.
    :param str output_activation: positive activation applied to the output of :math:`N_\theta`. ``"control_sigmoid"``
        uses a learned sigmoid :math:`\mathrm{softplus}(a) / (1 + \exp(-\mathrm{sigmoid}(b)\,k))`, ``"exp"`` uses an
        exponential, and ``"sig"`` uses a sigmoid. Any other value raises a warning and falls back to
        ``"control_sigmoid"``.
    :param bool blind: if ``True``, the network does not get the noise level as input. If ``False``, ``sig`` must be
        given to :meth:`forward`.

    |sep|

    :Examples:

        >>> import torch
        >>> from deepinv.models.NECTR import NECTR_denoiser
        >>> model = NECTR_denoiser(in_channel=3, latent_dim=32, window_rad=2)
        >>> y = torch.rand(1, 3, 16, 16)
        >>> with torch.no_grad():
        ...     x_hat, Z = model(y, sig=0.1)
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
            reference: torch.Tensor = None,
            nlm_type: str = "non_symm"
        ):
        super(NECTR, self).__init__(nlm_type, device)
        self.model_type = model_type

        self.residual_depth = residual_depth 
        self.proj_depth = proj_depth 
        self.latent_dim = latent_dim 
        self.sigma = 1
        self.blind = blind
        self.patch_size = 2*patch_rad + 1
        self.output_activation = self._check_output_activation(
            output_activation, ("control_sigmoid", "exp", "sig")
        )
        self.reference = reference

        self.window_rad = window_rad
        self.in_channel = in_channel

        if blind:
            in_dim=self.in_channel
        else:
            in_dim=(self.in_channel+1)

        self.in_layer = nn.Conv2d(in_dim, latent_dim, kernel_size=self.patch_size, padding=self.patch_size//2)

        self.in_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(),
            )
            for _ in range(self.residual_depth-1)
        ])
        self.in_proj.append(nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1))

        self.in_residual = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim*4, kernel_size=1, padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(latent_dim*4, latent_dim, kernel_size=1, padding=0),
            )
            for _ in range(self.residual_depth)
        ])

        self.in_norm = nn.ModuleList([
            nn.LayerNorm(self.latent_dim)
            for _ in range(self.residual_depth)
        ])

        if self.model_type == 'rbf':
            return

        self.proj = nn.ModuleList([])
        in_dim = 2*latent_dim if self.model_type=='concat' else latent_dim
        if proj_depth>0:
            for i in range(proj_depth):
                self.proj.append(
                    nn.ModuleList([
                        nn.Conv2d(in_dim//(2**i), in_dim//(2**(i+1)), kernel_size=1),
                        nn.LayerNorm(in_dim//(2**(i+1))),
                        nn.LeakyReLU(),
                    ])
                )

        in_dim = in_dim//(2**(proj_depth))
        if self.output_activation=='control_sigmoid':
            out_dim = in_channel*3
        else:
            out_dim = in_channel

        self.out = nn.Conv2d(in_dim, out_dim, bias=False, kernel_size=1)

    def pre_activation(self, x, sigma=None):
        # append the noise level map (non-blind only)
        x = self._append_sigma(x, sigma)

        # input layer
        x = self.in_layer(x)

        # residual layers
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()

        return x
    
    def N_theta(self, x, x_s):
        if self.model_type == 'concat':
            # concatinating x and (x_s) shifted x.
            x = torch.cat((x, x_s), dim=1)

        for proj in self.proj:
            for i, sub_layer in enumerate(proj):
                if i == 1: ## LayerNorm layer
                    x = x.permute(0, 2, 3, 1)   # b c h w -> b h w c
                    x = sub_layer(x)
                    x = x.permute(0, 3, 1, 2)   # b h w c -> b c h w


                else:
                    x = sub_layer(x)
        
        x = self.out(x)

        # positive output activation
        if self.output_activation == 'exp':
            x = torch.exp(x/self.sigma**2)

        elif self.output_activation=='control_sigmoid':
            k, a, b = torch.chunk(x, dim=1, chunks=3)
            x = self.controlled_sigmoid(k, a, b) + 1e-8

        elif self.output_activation=='sig':
            normalization_factor = -(self.sigma * self.sigma)
            return torch.nn.functional.sigmoid(x/normalization_factor)

        return x


class NECTRUNet(NECTR_Base):
    r"""
    NECTR non-local kernel denoiser with a U-Net N_theta.

    The NECTR (NonExpansive denoising for ConTractive Reconstruction) denoiser was introduced by
    :footcite:t:`sinha2026trainable`. It is globally nonexpansive by design, so it can be used inside plug-and-play
    algorithms to obtain a contractive reconstruction operator. This implementation is adapted from the
    `official NECTR repository <https://github.com/arghyasinha/nectr>`_ (Model 1, lightweight CNN).

    This denoiser is a learned non-local means filter. It does not predict the clean image directly. It predicts
    non-negative pairwise affinities between pixels and returns a weighted average of the input pixels.
    Only half of the search window is evaluated by the network. The weight computed for the offset :math:`d` is reused
    for the offset :math:`-d`.

    :param torch.device, str device: device to run the model on.
    :param str model_type: how the pixel pair is merged before going into the U-Net. Only ``"concat"`` is supported,
        because the U-Net expects ``2 * in_dim`` input channels.
    :param int in_channel: number of channels of the input image.
    :param int scales: number of downsampling/upsampling scales of the U-Net. The input spatial size must be
        divisible by ``2 ** (scales - 1)``.
    :param int window_rad: radius of the non-local search window. The window has
        ``(2 * window_rad + 1) ** 2`` pixels.
    :param str output_activation: positive activation applied to the output. ``"control_sigmoid"`` uses a learned
        sigmoid :math:`\mathrm{softplus}(a) / (1 + \exp(-\mathrm{sigmoid}(b)\,k))`. This is the only supported
        value: any other value raises a warning and falls back to ``"control_sigmoid"``, so that the weights are
        always positive.
    :param bool blind: if ``True``, the network does not get the noise level as input. If ``False``, ``sig`` must be
        given to :meth:`forward`.

    |sep|

    :Examples:

        >>> import torch
        >>> from deepinv.models.NECTR import NECTRUNet_denoiser
        >>> model = NECTRUNet_denoiser(in_channel=3, scales=2, window_rad=1)
        >>> y = torch.rand(1, 3, 16, 16)
        >>> with torch.no_grad():
        ...     x_hat, Z = model(y, sig=0.1)
        >>> x_hat.shape
        torch.Size([1, 3, 16, 16])

    """
    def __init__(
            self, 
            device: torch.device | str = None, 
            model_type: str = "concat", 
            in_channel: int = 3, 
            scales: int = 4,
            window_rad: int = 7,
            output_activation: str = "control_sigmoid",
            blind: bool = False,
            reference: torch.Tensor = None,
            nlm_type = "non_symm"
        ):
        super(NECTRUNet, self).__init__(nlm_type, device)

        self.scales = scales

        self.sigma = 1
        self.blind = blind
        self.output_activation = self._check_output_activation(output_activation, ("control_sigmoid",))

        self.window_rad = window_rad
        self.in_channel = in_channel
        self.reference = reference

        if blind:
            in_dim=self.in_channel
        else:
            in_dim=(self.in_channel+1)

        if self.output_activation=='control_sigmoid':
            out_dim = self.in_channel*3
        else:
            out_dim = self.in_channel

        self.unet = UNet(
            in_channels=2*in_dim,
            out_channels=self.in_channel,
            scales=self.scales,  # number of downsampling/upsampling scales
            residual=False,
            cat=True,
            batch_norm=False,
        ).to(device)

        self.layer_norm = nn.LayerNorm(self.in_channel)
        self.out = nn.Conv2d(self.in_channel, out_dim, bias=False, kernel_size=1)

    def pre_activation(self, x, sigma=None):
        r"""
        Prepare the reference image for the weight network.

        There is no learned feature extractor in this variant. In the non-blind setting, a constant noise-level map is
        appended to ``x`` as an extra channel. Otherwise ``x`` is returned unchanged.

        :param torch.Tensor x: input (reference) image of shape ``(B, C, H, W)``.
        :param float, int, list, torch.Tensor sigma: noise level. A scalar, or one value per image given as a list or a
            tensor of shape ``(B,)``, ``(B, 1)`` or ``(B, 1, 1, 1)``. Required when ``blind=False``.
        :return: (:class:`torch.Tensor`) tensor of shape ``(B, C, H, W)`` if blind, else ``(B, C + 1, H, W)``.
        """
        return self._append_sigma(x, sigma)

    def N_theta(self, x, x_s):
        r"""
        Compute the pairwise weights :math:`w_{i, i+d} = N_\theta(r, r_{\cdot + d})_i` for one spatial offset
        :math:`d`.

        The reference and its shifted copy are concatenated, then passed through the U-Net, a layer normalization, a
        pointwise convolution and the output activation.

        :param torch.Tensor x: prepared reference (output of :meth:`pre_activation`) of shape ``(B, in_dim, H, W)``.
        :param torch.Tensor x_s: the same tensor shifted by the offset :math:`d`.
        :return: (:class:`torch.Tensor`) weights of shape ``(B, in_channel, H, W)``.
        """
        # concatinating x and (x_s) shifted x.
        x = torch.cat((x, x_s), dim=1)
        x = self.unet(x)
        x = x.permute(0, 2, 3, 1)  # b c h w -> b h w c
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # b h w c -> b c h w

        x = self.out(x)

        if self.output_activation=='control_sigmoid':
            k, a, b = torch.chunk(x, dim=1, chunks=3)
            x = self.controlled_sigmoid(k, a, b) + 1e-8

        return x
