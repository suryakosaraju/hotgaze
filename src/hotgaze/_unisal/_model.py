# Copyright 2024 Richard Droste
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: Vendored from https://github.com/rdroste/unisal (commit: HEAD)
# for HotGaze deep saliency inference. Modified: stripped training code,
# dataset references, KwConfigClass dependency, and global device.
# Original: unisal/unisal/model.py

"""Vendored UNISAL model — inference subset only."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from ._cgru import ConvGRU
from ._mobilenet import InvertedResidual, MobileNetV2


def _log_softmax(x):
    """Log-softmax over spatial dimensions. From unisal/unisal/utils.py."""
    x_size = x.size()
    x = x.view(x.size(0), -1)
    x = F.log_softmax(x, dim=1)
    return x.view(x_size)


class DomainBatchNorm2d(nn.Module):
    """Domain-specific 2D BatchNorm — from unisal/unisal/model.py."""

    def __init__(self, num_features, sources, momenta=None, **kwargs):
        super().__init__()
        self.sources = sources
        if momenta is None:
            momenta = [0.1] * len(sources)
        self.momenta = momenta
        if "momentum" in kwargs:
            del kwargs["momentum"]
        for src, mnt in zip(sources, self.momenta):
            self.__setattr__(f"bn_{src}", nn.BatchNorm2d(num_features, momentum=mnt, **kwargs))
        self.this_source = None

    def forward(self, x):
        return self.__getattr__(f"bn_{self.this_source}")(x)


# Default backbone CNN kwargs
default_cnn_cfg = {
    "widen_factor": 1.0,
    "pretrained": True,
    "input_channel": 32,
    "last_channel": 1280,
}

# Default RNN kwargs
default_rnn_cfg = {
    "kernel_size": (3, 3),
    "gate_ksize": (3, 3),
    "dropout": (False, True, False),
    "drop_prob": (0.2, 0.2, 0.2),
    "mobile": True,
}


class UNISAL(nn.Module):
    """UNISAL saliency model — inference-only vendored copy.

    Original: Richard Droste, https://github.com/rdroste/unisal (Apache 2.0).
    Modified: removed training code, KwConfigClass, global device.
    """

    def __init__(
        self,
        rnn_input_channels=256,
        rnn_hidden_channels=256,
        cnn_cfg=None,
        rnn_cfg=None,
        res_rnn=True,
        bypass_rnn=True,
        drop_probs=(0.0, 0.6, 0.6),
        gaussian_init="manual",
        n_gaussians=16,
        smoothing_ksize=41,
        bn_momentum=0.01,
        static_bn_momentum=0.1,
        sources=("DHF1K", "Hollywood", "UCFSports", "SALICON"),
        ds_bn=True,
        ds_adaptation=True,
        ds_smoothing=True,
        ds_gaussians=True,
        verbose=1,
    ):
        super().__init__()

        assert gaussian_init in ("random", "manual")
        if bypass_rnn:
            assert res_rnn
        if n_gaussians > 0 and gaussian_init == "manual":
            n_gaussians = 16

        self.rnn_input_channels = rnn_input_channels
        self.rnn_hidden_channels = rnn_hidden_channels
        this_cnn_cfg = default_cnn_cfg.copy()
        this_cnn_cfg.update(cnn_cfg or {})
        self.cnn_cfg = this_cnn_cfg
        this_rnn_cfg = default_rnn_cfg.copy()
        this_rnn_cfg.update(rnn_cfg or {})
        self.rnn_cfg = this_rnn_cfg
        self.bypass_rnn = bypass_rnn
        self.res_rnn = res_rnn
        self.drop_probs = drop_probs
        self.gaussian_init = gaussian_init
        self.n_gaussians = n_gaussians
        self.smoothing_ksize = smoothing_ksize
        self.bn_momentum = bn_momentum
        self.sources = sources
        self.ds_bn = ds_bn
        self.static_bn_momentum = static_bn_momentum
        self.ds_adaptation = ds_adaptation
        self.ds_smoothing = ds_smoothing
        self.ds_gaussians = ds_gaussians
        self.verbose = verbose

        # Backbone CNN
        self.cnn = MobileNetV2(**self.cnn_cfg)

        # Post-CNN module
        post_cnn = [
            (
                "inv_res",
                InvertedResidual(
                    self.cnn.out_channels + n_gaussians,
                    rnn_input_channels,
                    1,
                    1,
                    bn_momentum=bn_momentum,
                ),
            )
        ]
        if self.drop_probs[0] > 0:
            post_cnn.insert(0, ("dropout", nn.Dropout2d(self.drop_probs[0], inplace=False)))
        self.post_cnn = nn.Sequential(OrderedDict(post_cnn))

        # Bypass-RNN
        if sources != ("SALICON",) or not self.bypass_rnn:
            self.rnn = ConvGRU(
                rnn_input_channels,
                hidden_channels=[rnn_hidden_channels],
                batchnorm=self.get_bn_module,
                **self.rnn_cfg,
            )
            self.post_rnn = self.conv_1x1_bn(rnn_hidden_channels, rnn_input_channels)

        # Upsampling 1
        self.upsampling_1 = nn.Sequential(OrderedDict([("us1", self.upsampling(2))]))

        channels_2x = 128
        self.skip_2x = self.make_skip_connection(
            self.cnn.feat_2x_channels, channels_2x, 2, self.drop_probs[1]
        )

        # Upsampling 2
        self.upsampling_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "inv_res",
                        InvertedResidual(
                            rnn_input_channels + channels_2x,
                            channels_2x,
                            1,
                            2,
                            batchnorm=self.get_bn_module,
                        ),
                    ),
                    ("us2", self.upsampling(2)),
                ]
            )
        )

        channels_4x = 64
        self.skip_4x = self.make_skip_connection(
            self.cnn.feat_4x_channels, channels_4x, 2, self.drop_probs[2]
        )

        # Post-US2
        self.post_upsampling_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "inv_res",
                        InvertedResidual(
                            channels_2x + channels_4x,
                            channels_4x,
                            1,
                            2,
                            batchnorm=self.get_bn_module,
                        ),
                    ),
                ]
            )
        )

        # Domain-specific modules
        for source_str in self.sources:
            source_str = f"_{source_str}".lower()
            if n_gaussians > 0:
                self.set_gaussians(source_str)
            self.__setattr__(
                "adaptation" + (source_str if self.ds_adaptation else ""),
                nn.Sequential(*[nn.Conv2d(channels_4x, 1, 1, bias=True)]),
            )
            smoothing = nn.Conv2d(1, 1, kernel_size=smoothing_ksize, padding=0, bias=False)
            with torch.no_grad():
                gaussian = self._make_gaussian_maps(
                    smoothing.weight.data, torch.Tensor([[[0.5, -2]] * 2])
                )
                gaussian /= gaussian.sum()
                smoothing.weight.data = gaussian
            self.__setattr__("smoothing" + (source_str if self.ds_smoothing else ""), smoothing)

    @property
    def this_source(self):
        return self._this_source

    @this_source.setter
    def this_source(self, source):
        for module in self.modules():
            if isinstance(module, DomainBatchNorm2d):
                module.this_source = source
        self._this_source = source

    def get_bn_module(self, num_features, **kwargs):
        momenta = [
            self.bn_momentum if src != "SALICON" else self.static_bn_momentum
            for src in self.sources
        ]
        if self.ds_bn:
            return DomainBatchNorm2d(num_features, self.sources, momenta=momenta, **kwargs)
        return nn.BatchNorm2d(num_features, **kwargs)

    def upsampling(self, factor):
        return nn.Sequential(
            *[nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=False)]
        )

    def set_gaussians(self, source_str, prefix="coarse_"):
        suffix = source_str if self.ds_gaussians else ""
        self.__setattr__(
            prefix + "gaussians" + suffix,
            self._initialize_gaussians(self.n_gaussians),
        )

    def _initialize_gaussians(self, n_gaussians):
        from itertools import product

        if self.gaussian_init == "manual":
            gaussians = torch.Tensor(
                [
                    list(product([0.25, 0.5, 0.75], repeat=2))
                    + [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75)]
                    + [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)]
                    + [(0.5, 0.5)],
                    [(-1.5, -1.5)] * 9 + [(0, -1.5)] * 3 + [(-1.5, 0)] * 3 + [(0, 0)],
                ]
            ).permute(1, 2, 0)
        elif self.gaussian_init == "random":
            with torch.no_grad():
                gaussians = torch.stack(
                    [
                        torch.randn(n_gaussians, 2, dtype=torch.float) * 0.1 + 0.5,
                        torch.randn(n_gaussians, 2, dtype=torch.float) * 0.2 - 1,
                    ],
                    dim=2,
                )
        else:
            raise NotImplementedError
        return nn.Parameter(gaussians, requires_grad=True)

    @staticmethod
    def _make_gaussian_maps(x, gaussians, size=None, scaling=6.0):
        if size is None:
            size = x.shape[-2:]
            bs = x.shape[0]
        else:
            size = [size] * 2
            bs = 1
        dtype = x.dtype
        device = x.device
        gaussian_maps = []
        map_template = torch.ones(*size, dtype=dtype, device=device)
        meshgrids = torch.meshgrid(
            [
                torch.linspace(0, 1, size[0], dtype=dtype, device=device),
                torch.linspace(0, 1, size[1], dtype=dtype, device=device),
            ],
            indexing="ij",
        )
        for yx_mu_logstd in torch.unbind(gaussians):
            m = map_template.clone()
            for mu_logstd, mgrid in zip(yx_mu_logstd, meshgrids):
                mu = mu_logstd[0]
                std = torch.exp(mu_logstd[1])
                m *= torch.exp(-(((mgrid - mu) / std) ** 2) / 2)
            m *= scaling
            gaussian_maps.append(m)
        gaussian_maps = torch.stack(gaussian_maps)
        gaussian_maps = gaussian_maps.unsqueeze(0).expand(bs, -1, -1, -1)
        return gaussian_maps

    def _get_gaussian_maps(self, x, source_str, prefix="coarse_", **kwargs):
        suffix = source_str if self.ds_gaussians else ""
        gaussians = self.__getattr__(prefix + "gaussians" + suffix)
        return self._make_gaussian_maps(x, gaussians, **kwargs)

    def make_skip_connection(self, input_channels, output_channels, expand_ratio, p, inplace=False):
        hidden_channels = round(input_channels * expand_ratio)
        return nn.Sequential(
            OrderedDict(
                [
                    ("expansion", self.conv_1x1_bn(input_channels, hidden_channels)),
                    ("dropout", nn.Dropout2d(p, inplace=inplace)),
                    (
                        "reduction",
                        nn.Sequential(
                            *[
                                nn.Conv2d(hidden_channels, output_channels, 1),
                                self.get_bn_module(output_channels),
                            ]
                        ),
                    ),
                ]
            )
        )

    def conv_1x1_bn(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            self.get_bn_module(oup),
            nn.ReLU6(inplace=True),
        )

    def forward(
        self, x, target_size=None, h0=None, return_hidden=False, source="SALICON", static=None
    ):
        if target_size is None:
            target_size = x.shape[-2:]
        self.this_source = source
        source_str = f"_{source.lower()}"
        if static is None:
            static = x.shape[1] == 1 or self.sources == ("SALICON",)

        feat_seq_1x, feat_seq_2x, feat_seq_4x = [], [], []
        for img in torch.unbind(x, dim=1):
            im_feat_1x, im_feat_2x, im_feat_4x = self.cnn(img)
            im_feat_2x = self.skip_2x(im_feat_2x)
            im_feat_4x = self.skip_4x(im_feat_4x)
            if self.n_gaussians > 0:
                gm = self._get_gaussian_maps(im_feat_1x, source_str)
                im_feat_1x = torch.cat((im_feat_1x, gm), dim=1)
            im_feat_1x = self.post_cnn(im_feat_1x)
            feat_seq_1x.append(im_feat_1x)
            feat_seq_2x.append(im_feat_2x)
            feat_seq_4x.append(im_feat_4x)

        feat_seq_1x = torch.stack(feat_seq_1x, dim=1)
        hidden, rnn_feat_seq, rnn_feat = (None,) * 3
        if not (static and self.bypass_rnn):
            rnn_feat_seq, hidden = self.rnn(feat_seq_1x, hidden=h0)

        output_seq = []
        for idx, im_feat in enumerate(torch.unbind(feat_seq_1x, dim=1)):
            if not (static and self.bypass_rnn):
                rnn_feat = rnn_feat_seq[:, idx, ...]
                rnn_feat = self.post_rnn(rnn_feat)
                if self.res_rnn:
                    im_feat = im_feat + rnn_feat
                else:
                    im_feat = rnn_feat
            im_feat = self.upsampling_1(im_feat)
            im_feat = torch.cat((im_feat, feat_seq_2x[idx]), dim=1)
            im_feat = self.upsampling_2(im_feat)
            im_feat = torch.cat((im_feat, feat_seq_4x[idx]), dim=1)
            im_feat = self.post_upsampling_2(im_feat)
            im_feat = self.__getattr__("adaptation" + (source_str if self.ds_adaptation else ""))(
                im_feat
            )
            im_feat = F.interpolate(im_feat, size=x.shape[-2:], mode="nearest")
            im_feat = F.pad(im_feat, [self.smoothing_ksize // 2] * 4, mode="replicate")
            im_feat = self.__getattr__("smoothing" + (source_str if self.ds_smoothing else ""))(
                im_feat
            )
            im_feat = F.interpolate(im_feat, size=target_size, mode="bilinear", align_corners=False)
            im_feat = _log_softmax(im_feat)
            output_seq.append(im_feat)

        output_seq = torch.stack(output_seq, dim=1)
        if return_hidden:
            return output_seq, hidden
        return output_seq
