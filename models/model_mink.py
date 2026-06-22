# from kan import *
from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import MultiheadAttention
import MinkowskiEngine as ME
from torch.cuda.amp import autocast


class SparseConvPadding(nn.Module):
    def __init__(
        self, 
        width: int = 1024,
        conv_fn: Callable = None,
    ) -> None:
        super().__init__()

        self.width = width
        self.conv = conv_fn
        padding_dim = self.conv.kernel_generator.kernel_size[0]
        padding_dilation = self.conv.kernel_generator.kernel_dilation[0]
        self.padding = ((padding_dim-1)*padding_dilation+1)//2

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        if self.padding == 0:
            return self.conv(x)

        offset = torch.tensor([0, self.width, 0, 0], dtype=torch.int32, device=x.C.device)
        coords_x, features_x = x.C, x.F
        padding_left = coords_x[:, 1] >= (self.width//2 - self.padding * x.tensor_stride[0])
        padding_right = coords_x[:, 1] < -(self.width//2 - self.padding * x.tensor_stride[0])
        padding_coords = torch.cat([
            coords_x[padding_left, :] - offset,
            coords_x,
            coords_x[padding_right, :] + offset,
        ], dim=0)
        padding_features = torch.cat([
            features_x[padding_left, :],
            features_x,
            features_x[padding_right, :],
        ], dim=0)

        sort_idx = padding_coords[:, 0].argsort()

        u = ME.SparseTensor(
            coordinates=padding_coords[sort_idx],
            features=padding_features[sort_idx],
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
        )
        v = self.conv(u)
        coords_v, features_v = v.C, v.F
        mask = (coords_v[:, 1] >= -self.width//2) & (coords_v[:, 1] < self.width//2)
        y = ME.SparseTensor(
            coordinates=coords_v[mask, :],
            features=features_v[mask, :],
            coordinate_manager=v.coordinate_manager,
            tensor_stride=v.tensor_stride,
        )

        return y
    

def MinkowskiSparseTensorCat(sparse_tensors: List[ME.SparseTensor], extra_sparse_tensors: List[ME.SparseTensor] = []) -> ME.SparseTensor:
    return ME.SparseTensor(
        coordinates=sparse_tensors[0].C,
        features=torch.cat(
            [sparse_tensors[0].F] + [st.F for st in sparse_tensors[1:]] + [st.features_at_coordinates(sparse_tensors[0].C.float()) for st in extra_sparse_tensors],
            dim=1
        ),
        coordinate_manager=sparse_tensors[0].coordinate_manager,
        tensor_stride=sparse_tensors[0].tensor_stride,
    )


class SparseMaxPoolBlock(nn.Sequential):
    def __init__(
        self,
        feat_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        *,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: Union[int, List[int], Tuple[int, ...]] = 1,
        width: int = 1024,
    ) -> None:
        super().__init__(
            ME.MinkowskiMaxPooling(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(feat_channels),
            ME.MinkowskiLeakyReLU(inplace=False),
        )


class SparseConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        *,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: Union[int, List[int], Tuple[int, ...]] = 1,
        width: int = 1024,
    ) -> None:
        super().__init__(
            SparseConvPadding(
                width=width,
                conv_fn=ME.MinkowskiConvolution(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dimension=3,
                )
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiLeakyReLU(inplace=False),
        )


class SparseConvTransposeBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        *,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: Union[int, List[int], Tuple[int, ...]] = 1,
        width: int = 1024,
    ) -> None:
        super().__init__(
            SparseConvPadding(
                width=width,
                conv_fn=ME.MinkowskiConvolutionTranspose(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dimension=3,
                )
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiLeakyReLU(inplace=False),
        )


class SparseResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        *,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: Union[int, List[int], Tuple[int, ...]] = 1,
        width: int = 1024,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            SparseConvPadding(
                width=width,
                conv_fn=ME.MinkowskiConvolution(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dimension=3,
                )
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiLeakyReLU(inplace=False),
            SparseConvPadding(
                width=width,
                conv_fn=ME.MinkowskiConvolution(
                    in_channels=out_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dimension=3,
                )
            ),
            ME.MinkowskiBatchNorm(out_channels),
        )

        if in_channels != out_channels or np.prod(stride) != 1:
            self.shortcut = nn.Sequential(
                SparseConvPadding(
                    width=width,
                    conv_fn=ME.MinkowskiConvolution(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        dimension=3,
                    )
                ),
                ME.MinkowskiBatchNorm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = ME.MinkowskiLeakyReLU(inplace=False)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        x = self.relu(self.main(x) + self.shortcut(x))
        return x


class SparseFFNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)
        self.nonlinear = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=False),
        )
        self.res = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        ffn_f = self.linear(x.F)
        ffn_f = self.nonlinear(ffn_f)
        res_f = self.res(x.F)
        y = ME.SparseTensor(
            features=ffn_f + res_f,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key,
            tensor_stride=x.tensor_stride,
        )
        return y


class RPGE(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        encoder_channels: List[int],
        decoder_channels: List[int],
        in_channels: int = 4,
        head_num: int = 4,
        max_down_layers: int = 4,
        width: int = 1024,
    ) -> None:
        super().__init__()
        self.stem_channels = stem_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.in_channels = in_channels
        self.head_num = head_num
        self.max_down_layers = max_down_layers
        num_channels = [stem_channels] + encoder_channels + decoder_channels

        self.len_encoder_channels = len(encoder_channels)
        self.len_decoder_channels = len(decoder_channels)

        assert self.len_encoder_channels >= self.len_decoder_channels
        assert self.head_num % 2 == 0

        self.stem = nn.Sequential(
            SparseFFNBlock(
                in_channels,
                num_channels[0] // 2,
            ),
            SparseFFNBlock(
                num_channels[0] // 2,
                num_channels[0],
            ),
        )

        self.encoders = nn.ModuleList()
        for k in range(self.len_encoder_channels):
            if k < self.max_down_layers:
                down_stride = 2
                down_dilation = 1
                conv_dilation = [1, 1, 1]
            else:
                down_stride = 1
                down_dilation = 2**(k - self.max_down_layers)
                conv_dilation = 2**(k + 1 - self.max_down_layers)
            self.encoders.append(
                nn.ModuleDict(
                    {
                        "downsample": SparseConvBlock(
                            num_channels[k],
                            num_channels[k],
                            2,
                            stride=down_stride,
                            dilation=down_dilation,
                            width=width,
                        ),
                        "maxpool": SparseMaxPoolBlock(
                            num_channels[k],
                            2,
                            stride=down_stride,
                            dilation=down_dilation,
                            width=width,
                        ),
                        "res": ME.MinkowskiConvolution(
                            num_channels[k],
                            num_channels[k],
                            2,
                            stride=down_stride,
                            dilation=down_dilation,
                            dimension=3
                        ),
                        "fuse": nn.Sequential(
                            SparseResBlock(
                                num_channels[k] + num_channels[k] + num_channels[k], 
                                num_channels[k + 1], 
                                3,
                                dilation=conv_dilation,
                                width=width,
                            ),
                            SparseResBlock(
                                num_channels[k + 1], 
                                num_channels[k + 1], 
                                3,
                                dilation=conv_dilation,
                                width=width,
                            ),
                        ),
                    }
                )
            )

        self.decoders = nn.ModuleList()
        for k in range(self.len_decoder_channels):
            if k > self.len_encoder_channels - self.max_down_layers:
                up_stride = 2
                up_dilation = 1
                conv_dilation = [1, 1, 1]
            else:
                up_stride = 1
                up_dilation = 2**(-k - 1 + self.len_encoder_channels - self.max_down_layers)
                conv_dilation = 2**(-k + self.len_encoder_channels - self.max_down_layers)
            self.decoders.append(
                nn.ModuleDict(
                    {
                        "upsample": SparseConvTransposeBlock(
                            num_channels[self.len_encoder_channels + k],
                            num_channels[self.len_encoder_channels + k + 1],
                            2,
                            stride=up_stride,
                            dilation=up_dilation,
                            width=width,
                        ),
                        "fuse": nn.Sequential(
                            SparseResBlock(
                                num_channels[self.len_encoder_channels + k + 1] + num_channels[self.len_encoder_channels - k - 1],
                                num_channels[self.len_encoder_channels + k + 1],
                                3,
                                dilation=conv_dilation,
                                width=width,
                            ),
                            SparseResBlock(
                                num_channels[self.len_encoder_channels + k + 1],
                                num_channels[self.len_encoder_channels + k + 1],
                                3,
                                dilation=conv_dilation,
                                width=width,
                            ),
                        ),
                    }
                )
            )


    def _unet_forward(
        self,
        x: ME.SparseTensor,
        encoders: nn.ModuleList,
        decoders: nn.ModuleList,
    ) -> ME.SparseTensor:
        if not encoders and not decoders:
            return x

        # downsample
        xc = encoders[0]["fuse"](
            MinkowskiSparseTensorCat(
                [encoders[0]["downsample"](x), encoders[0]["res"](x)], [encoders[0]["maxpool"](x)]
            )
        )

        # inner recursion
        yd = self._unet_forward(xc, 
                                encoders[1:], 
                                decoders[:-1] if len(decoders) == len(encoders) else decoders)

        # upsample and fuse
        if len(encoders) == len(decoders):
            y = decoders[-1]["fuse"](
                MinkowskiSparseTensorCat(
                    [decoders[-1]["upsample"](yd)], [x]
                )
            )
        else:
            y = yd

        return y

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        stem_x = self.stem(x)
        return self._unet_forward(stem_x, self.encoders, self.decoders)


class MMRegressor(nn.Module):
    def __init__(self,
        feat_channels: int = 512,
        head_channels: int = 512,
        out_channels: int = 4,
        head_num: int = 4,
        layers: int = 5,
    ) -> None:
        super(MMRegressor, self).__init__()
        assert head_channels % 2 == 0
        self.feat_channels = feat_channels
        self.head_channels = head_channels
        self.head_num = head_num
        self.head_dim = head_channels // head_num
        self.out_channels = out_channels
        self.layers = layers
        self.blocks = nn.ModuleList()
        self.relu = nn.LeakyReLU(inplace=False)
        self.proj_in = nn.Linear(self.feat_channels, self.head_channels) if self.feat_channels != self.head_channels else nn.Identity()

        for _ in range(self.layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "mlp": nn.Sequential(
                            nn.Linear(self.head_channels, self.head_channels),
                            nn.LayerNorm(self.head_channels),
                            nn.LeakyReLU(inplace=False),
                            nn.Linear(self.head_channels, self.head_channels * self.head_num),
                            nn.LayerNorm(self.head_channels * self.head_num),
                        ),
                        "norm": nn.LayerNorm(self.head_channels),
                    }
                )
            )

        self.pred_out = nn.Sequential(
            nn.Linear(self.head_channels, self.head_channels),
            nn.LayerNorm(self.head_channels),
            nn.LeakyReLU(inplace=False),
            nn.Linear(self.head_channels, self.out_channels) if self.head_channels != self.out_channels else nn.Identity(),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        input_x = self.proj_in(x)

        for block in self.blocks:
            mlp_x = block["mlp"](input_x)
            mlp_max = mlp_x.view(*mlp_x.shape[:-1], self.head_channels, self.head_num).max(dim=-1)[0]
            input_x = self.relu(block["norm"](input_x + mlp_max))
            
        pred_x = self.pred_out(input_x)

        return pred_x


class LEADER(nn.Module):
    def __init__(self,
        in_channels: int = 3,
        out_channels: int = 3,
        feat_channels: int = 512,
        width: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder = RPGE(
            stem_channels=32,
            encoder_channels=[32, 64, 128, 256, 384],
            decoder_channels=[feat_channels],
            in_channels=in_channels,
            max_down_layers=4,
            width=width,
        )
        self.decoder = MMRegressor(
            feat_channels=feat_channels, 
            head_channels=feat_channels, 
            out_channels=out_channels, 
            head_num=4,
            layers=5,
        )

    def forward(self, input):
        raise NotImplementedError