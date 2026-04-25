import torch
import torch.nn as nn
import torchvision

from typing import List, Union, Dict, Tuple


from typing import Union, List, Dict, Any
import torch
import torch.nn as nn
import torchvision


class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor.

    支持两种模式：
    1. 默认模式：只返回 feature_layers 指定层的特征
    2. return_all_intermediate=True：返回中间特征，并标记哪些层发生了下采样

    额外支持：
    - 输入可以是原始 RGB 图像
    - 也可以是来自 VGG 某一层的中间特征，并从该层之后继续提取
    """

    def __init__(
        self,
        feature_layers: Union[int, List[int]] = 34,
        use_bn: bool = False,
        use_input_norm: bool = True,
    ):
        super().__init__()

        self.use_bn = use_bn
        self.use_input_norm = use_input_norm

        if isinstance(feature_layers, int):
            feature_layers = [feature_layers]

        if len(feature_layers) == 0:
            raise ValueError("feature_layers 不能为空。")

        self.feature_layers = sorted(set(feature_layers))
        self.feature_layer_set = set(self.feature_layers)
        self.max_feature_layer = max(self.feature_layers)

        vgg = self._build_vgg(use_bn=use_bn)
        all_layers = list(vgg.features.children())

        if self.max_feature_layer >= len(all_layers):
            raise ValueError(
                f"feature_layers 中最大层号为 {self.max_feature_layer}，"
                f"但当前 VGG 只有 {len(all_layers)} 个 feature 层（最大索引为 {len(all_layers) - 1}）。"
            )

        # 只保留到最大需求层
        self.features = nn.Sequential(*all_layers[: self.max_feature_layer + 1])

        self.layer_names = [
            layer.__class__.__name__
            for layer in all_layers[: self.max_feature_layer + 1]
        ]

        if self.use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval().cuda()

    def _build_vgg(self, use_bn: bool):
        try:
            if use_bn:
                weights = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
                vgg = torchvision.models.vgg19_bn(weights=weights)
            else:
                weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
                vgg = torchvision.models.vgg19(weights=weights)
        except AttributeError:
            if use_bn:
                vgg = torchvision.models.vgg19_bn(pretrained=True)
            else:
                vgg = torchvision.models.vgg19(pretrained=True)
        return vgg

    def forward(
        self,
        x: torch.Tensor,
        return_all_intermediate: bool = False,
        input_layer: int = -1,
    ) -> Union[
        torch.Tensor,
        List[torch.Tensor],
        Dict[str, Any],
    ]:
        """
        Args:
            x:
                - 当 input_layer == -1 时，x 为原始输入图像，范围默认 [0, 1]
                - 当 input_layer >= 0 时，x 为 VGG 第 input_layer 层的输出特征
            return_all_intermediate:
                是否返回所有中间特征
            input_layer:
                输入特征来自 VGG 的哪一层输出。
                -1 表示输入是原始图像
                >=0 表示输入已经是该层输出，将从下一层继续提取

        Returns:
            当 return_all_intermediate=False 时：
                - 单层目标返回 Tensor
                - 多层目标返回 List[Tensor]

            当 return_all_intermediate=True 时：
                返回一个 dict
        """
        if input_layer < -1:
            raise ValueError("input_layer 不能小于 -1。")
        if input_layer > self.max_feature_layer:
            raise ValueError(
                f"input_layer={input_layer} 超过当前 extractor 可处理的最大层 {self.max_feature_layer}。"
            )

        # raw image input
        if input_layer == -1:
            if self.use_input_norm:
                x = (x - self.mean) / self.std
            start_idx = 0
        else:
            # x 已经是某一层输出特征，不再做 input norm
            start_idx = input_layer + 1

        all_features: List[torch.Tensor] = []
        selected_features: List[torch.Tensor] = []
        layer_ids: List[int] = []
        collected_layer_names: List[str] = []
        is_selected_list: List[bool] = []
        is_downsample_list: List[bool] = []

        prev_hw = x.shape[-2:]

        # 如果输入本身就是某一层输出，并且用户需要这一层，那么先收进去
        if input_layer >= 0:
            if return_all_intermediate:
                all_features.append(x)
                layer_ids.append(input_layer)
                collected_layer_names.append(self.layer_names[input_layer])
                is_selected_list.append(input_layer in self.feature_layer_set)
                is_downsample_list.append(False)  # 当前输入本身不是新经过一层算子得到的

            if input_layer in self.feature_layer_set:
                selected_features.append(x)

        # 继续从下一层开始跑
        for idx in range(start_idx, len(self.features)):
            layer = self.features[idx]
            x = layer(x)
            curr_hw = x.shape[-2:]

            is_downsample = (curr_hw[0] < prev_hw[0]) or (curr_hw[1] < prev_hw[1])

            if return_all_intermediate:
                all_features.append(x)
                layer_ids.append(idx)
                collected_layer_names.append(self.layer_names[idx])
                is_selected_list.append(idx in self.feature_layer_set)
                is_downsample_list.append(is_downsample)

            if idx in self.feature_layer_set:
                selected_features.append(x)

            prev_hw = curr_hw

        if len(selected_features) == 0:
            raise ValueError(
                f"没有提取到任何目标层特征。当前 input_layer={input_layer}，"
                f"feature_layers={self.feature_layers}。"
            )

        if not return_all_intermediate:
            if len(selected_features) == 1:
                return selected_features[0]
            return selected_features

        if len(selected_features) == 1:
            selected_output = selected_features[0]
        else:
            selected_output = selected_features

        downsample_layer_ids = [
            lid for lid, flag in zip(layer_ids, is_downsample_list) if flag
        ]

        return {
            "all_features": all_features,
            "selected_features": selected_output,
            "layer_ids": layer_ids,
            "selected_layer_ids": self.feature_layers,
            "layer_names": collected_layer_names,
            "is_selected": is_selected_list,
            "is_downsample": is_downsample_list,
            "downsample_layer_ids": downsample_layer_ids,
            "input_layer": input_layer,
        }


class PerceptualLoss(nn.Module):
    """Perceptual Loss, based on VGG features."""

    def __init__(
        self,
        feature_layers: Union[int, List[int]] = 34,
        use_bn: bool = False,
        use_input_norm: bool = True,
        loss_type: str = "l1",
    ):
        """
        Initialize perceptual loss.

        Args:
            feature_layers: VGG layer indices to extract. int for single layer, List[int] for multiple layers.
            use_bn: Whether to use VGG with Batch Normalization.
            use_input_norm: Whether to apply ImageNet normalization to inputs.
            loss_type: Distance metric type, "l1" or "l2".
        """
        super().__init__()
        self.feature_layers = feature_layers
        self.multi_scale = isinstance(feature_layers, list)

        self.feature_extractor = VGGFeatureExtractor(
            feature_layers=feature_layers,
            use_bn=use_bn,
            use_input_norm=use_input_norm,
        )

        if loss_type == "l1":
            self.distance_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.distance_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        weights: Union[float, List[float]] = 1.0,
    ) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            sr: Super-resolution image, range [0, 1].
            hr: High-resolution reference image, range [0, 1].
            weights: Weight for each layer's features. float for single layer, List[float] for multiple layers.

        Returns:
            Perceptual loss value.
        """
        # Extract features
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)

        # Compute loss
        if not self.multi_scale:
            # Single layer features
            return self.distance_fn(sr_features, hr_features) * weights

        # Multi-layer features
        if isinstance(weights, (int, float)):
            weights = [weights] * len(sr_features)

        loss = torch.tensor(0.0, device=sr.device)
        for sf, hf, w in zip(sr_features, hr_features, weights):
            loss += self.distance_fn(sf, hf) * w

        return loss


class SPRPerceptualLoss(nn.Module):
    """
    Super-Resolution Patch-Replacement Perceptual Loss (SPRPerceptualLoss).

    A multi-scale perceptual loss that evaluates SR quality using patch-replacement
    at multiple resolutions. Contains three parallel branches:
    - Branch A: Standard perceptual loss (SR vs HR)
    - Branch B: SPR_0 perceptual loss (initial hybrid vs HR)
    - Branch C: SPR_1...N perceptual loss (downsampled hybrids vs HR)

    Notes:
    - spr_patch_size 表示输入 mask 在原图分辨率上的 patch size
    - 每经过一次 VGG 的下采样层，对应的 patch size 也 / 2
    - 如果下一次下采样后的 patch size < 1，则停止继续构造新的下采样 mask
    """

    def __init__(
        self,
        feature_layers: List[int] = [2, 7, 17, 25, 34],
        use_bn: bool = False,
        use_input_norm: bool = True,
        loss_type: str = "l1",
        loss_weights: Dict[str, float] = None,
    ):
        super().__init__()

        self.feature_layers = sorted(set(feature_layers))

        if loss_weights is None:
            loss_weights = {
                "sr": 0.5,
                "spr_0": 1.0,
                "spr_multi": 0.1,
            }
        self.loss_weights = loss_weights

        self.feature_extractor = VGGFeatureExtractor(
            feature_layers=self.feature_layers,
            use_bn=use_bn,
            use_input_norm=use_input_norm,
        )

        if loss_type == "l1":
            self.distance_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.distance_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    @staticmethod
    def _to_feature_list(features):
        if isinstance(features, torch.Tensor):
            return [features]
        return list(features)

    def _normalize_weights(self, weights: Union[float, List[float]]) -> List[float]:
        if isinstance(weights, (int, float)):
            return [float(weights)] * len(self.feature_layers)

        weights = list(weights)
        if len(weights) != len(self.feature_layers):
            raise ValueError(
                f"weights length ({len(weights)}) must match feature_layers length "
                f"({len(self.feature_layers)})."
            )
        return [float(w) for w in weights]

    @staticmethod
    def _normalize_patch_size(
        spr_patch_size: Union[int, Tuple[int, int], List[int]]
    ) -> Tuple[float, float]:
        if isinstance(spr_patch_size, int):
            patch_h, patch_w = spr_patch_size, spr_patch_size
        elif isinstance(spr_patch_size, (tuple, list)) and len(spr_patch_size) == 2:
            patch_h, patch_w = spr_patch_size
        else:
            raise ValueError(
                "spr_patch_size must be int or tuple/list of length 2, "
                f"but got {spr_patch_size}"
            )

        patch_h = float(patch_h)
        patch_w = float(patch_w)

        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"spr_patch_size must be positive, but got {(patch_h, patch_w)}"
            )

        return patch_h, patch_w

    def __cal_loss(
        self,
        hr_feature_info: Dict,
        input_feature_info: Dict,
        weights: Union[float, List[float]],
    ) -> torch.Tensor:
        hr_loss_layers = self._to_feature_list(
            hr_feature_info.get("selected_features", [])
        )
        input_loss_layers = self._to_feature_list(
            input_feature_info.get("selected_features", [])
        )
        weights = self._normalize_weights(weights)

        if len(hr_loss_layers) != len(self.feature_layers):
            raise ValueError(
                f"HR selected feature count mismatch: "
                f"{len(hr_loss_layers)} vs {len(self.feature_layers)}"
            )
        if len(input_loss_layers) != len(self.feature_layers):
            raise ValueError(
                f"Input selected feature count mismatch: "
                f"{len(input_loss_layers)} vs {len(self.feature_layers)}"
            )

        loss = hr_loss_layers[0].new_tensor(0.0)
        for hlf, ilf, w in zip(hr_loss_layers, input_loss_layers, weights):
            if hlf.shape != ilf.shape:
                raise ValueError(
                    f"Feature shape mismatch: HR {hlf.shape} vs Input {ilf.shape}"
                )
            loss = loss + self.distance_fn(hlf, ilf) * w
        return loss

    def __get_loss(
        self,
        hr_feature_info: Dict,
        input_feature: torch.Tensor,
        weights: Union[float, List[float]],
        input_layer: int = -1,
    ) -> torch.Tensor:
        """
        input_feature:
            - input_layer = -1: raw RGB image
            - input_layer >= 0: VGG 第 input_layer 层输出特征
        """
        target_layers = [x for x in self.feature_layers if x > input_layer]

        if len(target_layers) == 0:
            return input_feature.new_tensor(0.0)

        input_feature_info = self.feature_extractor(
            input_feature,
            return_all_intermediate=True,
            input_layer=input_layer,
        )

        input_loss_layers = self._to_feature_list(
            input_feature_info.get("selected_features", [])
        )
        if len(input_loss_layers) != len(target_layers):
            raise ValueError(
                f"Input selected feature count mismatch after input_layer={input_layer}: "
                f"{len(input_loss_layers)} vs expected {len(target_layers)}"
            )

        hr_loss_layers = self._to_feature_list(
            hr_feature_info.get("selected_features", [])
        )
        if len(hr_loss_layers) != len(self.feature_layers):
            raise ValueError(
                f"HR selected feature count mismatch: "
                f"{len(hr_loss_layers)} vs {len(self.feature_layers)}"
            )

        weights = self._normalize_weights(weights)

        start_idx = self.feature_layers.index(target_layers[0])
        hr_target_layers = hr_loss_layers[start_idx:]
        target_weights = weights[start_idx:]

        loss = hr_target_layers[0].new_tensor(0.0)
        for hlf, ilf, w in zip(hr_target_layers, input_loss_layers, target_weights):
            if hlf.shape != ilf.shape:
                raise ValueError(
                    f"Feature shape mismatch: HR {hlf.shape} vs Input {ilf.shape}"
                )
            loss = loss + self.distance_fn(hlf, ilf) * w
        return loss

    @staticmethod
    def downsample_mask_patches(mask: torch.Tensor, scale: int = 2) -> torch.Tensor:
        """
        Downsample binary patch mask by majority vote.

        For each scale x scale block:
        - sum > half  -> 1
        - else        -> 0

        Note:
        - tie goes to 0
        - H and W must be divisible by scale
        """
        if scale < 1:
            raise ValueError(f"scale must >= 1, got {scale}")

        B, C, H, W = mask.shape
        if H % scale != 0 or W % scale != 0:
            raise ValueError(
                f"Mask size {(H, W)} must be divisible by scale={scale}"
            )

        new_H, new_W = H // scale, W // scale
        if new_H == 0 or new_W == 0:
            return mask

        x = mask.contiguous().view(B, C, new_H, scale, new_W, scale)
        block_sum = x.sum(dim=(3, 5))
        threshold = (scale * scale) / 2.0
        downsampled = (block_sum > threshold).to(mask.dtype)
        return downsampled

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        mask: torch.Tensor,
        spr_patch_size: Union[int, Tuple[int, int], List[int]],
        weights: Union[float, List[float]] = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sr:
                super-resolved image
            hr:
                high-resolution image
            mask:
                binary patch mask at image resolution
            spr_patch_size:
                patch size of the input mask.
                - int: means square patch, e.g. 8 -> (8, 8)
                - tuple/list: (patch_h, patch_w)
            weights:
                perceptual loss weights

        Returns:
            {
                "total_loss": ...,
                "loss_sr": ...,
                "loss_spr_0": ...,
                "loss_spr_multi": ...,
            }
        """
        if sr.shape != hr.shape:
            raise ValueError(f"sr.shape {sr.shape} != hr.shape {hr.shape}")

        if mask.shape[0] != sr.shape[0] or mask.shape[-2:] != sr.shape[-2:]:
            raise ValueError(
                f"mask.shape {mask.shape} is not compatible with sr.shape {sr.shape}"
            )

        patch_h, patch_w = self._normalize_patch_size(spr_patch_size)

        weights = self._normalize_weights(weights)
        mask = mask.detach().to(sr.dtype)

        # ========================================
        # P1: SR vs HR
        # ========================================
        sr_feature_info = self.feature_extractor(sr, return_all_intermediate=True)
        hr_feature_info = self.feature_extractor(hr, return_all_intermediate=True)

        loss_sr = self.__cal_loss(hr_feature_info, sr_feature_info, weights)

        # ========================================
        # P2: SPR_0
        # ========================================
        spr_0 = hr * mask + (1.0 - mask) * sr
        loss_spr_0 = self.__get_loss(
            hr_feature_info, spr_0, weights, input_layer=-1
        )

        # ========================================
        # P3: SPR_1...N
        # ========================================
        all_hr_features = hr_feature_info.get("all_features", [])
        all_sr_features = sr_feature_info.get("all_features", [])

        now_mask = mask.clone().detach()[:, 0, :, :]
        now_mask = now_mask.unsqueeze(1)

        now_patch_h = patch_h
        now_patch_w = patch_w

        loss_spr_multi_list = []

        for dsl_idx in hr_feature_info.get("downsample_layer_ids", []):
            # 下一层特征会再下采样一次，因此 patch size 也要 /2
            next_patch_h = now_patch_h / 2.0
            next_patch_w = now_patch_w / 2.0

            # 如果下一次下采样后 patch size < 1，则停止继续创建新的 mask
            if next_patch_h < 1.0 or next_patch_w < 1.0:
                break

            now_mask = self.downsample_mask_patches(now_mask, scale=2)
            now_patch_h = next_patch_h
            now_patch_w = next_patch_w

            srf = all_sr_features[dsl_idx]
            hrf = all_hr_features[dsl_idx]

            if srf.shape != hrf.shape:
                raise ValueError(
                    f"SR/HR feature mismatch at layer {dsl_idx}: "
                    f"{srf.shape} vs {hrf.shape}"
                )

            if now_mask.shape[-2:] != srf.shape[-2:]:
                raise ValueError(
                    f"Mask shape {now_mask.shape[-2:]} does not match feature shape "
                    f"{srf.shape[-2:]} at layer {dsl_idx}"
                )

            spr = hrf * now_mask + (1.0 - now_mask) * srf
            loss_spr_multi_list.append(
                self.__get_loss(
                    hr_feature_info,
                    spr,
                    weights,
                    input_layer=dsl_idx,
                )
            )

        if len(loss_spr_multi_list) == 0:
            loss_spr_multi = sr.new_tensor(0.0)
        else:
            loss_spr_multi = torch.stack(loss_spr_multi_list).sum()

        total_loss = (
            self.loss_weights["sr"] * loss_sr
            + self.loss_weights["spr_0"] * loss_spr_0
            + self.loss_weights["spr_multi"] * loss_spr_multi
        )

        return {
            "total_loss": total_loss,
            "loss_sr": loss_sr,
            "loss_spr_0": loss_spr_0,
            "loss_spr_multi": loss_spr_multi,
        }