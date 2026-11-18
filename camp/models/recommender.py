"""Attention recommender.

The forward pass returns a heatmap that is already a probability: the decoder
ends in a sigmoid. Image features may be supplied precomputed, but only while
the module is in eval mode.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionRecommender(nn.Module):
    """Cross-attend a context query to a target image and decode a heatmap.

    ``embedding_dim`` is 2048 and ``num_heads`` is 16 in every checkpoint.
    ``dropout_rate`` is stored so train and load share one constructor. The
    live decoder does not apply it. ``skip_idx`` is -3.
    """

    def __init__(self, backbone_model, embedding_dim=2048, num_heads=16, dropout_rate=0.10):
        super().__init__()
        self.backbone = backbone_model
        self.feature_extractor = self.backbone.get_encoder()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        concat_dim, skip_channels = self._probe_feature_dims()
        self._build_projections(concat_dim, skip_channels)
        self._build_decoder()
        self._build_attention()

    def _probe_feature_dims(self):
        """Run one 1x3x512x512 forward so concat and skip widths match the encoder.

        This probe is load-bearing. ``concat_dim`` and ``skip_channels`` are
        read off ``features[-1]``, ``features[-2]``, and ``features[-3]``.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512)
            features = self.feature_extractor(dummy)
            self.skip_idx = -3
            skip_channels = features[self.skip_idx].shape[1]
            target_size = features[-1].shape[-2:]
            upsampled = F.interpolate(
                features[-2], size=target_size, mode="bilinear", align_corners=False
            )
            concat_dim = upsampled.shape[1] + features[-1].shape[1]
        return concat_dim, skip_channels

    def _build_projections(self, concat_dim, skip_channels):
        self.query_proj = nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.similarity_proj = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def _build_decoder(self):
        """Decoder whose input is 64 channels. Layers [0:3] run before the skip concat."""
        mid_channels = 64
        self.heatmap_decoder = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def _build_attention(self):
        self.head_dim = self.embedding_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True,
        )

    def extract_features(self, image, return_skip=False, precomputed_features=None):
        """Fuse the last two encoder maps. Optionally also return the skip map.

        ``precomputed_features`` is used only in eval mode. Its ``[-1]``,
        ``[-2]``, and ``[-3]`` entries must match a normal encoder forward.
        Train mode always runs the encoder.
        """
        device = next(self.feature_extractor.parameters()).device
        if precomputed_features is not None and not self.training:
            features = [feature.to(device) for feature in precomputed_features]
        else:
            image = image.to(device)
            features = self.feature_extractor(image)

        final_feat = features[-1]
        if isinstance(features, (list, tuple)):
            target_size = features[-1].shape[-2:]
            upsampled = F.interpolate(
                features[-2], size=target_size, mode="bilinear", align_corners=False
            )
            final_feat = torch.cat([upsampled, features[-1]], dim=1)

        if return_skip:
            return final_feat, features[self.skip_idx]
        return final_feat

    def forward(
        self,
        image,
        lesion_coords=None,
        patch=None,
        patch_with_context=None,
        source_context=None,
        source_box=None,
        weight_mask=None,
        context_mask=None,
        return_attention=False,
        return_intermediate=False,
        precomputed_image_features=None,
    ):
        """Match the patch context against ``image`` and return a probability heatmap.

        Unused arguments are kept so existing callers do not break.
        ``precomputed_image_features`` is ignored unless the module is in eval.
        """
        device = self.device
        image = image.to(device)
        patch = patch.to(device)

        if patch_with_context is None:
            patch_with_context = patch
        else:
            patch_with_context = patch_with_context.to(device)

        if context_mask is not None:
            context_mask = context_mask.to(device)

        image_features, image_skip = self.extract_features(
            image,
            return_skip=True,
            precomputed_features=precomputed_image_features,
        )

        exclusion_mask = None
        if context_mask is not None:
            exclusion_mask = context_mask
        elif patch is not None:
            exclusion_mask = (patch.abs().sum(dim=1, keepdim=True) > 0.1).float()

        _, patch_ctx_features = self.get_patch_features(
            patch=patch,
            patch_with_context=patch_with_context,
            context_mask=exclusion_mask,
        )

        query_proj = self.query_proj(patch_ctx_features)
        key_proj = self.key_proj(image_features)
        value_proj = self.value_proj(image_features)

        batch, channels, height, width = key_proj.shape
        patch_h, patch_w = query_proj.shape[-2:]

        ctx_mask_small = F.interpolate(
            exclusion_mask.float(), size=(patch_h, patch_w), mode="nearest"
        )
        context_mask_inv = (1.0 - ctx_mask_small).squeeze(1) > 0.5

        query_flat = query_proj.permute(0, 2, 3, 1)
        query_avg = self._masked_query_mean(query_flat, context_mask_inv)

        key = key_proj.view(batch, channels, -1).permute(0, 2, 1)
        value = value_proj.view(batch, channels, -1).permute(0, 2, 1)

        _attn_output, attn_weights = self.context_attention(
            query=query_avg, key=key, value=value
        )
        similarity_map = attn_weights.view(batch, 1, height, width)
        decoded = self.similarity_proj(similarity_map)

        # Layers [0:3], then the skip concatenation, then layers [3:].
        # Do not replace this with heatmap_decoder(decoded).
        decoded = self.heatmap_decoder[0](decoded)
        decoded = self.heatmap_decoder[1](decoded)
        decoded = self.heatmap_decoder[2](decoded)

        skip = self.skip_proj(image_skip)
        if decoded.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(
                skip, size=decoded.shape[-2:], mode="bilinear", align_corners=False
            )
        decoded = torch.cat([decoded, skip], dim=1)

        for layer_index in range(3, len(self.heatmap_decoder)):
            decoded = self.heatmap_decoder[layer_index](decoded)

        heatmap = F.interpolate(
            decoded, size=image.shape[-2:], mode="bilinear", align_corners=False
        )

        if return_attention and return_intermediate:
            return heatmap, None, patch_ctx_features, image_features
        if return_attention:
            return heatmap, None
        if return_intermediate:
            return heatmap, patch_ctx_features, image_features
        return heatmap

    def _masked_query_mean(self, query_flat, token_mask):
        """Mean of query tokens inside ``token_mask``. An empty mask stays zero.

        ``query_flat`` is ``[B, H, W, E]`` and ``token_mask`` is ``[B, H, W]``.
        """
        mask = token_mask.to(dtype=query_flat.dtype).unsqueeze(-1)
        counts = mask.sum(dim=(1, 2))
        summed = (query_flat * mask).sum(dim=(1, 2))
        mean = summed / counts.clamp(min=1)
        mean = torch.where(counts > 0, mean, torch.zeros_like(mean))
        return mean.unsqueeze(1)

    def get_patch_features(self, patch, patch_with_context, context_mask):
        """Split one context encoding into object and background features.

        ``context_mask`` is 1 on the object and 0 on the background.
        """
        features_with_context = self.extract_features(patch_with_context)
        mask_small = F.interpolate(
            context_mask.float(),
            size=features_with_context.shape[-2:],
            mode="nearest",
        )
        object_mask = (mask_small > 0.5).float()
        context_mask_inv = 1.0 - object_mask
        patch_object_features = features_with_context * object_mask
        patch_context_features = features_with_context * context_mask_inv
        return patch_object_features, patch_context_features

    def train(self, mode=True):
        """Keep the frozen encoder in eval while the decoder follows ``mode``."""
        super().train(mode)
        if hasattr(self, "backbone"):
            self.backbone.eval()
        if hasattr(self, "feature_extractor"):
            self.feature_extractor.eval()
        return self

    @property
    def device(self):
        """Device of the first parameter."""
        return next(self.parameters()).device

    def save_model(self, path):
        """Save trainable weights. The encoder is not included."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.") and not key.startswith("feature_extractor.")
        }
        metadata = {
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "version": "1.0",
        }
        torch.save({"state_dict": state_dict, "metadata": metadata}, path)
        logger.debug("Saved recommender to %s", path)

    def load_model(self, path):
        """Load a checkpoint. Unknown keys, including removed modules, are ignored."""
        checkpoint = torch.load(path, map_location=self.device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=False)
        logger.debug("Loaded recommender from %s", path)
        return self


def build_attention_recommender(
    segmentor,
    embedding_dim=2048,
    num_heads=16,
    dropout_rate=0.10,
):
    """Build the recommender used by both ``CAMP.load`` and training.

    ``embedding_dim=2048`` and ``num_heads=16`` are the checkpoint widths.
    ``dropout_rate`` defaults to 0.10 on both paths.
    """
    return AttentionRecommender(
        backbone_model=segmentor,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )
