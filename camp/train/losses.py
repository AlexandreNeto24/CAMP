"""Recommender losses.

``compute_loss`` is mean squared error on the heatmap, plus a penalty on
activations that fall on the segmented object. An optional contrastive term
pulls the patch context toward the high-target region and away from
hard negatives.
"""

import torch
import torch.nn.functional as F


def compute_loss(pred_heatmap, target_heatmap, segmentor_mask=None,
                 inpainting_mask=None,
                 patch_features=None, image_features=None,
                 contrastive_weight=0.0,
                 context_weight=0.5,
                 use_context_primary=False):
    """Heatmap MSE, object suppression, and an optional contrastive term.

    ``context_weight`` scales the penalty on predictions inside
    ``segmentor_mask``. ``contrastive_weight`` scales
    ``compute_contrastive_loss``. ``inpainting_mask`` and
    ``use_context_primary`` are accepted so existing callers keep working;
    this function does not read them.
    """
    del inpainting_mask, use_context_primary

    pred_heatmap = pred_heatmap.clamp(min=1e-7, max=1 - 1e-7)
    target_heatmap = target_heatmap.clamp(min=0.0, max=1.0)

    loss_primary = F.mse_loss(pred_heatmap, target_heatmap)

    loss_object_suppression = torch.tensor(0.0, device=pred_heatmap.device)

    if segmentor_mask is not None:
        segmentor_mask = segmentor_mask.clamp(min=0.0, max=1.0)

        if segmentor_mask.shape[-2:] != pred_heatmap.shape[-2:]:
            segmentor_mask = F.interpolate(
                segmentor_mask,
                size=pred_heatmap.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # Predictions should be zero where the segmentor marks an object.
        obj_activation = pred_heatmap * segmentor_mask
        loss_object_suppression = (obj_activation ** 2).mean()

    total_loss = loss_primary + context_weight * loss_object_suppression

    if contrastive_weight > 0 and patch_features is not None and image_features is not None:
        # Positives: the target heatmap is high. Skip empty maps.
        positive_mask = (target_heatmap > 0.5).float()

        if positive_mask.sum() > 0:
            valid_negatives = 1.0 - positive_mask

            # Hard negatives: high confidence where the target is zero.
            hard_negatives = (pred_heatmap.detach() > 0.3).float() * valid_negatives

            # Fall back to every valid negative when hard mining is too thin.
            if hard_negatives.sum() < 100:
                negative_mask = valid_negatives
            else:
                negative_mask = hard_negatives

            loss_contrastive = compute_contrastive_loss(
                patch_features,
                image_features,
                positive_mask,
                negative_mask
            )

            total_loss += contrastive_weight * loss_contrastive

    return total_loss


def compute_contrastive_loss(patch_features, image_features, positive_mask, negative_mask, temperature=0.07):
    """Compare the patch context with positive and negative image regions.

    Args:
        patch_features: Context features of the patch, ``[B, E, h, w]``.
        image_features: Features of the full image, ``[B, E, H, W]``.
        positive_mask: Regions whose context should match the patch.
        negative_mask: Regions whose context should not match the patch.
        temperature: Softmax temperature of the NCE term.
    """
    _batch, channels, _height, _width = patch_features.shape
    _batch, image_channels, feat_h, feat_w = image_features.shape

    shared = min(channels, image_channels)
    patch_features = patch_features[:, :shared, :, :]
    image_features = image_features[:, :shared, :, :]

    # Average only the non-zero pixels. Those are the context.
    patch_pixel_count = (patch_features.abs().sum(dim=1, keepdim=True) > 0).sum(dim=(2, 3)).clamp(min=1)
    patch_context_avg = patch_features.sum(dim=(2, 3)) / patch_pixel_count

    feature_size = (feat_h, feat_w)

    if positive_mask.dim() == 3:
        positive_mask = positive_mask.unsqueeze(1)
    if negative_mask.dim() == 3:
        negative_mask = negative_mask.unsqueeze(1)

    pos_mask_small = F.interpolate(
        positive_mask.float(),
        size=feature_size,
        mode='nearest'
    )
    pos_mask = (pos_mask_small > 0.5).float()

    neg_mask_small = F.interpolate(
        negative_mask.float(),
        size=feature_size,
        mode='nearest'
    )
    neg_mask = (neg_mask_small > 0.5).float()

    pos_features_sum = (image_features * pos_mask).sum(dim=(2, 3))
    pos_count = pos_mask.sum(dim=(2, 3)).clamp(min=1)
    pos_features_avg = pos_features_sum / pos_count

    neg_features_sum = (image_features * neg_mask).sum(dim=(2, 3))
    neg_count = neg_mask.sum(dim=(2, 3)).clamp(min=1)
    neg_features_avg = neg_features_sum / neg_count

    patch_norm = F.normalize(patch_context_avg, dim=1, p=2)
    pos_norm = F.normalize(pos_features_avg, dim=1, p=2)
    neg_norm = F.normalize(neg_features_avg, dim=1, p=2)

    pos_sim = (patch_norm * pos_norm).sum(dim=1, keepdim=True)
    neg_sim = (patch_norm * neg_norm).sum(dim=1, keepdim=True)

    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature

    max_sim = torch.max(pos_sim, neg_sim)
    pos_contrib = torch.exp(pos_sim - max_sim)
    neg_contrib = torch.exp(neg_sim - max_sim)

    log_sum_exp = torch.log(pos_contrib + neg_contrib) + max_sim
    loss_nce = -pos_sim + log_sum_exp

    return loss_nce.mean()
