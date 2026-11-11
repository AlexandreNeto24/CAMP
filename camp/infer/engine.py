"""Test-time placement search.

Angles are collapsed to the four ``rot90`` orientations before the forward.
Each box carries the one winning scale, horizontal flip, and angle.
``CAMP.blend`` applies that triple to the context crop and its mask.
"""

import torch
import torch.nn.functional as F

from camp.infer.extractors import extract_recommendations_from_heatmap


def collapse_rot90_angles(angles):
    """Keep the first angle in each ``angle // 90`` group.

    ``torch.rot90(k=angle // 90)`` maps 0, 15, 30, ... onto the same pixels.
    Keeping the first member preserves iteration order, including the
    identity config (angle 0) that receives the +0.001 tie bonus.
    """
    seen = set()
    unique = []
    for angle in angles:
        key = int(angle) // 90
        if key in seen:
            continue
        seen.add(key)
        unique.append(angle)
    return unique


def _pool_max(heatmaps, patch_hw):
    """Max of an average pool whose kernel is the patch, clamped to the map."""
    patch_h, patch_w = patch_hw
    height, width = heatmaps.shape[-2:]
    kernel_h = min(int(patch_h), int(height))
    kernel_w = min(int(patch_w), int(width))
    if kernel_h > 0 and kernel_w > 0:
        pooled = F.avg_pool2d(heatmaps, kernel_size=(kernel_h, kernel_w), stride=1)
        return pooled.amax(dim=(1, 2, 3))
    return heatmaps.amax(dim=(1, 2, 3))


def _regional_scores(heatmaps, batch_patches):
    """One pooling reduction per distinct patch size in this scale batch."""
    sizes = [tuple(int(size) for size in patch.shape[-2:]) for patch in batch_patches]
    if len(set(sizes)) == 1:
        return _pool_max(heatmaps, sizes[0])

    scores = heatmaps.new_empty(heatmaps.shape[0])
    groups = {}
    for index, size in enumerate(sizes):
        groups.setdefault(size, []).append(index)
    for size, indices in groups.items():
        index_tensor = torch.tensor(indices, device=heatmaps.device, dtype=torch.long)
        pooled = _pool_max(heatmaps.index_select(0, index_tensor), size)
        scores.index_copy_(0, index_tensor, pooled)
    return scores


def _expand_features(features, batch_size):
    """Repeat a batch-1 encoder feature list across the TTO batch."""
    expanded = []
    for feature in features:
        if feature.shape[0] == batch_size:
            expanded.append(feature)
        elif feature.shape[0] == 1:
            expanded.append(feature.expand(batch_size, -1, -1, -1))
        else:
            raise ValueError(
                "Cannot expand encoder features from batch "
                f"{feature.shape[0]} to {batch_size}."
            )
    return expanded


class RecommendationEngine:
    """Run the recommender over scales, flips, and rot90 orientations."""

    def __init__(self, model):
        self.model = model

    def recommend_placements(
        self,
        image,
        patch,
        patch_with_context=None,
        patch_box=None,
        top_k=5,
        use_tto=True,
        tto_angles=None,
        tto_flips=None,
        tto_scales=None,
        precomputed_image_features=None,
    ):
        """Score placements and return up to ``top_k`` boxes.

        With TTO on, scales are 0.9, 1.0, and 1.1, flips are off then on, and
        angles are every 15 degrees collapsed to four ``rot90`` orientations.
        The identity geometry (scale 1, angle 0, no flip) gets +0.001 so it
        wins exact ties. Every returned box stores that one winning triple.
        ``precomputed_image_features`` is an eval-only encoder list; it is
        ignored while the module is training.
        """
        if tto_angles is None:
            tto_angles = list(range(0, 360, 15))
        if tto_flips is None:
            tto_flips = [False, True]
        if tto_scales is None:
            tto_scales = [0.9, 1.0, 1.1]

        device = self.model.device
        image = image.to(device)
        patch = patch.to(device)

        if patch_with_context is not None:
            patch_with_context = patch_with_context.to(device)

        if image.shape[0] != 1:
            raise ValueError("Inference currently supports only batch_size=1")

        if not use_tto:
            tto_angles = [0]
            tto_flips = [False]
            tto_scales = [1.0]

        tto_angles = collapse_rot90_angles(tto_angles)

        image_feature_bank = None
        if not self.model.training:
            if precomputed_image_features is None:
                with torch.no_grad():
                    image_feature_bank = [
                        feature.detach()
                        for feature in self.model.feature_extractor(image)
                    ]
            else:
                image_feature_bank = precomputed_image_features

        best_score = -float("inf")
        best_heatmap = None
        best_config = None
        best_patch_tensor = None

        for scale in tto_scales:
            curr_patch = patch
            curr_context = patch_with_context

            if scale != 1.0:
                curr_patch = F.interpolate(
                    curr_patch,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                if curr_context is not None:
                    curr_context = F.interpolate(
                        curr_context,
                        scale_factor=scale,
                        mode="bilinear",
                        align_corners=False,
                        recompute_scale_factor=False,
                    )

            base_patch_scaled = curr_patch[0]
            base_context_scaled = curr_context[0] if curr_context is not None else None

            batch_patches = []
            batch_contexts = []
            batch_configs = []

            for flip in tto_flips:
                for angle in tto_angles:
                    transformed_patch = base_patch_scaled
                    transformed_context = base_context_scaled

                    if flip:
                        transformed_patch = torch.flip(transformed_patch, dims=[-1])
                        if transformed_context is not None:
                            transformed_context = torch.flip(transformed_context, dims=[-1])

                    if angle != 0:
                        turns = int(angle) // 90
                        transformed_patch = torch.rot90(
                            transformed_patch, k=turns, dims=[-2, -1]
                        )
                        if transformed_context is not None:
                            transformed_context = torch.rot90(
                                transformed_context, k=turns, dims=[-2, -1]
                            )

                    batch_patches.append(transformed_patch)
                    batch_contexts.append(transformed_context)
                    batch_configs.append({"scale": scale, "angle": angle, "flip": flip})

            input_patches = torch.stack(batch_patches)
            input_contexts = (
                torch.stack(batch_contexts) if batch_contexts[0] is not None else None
            )
            input_images = image.repeat(len(batch_configs), 1, 1, 1)

            exclusion_mask = None
            if input_patches.shape[1] == 4:
                exclusion_mask = (input_patches[:, 3:4] > 0.1).float()
                input_patches_rgb = input_patches[:, :3]
                if input_contexts is not None and input_contexts.shape[1] == 4:
                    input_contexts_rgb = input_contexts[:, :3]
                else:
                    input_contexts_rgb = input_contexts
            else:
                input_patches_rgb = input_patches
                input_contexts_rgb = input_contexts

            feature_batch = None
            if image_feature_bank is not None:
                feature_batch = _expand_features(image_feature_bank, input_images.shape[0])

            with torch.no_grad():
                heatmaps, _ = self.model(
                    input_images,
                    patch=input_patches_rgb,
                    patch_with_context=input_contexts_rgb,
                    context_mask=exclusion_mask,
                    return_attention=True,
                    precomputed_image_features=feature_batch,
                )

            # One transfer of the per-geometry maxima. The +0.001 identity
            # bonus is applied in float64, matching the previous per-item path.
            score_values = _regional_scores(heatmaps, batch_patches).detach().cpu().tolist()
            current_batch_best_score = -float("inf")
            current_batch_best_idx = -1
            for index, score in enumerate(score_values):
                config = batch_configs[index]
                if (
                    config["scale"] == 1.0
                    and config["angle"] == 0
                    and not config["flip"]
                ):
                    score += 0.001
                if score > current_batch_best_score:
                    current_batch_best_score = score
                    current_batch_best_idx = index

            if current_batch_best_idx != -1 and current_batch_best_score > best_score:
                best_score = current_batch_best_score
                best_heatmap = heatmaps[current_batch_best_idx].unsqueeze(0)
                best_config = batch_configs[current_batch_best_idx]
                best_patch_tensor = batch_patches[current_batch_best_idx]

        winner_h, winner_w = best_patch_tensor.shape[-2:]

        output_size = None
        if patch_box is not None:
            orig_w = patch_box[2] - patch_box[0]
            orig_h = patch_box[3] - patch_box[1]
            scale = best_config["scale"]
            output_size = (orig_h * scale, orig_w * scale)

        recommendations = extract_recommendations_from_heatmap(
            best_heatmap[0, 0],
            image.shape[-2:],
            (winner_h, winner_w),
            top_k=top_k,
            output_size=output_size,
        )

        for rec in recommendations:
            rec["scale"] = best_config["scale"]
            rec["angle"] = best_config["angle"]
            rec["flip"] = best_config["flip"]

        return recommendations
