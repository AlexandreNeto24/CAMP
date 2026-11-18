"""U-Net wrapper. ``forward`` returns logits, not probabilities."""

import logging

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PretrainedUNet(nn.Module):
    """segmentation-models-pytorch U-Net++. ``forward`` returns logits."""
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_dropout=0.2):
        super().__init__()
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_dropout=decoder_dropout
        )
        # Width of the encoder's last feature map.
        # resnet34 is 512, resnet50 is 2048, efficientnet-b0 is 1280,
        # efficientnet-b3 is 1536.
        self.feature_dim = self.model.encoder.out_channels[-1]

    def load_adapted_state_dict(self, checkpoint_path, strict=False):
        """Load a 3-channel checkpoint into a model with extra channels.

        RGB weights are copied into channels 0–2. Channels 3 and 4, used
        for coordinate maps, stay zero so the first forward pass is unchanged.
        """
        device = next(self.parameters()).device
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        own_state = self.state_dict()
        
        adapted_state_dict = {}
        
        for name, param in state_dict.items():
            if name not in own_state:
                continue
                
            if own_state[name].shape != param.shape:
                # First convolution grew from 3 input channels to in_channels.
                if (len(param.shape) == 4 and 
                    own_state[name].shape[0] == param.shape[0] and
                    own_state[name].shape[2:] == param.shape[2:] and
                    own_state[name].shape[1] == self.in_channels and 
                    param.shape[1] == 3):
                    
                    logger.warning(
                        "Adapting layer %s: %s -> %s",
                        name,
                        tuple(param.shape),
                        tuple(own_state[name].shape),
                    )
                    
                    new_weight = torch.zeros_like(own_state[name])
                    
                    new_weight[:, :3, :, :] = param
                    new_weight[:, 3:, :, :] = 0.0 
                    
                    adapted_state_dict[name] = new_weight
                else:
                    logger.warning(
                        "Shape mismatch in %s: %s vs %s; skipping",
                        name,
                        tuple(param.shape),
                        tuple(own_state[name].shape),
                    )
            else:
                adapted_state_dict[name] = param
        
        missing = self.load_state_dict(adapted_state_dict, strict=strict)
        return missing

    def forward(self, x):
        return self.model(x)

    def get_encoder(self):
        """Return the encoder, for use as a feature extractor."""
        return self.model.encoder
