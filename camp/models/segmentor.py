import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class PretrainedUNet(nn.Module):
    """
    Wrapper para um U-Net de segmentation-models-pytorch para compatibilidade
    com o pipeline existente.
    """
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
        # Determina a dimensão das features do encoder
        # Isto é um valor conhecido para cada arquitetura.
        # Ex: resnet34 -> 512, resnet50 -> 2048, efficientnet-b0 -> 1280, efficientnet-b3 -> 1536
        self.feature_dim = self.model.encoder.out_channels[-1]

    def load_adapted_state_dict(self, checkpoint_path, strict=False):
        """
        Carrega pesos de 3 canais num modelo de 5 canais (CoordConv).
        Estratégia: Copia RGB para 0-2 e inicializa X,Y (3-4) com zeros ou média.
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
                # Caso clássico: Primeira convolução (ex: 64, 5, 3, 3) vs (64, 3, 3, 3)
                if (len(param.shape) == 4 and 
                    own_state[name].shape[0] == param.shape[0] and
                    own_state[name].shape[2:] == param.shape[2:] and
                    own_state[name].shape[1] == self.in_channels and 
                    param.shape[1] == 3):
                    
                    print(f"⚠️ Adaptando layer {name}: {param.shape} -> {own_state[name].shape} (CoordConv Injection)")
                    
                    new_weight = torch.zeros_like(own_state[name])
                    
                    # Copia RGB (canais 0,1,2)
                    new_weight[:, :3, :, :] = param
                    
                    # Inicializa X,Y (canais 3,4) com valores pequenos ou zero
                    # Zero é seguro para não estragar a inferência inicial
                    new_weight[:, 3:, :, :] = 0.0 
                    
                    adapted_state_dict[name] = new_weight
                else:
                    print(f"❌ Mismatch in {name}: {param.shape} vs {own_state[name].shape} - pulando")
            else:
                adapted_state_dict[name] = param
        
        # Carrega
        missing = self.load_state_dict(adapted_state_dict, strict=strict)
        return missing

    def forward(self, x):
        return self.model(x)

    def get_encoder(self):
        """Retorna a parte do encoder para ser usada como feature extractor."""
        return self.model.encoder
