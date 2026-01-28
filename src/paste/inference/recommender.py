import torch
import torch.nn.functional as F
from .extractors import extract_recommendations_from_heatmap


class RecommendationEngine:
    """Encapsula lógica de recomendação para um modelo treinado"""
    
    def __init__(self, model):
        self.model = model
    
    # ✅ ATUALIZADO: Suporte Robusto a TTO (Escala + Rotação + Flip)
    def recommend_placements(self, image, patch, patch_with_context=None, patch_box=None, top_k=5, 
                             use_tto=True, tto_angles=[0, 90, 180, 270], tto_flips=[False, True],
                             tto_scales=[0.9, 1.0, 1.1]):
        """
        Recomenda locais para colagem testando variações geométricas (TTO).
        Estratégia generalista:
        - Escalas: Redução (0.9), Original (1.0), Ampliação (1.1)
        - Rotações: Ortogonais (0, 90, 180, 270)
        - Flips: Horizontal
        """
        device = self.model.device
        image = image.to(device)
        patch = patch.to(device)
        
        if patch_with_context is not None:
            patch_with_context = patch_with_context.to(device)

        if image.shape[0] != 1:
            raise ValueError("Inference currently supports only batch_size=1")

        # Define padrões se TTO estiver desligado ou listas vazias
        if not use_tto:
            tto_angles = [0]
            tto_flips = [False]
            tto_scales = [1.0]

        best_score = -float('inf')
        best_heatmap = None
        best_config = None
        best_patch_tensor = None
        
        # Loop Externo: Escalas (Tamanhos de tensor diferentes impedem batch único simples)
        for scale in tto_scales:
            # 1. Aplica Escala (Zoom)
            curr_patch = patch
            curr_context = patch_with_context
            
            if scale != 1.0:
                # Interpolação bilinear
                curr_patch = F.interpolate(curr_patch, scale_factor=scale, mode='bilinear', align_corners=False, recompute_scale_factor=False)
                if curr_context is not None:
                    curr_context = F.interpolate(curr_context, scale_factor=scale, mode='bilinear', align_corners=False, recompute_scale_factor=False)

            # 2. Gera batch para Variações Geométricas (Rotação/Flip) desta escala
            base_patch_scaled = curr_patch[0]
            base_context_scaled = curr_context[0] if curr_context is not None else None
            
            batch_patches = []
            batch_contexts = []
            batch_configs = []

            for flip in tto_flips:
                for angle in tto_angles:
                    # Aplica flip?
                    p_t = base_patch_scaled
                    c_t = base_context_scaled
                    
                    if flip:
                        p_t = torch.flip(p_t, dims=[-1]) # Flip horizontal
                        if c_t is not None:
                            c_t = torch.flip(c_t, dims=[-1])
                    
                    # Aplica rotação?
                    if angle != 0:
                        # rot90 requer k=1, 2, 3 (90, 180, 270)
                        k = angle // 90
                        p_t = torch.rot90(p_t, k=k, dims=[-2, -1])
                        if c_t is not None:
                            c_t = torch.rot90(c_t, k=k, dims=[-2, -1])
                    
                    batch_patches.append(p_t)
                    batch_contexts.append(c_t)
                    batch_configs.append({'scale': scale, 'angle': angle, 'flip': flip})

            # Empilha batch local
            input_patches = torch.stack(batch_patches)
            input_contexts = torch.stack(batch_contexts) if batch_contexts[0] is not None else None
            input_images = image.repeat(len(batch_configs), 1, 1, 1)

            # Tratamento Alpha/Mask
            exclusion_mask = None
            if input_patches.shape[1] == 4:
                exclusion_mask = (input_patches[:, 3:4] > 0.1).float()
                input_patches_rgb = input_patches[:, :3]
                if input_contexts is not None and input_contexts.shape[1] == 4:
                    input_contexts_rgb = input_contexts[:, :3]
                else:
                    input_contexts_rgb = input_contexts # Assume RGB se não tiver channel 4
            else:
                input_patches_rgb = input_patches
                input_contexts_rgb = input_contexts

            # Inferência em Batch
            with torch.no_grad():
                heatmaps, _ = self.model(
                    input_images, 
                    patch=input_patches_rgb, 
                    patch_with_context=input_contexts_rgb, 
                    context_mask=exclusion_mask,
                    return_attention=True
                )

            # --- AVALIAÇÃO ROBUSTA (Regional vs Pixel) ---
            # Em vez de olhar para o pixel máximo, usamos Average Pooling com o tamanho do patch.
            # Isto simula a extração de regiões e garante que escolhemos a geometria
            # que maximiza a CONFIANÇA MÉDIA na área de colagem, não apenas um ponto de ruído.
            
            current_batch_best_score = -float('inf')
            current_batch_best_idx = -1
            
            for i in range(len(batch_configs)):
                hm = heatmaps[i] # [1, H, W]
                # O tamanho do kernel deve ser o tamanho do patch (transformado)
                # batch_patches já tem as rotações e escalas aplicadas, logo tem o H,W correto.
                ph, pw = batch_patches[i].shape[-2:]
                
                # Proteção contra kernels maiores que a imagem (casos extremos de zoom/rot)
                kh = min(ph, hm.shape[-2])
                kw = min(pw, hm.shape[-1])

                # Média local usando AvgPool (kernel size = tamanho do patch)
                # Isso acha a região onde a "densidade" de atenção é máxima para este tamanho de patch
                if kh > 0 and kw > 0:
                    local_avg = F.avg_pool2d(hm.unsqueeze(0), kernel_size=(kh, kw), stride=1)
                    score = local_avg.max().item()
                else:
                    score = hm.max().item() # Fallback

                # Penaliza configurações extremas levemente para preferir 'original' em empate
                # ex: scale=1.0 ganha de scale=1.1 se score for igual
                encourage_standard = 0.0
                if batch_configs[i]['scale'] == 1.0 and batch_configs[i]['angle'] == 0 and not batch_configs[i]['flip']:
                    encourage_standard = 0.001 
                
                score += encourage_standard

                if score > current_batch_best_score:
                    current_batch_best_score = score
                    current_batch_best_idx = i

            # Se encontrou algo válido neste batch, compara com o melhor global
            if current_batch_best_idx != -1 and current_batch_best_score > best_score:
                best_score = current_batch_best_score
                best_heatmap = heatmaps[current_batch_best_idx].unsqueeze(0) # [1, 1, H, W]
                best_config = batch_configs[current_batch_best_idx]
                best_patch_tensor = batch_patches[current_batch_best_idx]

        # --- FIM DO LOOP ---
        
        winner_h, winner_w = best_patch_tensor.shape[-2:]
        
        # Determina tamanho de saída (para a caixa de colagem)
        output_size = None
        if patch_box is not None:
             # Calcula dimensões originais: W = x2-x1, H = y2-y1
            orig_w = patch_box[2] - patch_box[0]
            orig_h = patch_box[3] - patch_box[1]
            
            # Aplica escala do TTO vencedor
            scale = best_config['scale']
            final_w = orig_w * scale
            final_h = orig_h * scale
            
            output_size = (final_h, final_w)

        recommendations = extract_recommendations_from_heatmap(
            best_heatmap[0, 0], 
            image.shape[-2:], 
            (winner_h, winner_w), 
            top_k=top_k,
            output_size=output_size # Passa tamanho corrigido se disponível
        )
        
        # Anexa os metadados da geometria vencedora
        for rec in recommendations:
            rec['scale'] = best_config['scale']
            rec['angle'] = best_config['angle']
            rec['flip'] = best_config['flip']
        
        return recommendations