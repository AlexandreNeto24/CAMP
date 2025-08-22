import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # Adicionar este import
from pathlib import Path
import os
import sys
from torchvision.ops import roi_align
# Configuração corrigida de caminhos
project_root = Path(__file__).parent.parent.parent  # src/paste/ → src/ → project_root
sys.path.insert(0, str(project_root))

# Adicione o caminho para o ultralytics clonado
ultralytics_path = os.path.abspath(os.path.join(project_root, "src", "ultralytics"))
sys.path.insert(0, ultralytics_path)

# Adicione tratamento de erro para melhor diagnóstico
try:
    from ultralytics import YOLO
    print("Ultralytics importado com sucesso no paste_recommender.py!")
except ImportError as e:
    print(f"Erro ao importar ultralytics no paste_recommender.py: {e}")
    print(f"Caminhos de busca: {sys.path}")
    sys.exit(1)
    
class SimpleContextClassifier(nn.Module):
    """
    Classificador simples que avalia se um patch de lesão é compatível com uma região alvo.
    Substitui o complexo sistema de recomendação com um modelo mais simples e direto.
    """
    def __init__(self, backbone_path, embedding_size=256):
        super().__init__()
        # Carregar backbone pré-treinado (detector YOLO)
        self.backbone = YOLO(backbone_path).model.model[:6]
        
        # Congelar backbone para estabilidade
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Pooling global para obter features fixas
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determinar dimensão das features de forma robusta
        dummy_input = torch.zeros(1, 3, 128, 128).to(next(self.backbone.parameters()).device)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            pooled = self.avgpool(features).flatten(1)
            feature_dim = pooled.shape[1]
        
        # Classificador simples para score 0-1
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim*2, embedding_size),  # Concatena features de região e patch
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_size, 1),
            nn.Sigmoid()
        )
        
        # Armazenar dimensão para referência
        self.feature_dim = feature_dim
    
    def forward(self, region, patch):
        """
        Classifica se um patch combina bem com uma região
        
        Args:
            region: Tensor da região alvo [B,C,H,W]
            patch: Tensor do patch de lesão [B,C,H,W]
            
        Returns:
            Pontuação de compatibilidade (0-1)
        """
        device = next(self.backbone.parameters()).device
        region = region.to(device)
        patch = patch.to(device)
        
        # Garantir dimensão de batch
        if region.dim() == 3:
            region = region.unsqueeze(0)
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        
        # Extrair features
        with torch.no_grad():
            region_feats = self.backbone(region)
            patch_feats = self.backbone(patch)
        
        # Lidar com qualquer formato de features (lista/tupla ou tensor único)
        if isinstance(region_feats, (list, tuple)):
            region_feats = region_feats[-1]
            patch_feats = patch_feats[-1]
        
        # Pooling global e flatten
        region_feat = self.avgpool(region_feats).flatten(1)
        patch_feat = self.avgpool(patch_feats).flatten(1)
        
        # Garantir que as features têm a dimensão esperada
        if region_feat.shape[1] != self.feature_dim:
            region_feat = F.adaptive_avg_pool1d(region_feat.unsqueeze(1), self.feature_dim).squeeze(1)
        if patch_feat.shape[1] != self.feature_dim:
            patch_feat = F.adaptive_avg_pool1d(patch_feat.unsqueeze(1), self.feature_dim).squeeze(1)
        
        # Concatenar
        combined = torch.cat([region_feat, patch_feat], dim=1)
        
        # Classificar
        return self.classifier(combined)

class PasteRecommender(nn.Module):
    """
    Módulo simplificado de recomendação de colagem baseado em um detector YOLO pré-treinado.
    """
    def __init__(self, detector_path, embedding_dim=128):
        super().__init__()
        # Carrega detector pré-treinado
        self.detector = YOLO(detector_path)
        self.device = next(self.detector.model.parameters()).device
        
        # Obtém dimensão das features do backbone do detector
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 640, 640).to(self.device)
            features = self.detector.model.model[:6](dummy_input)
            if isinstance(features, (list, tuple)):
                in_channels = sum([f.shape[1] for f in features[-2:]])
            else:
                in_channels = features.shape[1]
        
        # Embedding simples para contexto (único componente mantido)
        self.context_head = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        ).to(self.device)
        
        # ADICIONAR: Codificadores separados para objeto e contexto
        self.object_encoder = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        ).to(self.device)
        
        self.context_encoder = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        ).to(self.device)
        
        # ADICIONAR: Módulo de relação objeto-contexto
        self.relation_module = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        ).to(self.device)
        
        # Flag para controle interno de estado de treinamento
        self._training = True
    
    def initialize_feature_cache(self, max_cache_items=100):
        """
        Inicializa um cache para features extraídas
        
        Args:
            max_cache_items: Tamanho máximo do cache
        """
        self.feature_cache = {}
        self.feature_cache_keys = []
        self.max_cache_items = max_cache_items
        self.cache_enabled = True
        
    def extract_features_with_cache(self, image, cache_key=None):
        """
        Versão da extração de features com suporte a cache
        
        Args:
            image: Tensor da imagem
            cache_key: Chave para o cache (opcional)
            
        Returns:
            Lista de feature maps
        """
        # Se cache não estiver habilitado ou não houver chave, extrai normalmente
        if not hasattr(self, 'cache_enabled') or not self.cache_enabled or cache_key is None:
            return self.extract_features(image)
        
        # Converte tensor para hash usando sua soma
        if cache_key is None:
            cache_key = str(image.sum().item())
        
        # Verifica se já está no cache
        if cache_key in self.feature_cache:
            # Move para o topo da lista de uso recente
            if cache_key in self.feature_cache_keys:
                self.feature_cache_keys.remove(cache_key)
            self.feature_cache_keys.append(cache_key)
            return self.feature_cache[cache_key]
        
        # Extrai features normalmente
        features = self.extract_features(image)
        
        # Adiciona ao cache
        self.feature_cache[cache_key] = features
        self.feature_cache_keys.append(cache_key)
        
        # Se excedeu o tamanho máximo, remove o item menos recente
        if len(self.feature_cache_keys) > self.max_cache_items:
            oldest_key = self.feature_cache_keys.pop(0)
            del self.feature_cache[oldest_key]
        
        return features
        
    # Sobrescreve os métodos train() e eval() para evitar conflitos com YOLO
    def train(self, mode=True):
        """
        Define o modo de treinamento apenas para os módulos internos,
        sem chamar train() no detector YOLO.
        """
        self._training = mode
        
        # Apenas aplica no context_head e outros módulos normais do nn.Module
        self.context_head.train(mode)
        
        return self
    
    def eval(self):
        """
        Define o modo de avaliação apenas para os módulos internos,
        sem chamar eval() no detector YOLO.
        """
        return self.train(False)
        
    def extract_features(self, image):
        """Extrai features do backbone"""
        # Determinar o dispositivo diretamente do modelo para garantir consistência
        device = next(self.detector.model.parameters()).device
    
        # Garantir que a imagem está no mesmo dispositivo que o modelo
        image = image.to(device, non_blocking=True)
    
        self.detector.model.eval()  # Sempre use eval para extração de features
        with torch.no_grad():
            # Usa apenas o backbone
            backbone = self.detector.model.model[:6]
            features = backbone(image)
            
            # Garante formato consistente de saída
            if isinstance(features, (list, tuple)):
                return features[-2:] # Últimas duas camadas de feature maps
            else:
                return [features]
    
    def extract_roi_features(self, feature_maps, boxes, batch_indices=None):
        """Extrai features de regiões específicas usando ROI Align"""
        if boxes.numel() == 0:
            return torch.empty((0, sum([f.shape[1] for f in feature_maps])), device=boxes.device)
        
        # CORREÇÃO: Determina o dispositivo do context_head e garante consistência
        device = next(self.context_head.parameters()).device
        
        # CORREÇÃO: Garante que todas as entradas estejam no mesmo dispositivo
        boxes = boxes.to(device)
        feature_maps = [f.to(device) for f in feature_maps]
        
        if batch_indices is None:
            batch_indices = torch.zeros(boxes.size(0), dtype=torch.int, device=device)
        else:
            batch_indices = batch_indices.to(device)
        
        # Converte para formato esperado pelo roi_align e garante tipo consistente
        boxes = boxes.to(dtype=torch.float32)
        batch_indices = batch_indices.to(dtype=torch.float32)
        rois = torch.cat([batch_indices.unsqueeze(1), boxes], dim=1)
        
        # MODIFICAÇÃO AQUI: Implementação condicional dependendo do dispositivo
        use_cuda = torch.cuda.is_available() and device.type == 'cuda'
        
        # Se o modelo estiver em CUDA e detectamos que o roi_align precisa de CUDA
        if use_cuda:
            # Move tudo para CUDA
            rois = rois.cuda()
            feature_maps = [f.cuda() for f in feature_maps]
        else:
            # Para CPU, use uma implementação alternativa
            return self._cpu_extract_roi_features(feature_maps, boxes, batch_indices)
        
        roi_features = []
        for f in feature_maps:
            # Garante que o feature map também é float32
            f = f.to(dtype=torch.float32, device=device)
            spatial_scale = f.shape[2] / 640
            
            # Libera memória antes da operação roi_align
            torch.cuda.empty_cache()
            
            # Processa em chunks menores para economizar memória
            max_rois_per_batch = 100
            num_rois = rois.shape[0]
            chunk_results = []
            
            for i in range(0, num_rois, max_rois_per_batch):
                roi_chunk = rois[i:i+max_rois_per_batch]
                aligned = roi_align(f, roi_chunk, output_size=(7, 7), spatial_scale=spatial_scale)
                pooled = F.adaptive_avg_pool2d(aligned, (1, 1)).view(roi_chunk.size(0), -1)
                chunk_results.append(pooled)
                
                # Libera memória após cada chunk
                torch.cuda.empty_cache()
        
        if chunk_results:
            pooled_all = torch.cat(chunk_results, dim=0)
            roi_features.append(pooled_all)
        else:
            roi_features.append(torch.empty((0, f.shape[1]), device=device))
    
        # CORREÇÃO: Garante que embeddings está no mesmo dispositivo que context_head
        if roi_features:
            embeddings = torch.cat(roi_features, dim=1).to(device)
            embeddings = F.normalize(embeddings, dim=1)
            
            # Aplica context head para obter embedding final
            self.context_head = self.context_head.to(device)  # Garantia extra
            embeddings = self.context_head(embeddings)
            return F.normalize(embeddings, dim=1)
        else:
            # Fallback para tensor vazio com dimensão adequada
            empty_dim = self.context_head[-1].out_features  # Dimensão da saída do context_head
            return torch.zeros((0, empty_dim), device=device)
        
    # Adicione após o método extract_roi_features, aproximadamente linha 370

    def _memory_optimization(self, operation='check', threshold_mb=1000):
        """
        Sistema de gerenciamento de memória para evitar OOM errors
        
        Args:
            operation: 'check' para verificar, 'clean' para limpar
            threshold_mb: Limiar para limpeza em MB
            
        Returns:
            Uso atual de memória em MB se operation='check'
        """
        if not torch.cuda.is_available():
            return 0
            
        # Força coleta de lixo em PyTorch
        if operation in ['clean', 'both']:
            torch.cuda.empty_cache()
            
        # Checa uso atual
        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        if operation in ['check', 'both']:
            # Se ultrapassou o limiar, força limpeza
            if current_memory > threshold_mb:
                torch.cuda.empty_cache()
                # Atualiza uso após limpeza
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                
        return current_memory
    
    def recommend_placements(self, target_img, patch, top_k=5, conf_threshold=0.25, domain_stats=None):
        """
        Recomenda os melhores locais para colar o patch na imagem.
        """
        self._memory_optimization(operation='clean')
        
        device = next(self.detector.model.parameters()).device
        batch_size = target_img.shape[0]
        results = []
        
        # CORREÇÃO: Garante que os tensores estão no dispositivo correto
        target_img = target_img.to(device)
        patch = patch.to(device)
        
        for b in range(batch_size):
            # 1. Detectar objetos e filtrar por confiança
            # CORREÇÃO: Acessa as detecções corretamente da API atualizada
            results_obj = self.detector(target_img[b:b+1])
            
            # Verifica se há detecções e obtém os dados das boxes
            if hasattr(results_obj[0], 'boxes') and results_obj[0].boxes.data.shape[0] > 0:
                detections = results_obj[0].boxes.data  # [x1, y1, x2, y2, conf, cls]
                
                # Filtrar por confiança
                confidence_mask = detections[:, 4] > conf_threshold
                detections = detections[confidence_mask]
            else:
                # Se não há detecções ou formato inválido
                detections = torch.zeros((0, 6), device=device)

            # 2. Se não encontrou objetos, gerar grade de candidatos
            if len(detections) == 0:
                candidate_boxes = self._generate_grid_candidates(target_img[b].shape[-2:])
            else:
                # Usar boxes detectadas expandidas
                candidate_boxes = []
                for det in detections:
                    x1, y1, x2, y2 = det[:4]
                    # Expandir box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    exp_w, exp_h = w * 1.5, h * 1.5
                    
                    new_x1 = max(0, cx - exp_w/2)
                    new_y1 = max(0, cy - exp_h/2)
                    new_x2 = min(target_img.shape[-1], cx + exp_w/2)
                    new_y2 = min(target_img.shape[-2], cy + exp_h/2)
                    
                    candidate_boxes.append([new_x1, new_y1, new_x2, new_y2])
                    
                    # Também adiciona a box original como candidata
                    candidate_boxes.append([x1, y1, x2, y2])
                
                # Complementar com grid se temos poucas detecções
                if len(candidate_boxes) < 5:
                    # Adicionar grid básico
                    grid_boxes = self._generate_grid_candidates(target_img[b].shape[-2:], grid_size=4)
                    candidate_boxes.extend(grid_boxes)
                    
                    # Detectar regiões com contexto similar para melhor qualidade
                    similar_regions = self.detect_similar_context_regions(
                        target_img[b:b+1], patch[b:b+1], top_k=5
                    )
                    
                    # Adicionar regiões similares como candidatos prioritários
                    if similar_regions:
                        similar_boxes = torch.tensor(similar_regions, device=device)
                        # Converter candidate_boxes para tensor se ainda for lista
                        if isinstance(candidate_boxes, list):
                            candidate_boxes = torch.tensor(candidate_boxes, device=device)
                        # Inserir no início da lista para maior prioridade
                        candidate_boxes = torch.cat([similar_boxes, candidate_boxes])
        
            # Converter para tensor
            candidate_boxes = torch.tensor(candidate_boxes, device=device)
            
            # 3. Analisar informativeness de cada região candidata
            context_analysis = self.analyze_context_informativity(
                target_img[b:b+1], patch[b:b+1], candidate_boxes, top_k=min(len(candidate_boxes), 10)
            )
            
            # 4. Para cada região candidata promissora, avaliar qualidade
            final_recommendations = []
            
            for ctx in context_analysis:
                box = ctx["box"]
                # Avaliar qualidade da região para colagem
                quality_score, details = self.evaluate_region_quality(
                    target_img[b:b+1], box, reference_lesions=self.reference_contexts if hasattr(self, 'reference_contexts') else None
                )

                # Adicionar avaliação anatômica
                anatomical_score, anatomical_details = self.evaluate_anatomical_compatibility(
                    target_img[b:b+1], patch[b:b+1], box, domain_stats=domain_stats
                )

                # Adicionar detalhes anatômicos ao resultado
                details.update(anatomical_details)

                # Calcular score combinado (com pesos balanceados)
                combined_score = 0.5 * ctx["similarity"] + 0.2 * ctx["informativity"] + 0.15 * quality_score + 0.15 * anatomical_score
                
                if combined_score > 0.3:  # Limiar mínimo para recomendação
                    final_recommendations.append({
                        "box": box,
                        "similarity_score": float(ctx["similarity"]),
                        "informativity_score": float(ctx["informativity"]),
                        "quality_score": float(quality_score),
                        "combined_score": float(combined_score),
                        "details": details
                    })
            
            # Ordenar por score combinado
            final_recommendations.sort(key=lambda x: x["combined_score"], reverse=True)
            results.append(final_recommendations[:top_k])
        
        # Aplanar resultados se batch_size=1
        if batch_size == 1:
            return results[0]
        
        return results
    
    def visualize_recommendations(self, target_img, patch, recommendations=None, save_path=None, show_scores=True, show_simulated=False):
        """
        Visualiza as recomendações de colagem com detalhes informativos.
        
        Args:
            target_img: Tensor [B,C,H,W] ou [C,H,W]
            patch: Tensor do patch a ser colado
            recommendations: Lista de recomendações da função recommend_placements
            save_path: Caminho para salvar a visualização
            show_scores: Se deve mostrar os scores de cada componente
            show_simulated: Se deve mostrar simulações das colagens
        
        Returns:
            Figura matplotlib
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        
        if recommendations is None:
            recommendations = self.recommend_placements(target_img, patch)
            
        if not recommendations:
            print("Nenhuma recomendação encontrada")
            return None
            
        # Normalizar tensores para visualização
        if len(target_img.shape) == 4:
            target_img = target_img[0]
        if len(patch.shape) == 4:
            patch = patch[0]
        
        target_img_np = target_img.permute(1, 2, 0).cpu().numpy()
        target_img_np = np.clip(target_img_np, 0, 1)
        
        patch_np = patch.permute(1, 2, 0).cpu().numpy()
        patch_np = np.clip(patch_np, 0, 1)
        
        # Criar figura
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(target_img_np)
        
        # Mostrar patch como inset
        patch_ax = fig.add_axes([0.02, 0.02, 0.2, 0.2])
        patch_ax.imshow(patch_np)
        patch_ax.set_title("Patch")
        patch_ax.axis('off')
        
        # Cores para diferentes rankings
        colors = ['lime', 'cyan', 'yellow', 'magenta', 'white']
        
        # Desenhar recomendações
        for i, rec in enumerate(recommendations[:5]):  # Limitado a 5 recomendações
            x1, y1, x2, y2 = rec["box"]
            color = colors[i % len(colors)]
            
            # Retângulo
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Texto com scores
            if show_scores:
                score_text = f"#{i+1} Score: {rec['combined_score']:.2f}"
                if 'similarity_score' in rec:
                    score_text += f"\nSim: {rec['similarity_score']:.2f}"
                if 'informativity_score' in rec:
                    score_text += f" Inf: {rec['informativity_score']:.2f}"
                if 'quality_score' in rec:
                    score_text += f" Qual: {rec['quality_score']:.2f}"
                    
                ax.text(
                    x1, y1-5, score_text, color='white', fontsize=10, weight='bold',
                    bbox=dict(facecolor=color, alpha=0.7, pad=1)
                )
        
        ax.set_title("Recomendações de Colagem")
        ax.axis('off')
        
        # Mostrar simulações de colagem se solicitado
        if show_simulated and recommendations:
            # Ajustar o layout para incluir as simulações
            fig.set_size_inches(18, 12)
            
            # Criar subplots para simulações
            num_sims = min(3, len(recommendations))
            
            for i in range(num_sims):
                # Criar um novo subplot na parte inferior
                sim_ax = fig.add_subplot(2, num_sims, num_sims + i + 1)
                
                # Obter box e simular colagem
                box = recommendations[i]["box"]
                simulated_img = self.simulate_paste_result(
                    target_img, patch, box, blend_mode='poisson'
                )
                
                # Mostrar imagem simulada
                sim_ax.imshow(simulated_img)
                sim_ax.set_title(f"Simulação #{i+1}")
                sim_ax.axis('off')
                
                # Adicionar box indicando localização
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none'
                )
                sim_ax.add_patch(rect)
    
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
        
    def _cpu_extract_roi_features(self, feature_maps, boxes, batch_indices=None):
        """Implementação alternativa para CPU usando interpolação"""
        import torch.nn.functional as F
        
        if boxes.numel() == 0:
            return torch.empty((0, sum([f.shape[1] for f in feature_maps])), device=boxes.device)
        
        # Resultados para cada mapa de características
        roi_features = []
        
        for f in feature_maps:
            batch_size, channels, height, width = f.shape
            
            # Para cada box, extrai manualmente features
            features_list = []
            
            for i, box in enumerate(boxes):
                try:
                    # Obter coordenadas normalizadas da box e converter para Python float
                    if hasattr(box, 'tolist'):
                        x1, y1, x2, y2 = box.tolist()  # Tensor para lista Python
                    else:
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    
                    # Converter para índices de grid (inteiros Python explicitamente)
                    x1_idx = max(0, min(int((x1 / 640) * width), width - 1))
                    y1_idx = max(0, min(int((y1 / 640) * height), height - 1))
                    x2_idx = max(x1_idx + 1, min(int((x2 / 640) * width), width))
                    y2_idx = max(y1_idx + 1, min(int((y2 / 640) * height), height))
                    
                    # Obter batch_idx (inteiro Python explicitamente)
                    if batch_indices is None:
                        batch_idx = 0
                    else:
                        # Extrai valor como inteiro Python
                        idx_tensor = batch_indices[i]
                        if idx_tensor.is_cuda:
                            batch_idx = int(idx_tensor.cpu().item())
                        else:
                            batch_idx = int(idx_tensor.item())
                    
                    batch_idx = min(batch_idx, batch_size - 1)  # Limita para prevenir índices inválidos
                    
                    # Extrair região
                    roi = f[batch_idx:batch_idx+1, :, y1_idx:y2_idx, x1_idx:x2_idx]
                    
                    # Redimensionar para tamanho fixo (7x7)
                    if roi.numel() > 0:
                        roi_resized = F.interpolate(roi, size=(7, 7), mode='bilinear', align_corners=False)
                        pooled = F.adaptive_avg_pool2d(roi_resized, (1, 1)).view(-1)
                        features_list.append(pooled)
                    else:
                        # Fallback para caso de ROI vazio
                        pooled = torch.zeros(channels, device=boxes.device)
                        features_list.append(pooled)
                        
                except Exception as e:
                    print(f"Erro ao processar box {i}: {box}, erro: {e}")
                    # Fallback: usar zeros como características
                    pooled = torch.zeros(channels, device=boxes.device)
                    features_list.append(pooled)
                
            if features_list:
                pooled_all = torch.stack(features_list)
                roi_features.append(pooled_all)
            else:
                roi_features.append(torch.empty((0, channels), device=boxes.device))
        
        # Concatena features de diferentes níveis
        embeddings = torch.cat(roi_features, dim=1)
        embeddings = F.normalize(embeddings, dim=1)
        
        # Aplica context head para obter embedding final
        embeddings = self.context_head(embeddings)
        return F.normalize(embeddings, dim=1)
    
    def extract_context_features(self, image, box, expansion_ratio=2.0):
        """Extrai características do contexto expandindo a box"""
        # Calcula box expandida
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # Expande mantendo proporção e centro
        context_width = width * expansion_ratio
        context_height = height * expansion_ratio
        
        context_box = torch.tensor([
            max(0, cx - context_width/2),
            max(0, cy - context_height/2),
            min(640, cx + context_width/2),
            min(640, cy + context_height/2)
        ], device=box.device).unsqueeze(0)
        
        # Extrai features
        features = self.extract_features(image)
        context_embeddings = self.extract_roi_features(features, context_box)
        
        # Usa o codificador de contexto específico
        return self.context_encoder(context_embeddings)
    
    def extract_reference_contexts(self, dataset, num_samples=100):
        """
        Extrai e modela os contextos de referência onde as lesões reais aparecem
        """
        print("Extraindo contextos de referência...")
        contexts = []
        
        # Para estatísticas de regiões
        illumination_values = []
        texture_values = []
        
        # Coleta contextos de lesões reais do dataset
        for i in range(min(num_samples, len(dataset))):
            img, _, boxes = dataset[i]
            
            if isinstance(img, torch.Tensor):
                img = img.to(self.device)
            
            # Extrai features da imagem
            features = self.extract_features(img.unsqueeze(0) if img.dim() == 3 else img)
            
            # Para cada lesão na imagem
            for box in boxes:
                if box.size(0) > 5 and box[5] == 1:  # Se é uma lesão
                    # Converte para xyxy
                    box_xyxy = self._convert_yolo_to_xyxy(box[:5].unsqueeze(0)).squeeze(0)
                    
                    # Expande box para capturar contexto
                    context_box = self._expand_box(box_xyxy, ratio=2.0)
                    
                    # Extrai features do contexto
                    context_features = self.extract_roi_features(features, context_box.unsqueeze(0))
                    contexts.append(context_features)
                    
                    # Extrai estatísticas visuais da região
                    try:
                        x1, y1, x2, y2 = [int(coord.item()) for coord in context_box]
                        region = img[..., y1:y2, x1:x2]
                        
                        if region.numel() > 0:
                            # Características de iluminação
                            illumination_values.append(region.mean().item())
                            
                            # Características de textura (variância local)
                            texture_values.append(torch.var(region.float(), dim=[-2, -1]).mean().item())
                    except:
                        pass  # Ignora erros de extração de estatísticas
    
        if not contexts:
            print("Aviso: Nenhum contexto de referência encontrado")
            return None
        
        # Combina todos os contextos
        reference_contexts = torch.cat(contexts)
        
        # Calcula estatísticas do espaço de características
        mean_context = torch.mean(reference_contexts, dim=0)
        std_context = torch.std(reference_contexts, dim=0)
        
        # Estatísticas de regiões
        region_stats = {
            'mean_illumination': float(np.mean(illumination_values)) if illumination_values else 0.5,
            'std_illumination': float(np.std(illumination_values) + 1e-5) if illumination_values else 0.2,
            'mean_texture': float(np.mean(texture_values)) if texture_values else 0.1,
            'std_texture': float(np.std(texture_values) + 1e-5) if texture_values else 0.05
        }
        
        print(f"Estatísticas de referência extraídas: {region_stats}")
        
        return {
            'mean': mean_context,
            'std': std_context,
            'samples': reference_contexts,
            'region_stats': region_stats
        }
    
    def _convert_yolo_to_xyxy(self, box, img_size=640):
        """Converte box de formato YOLO para xyxy"""
        if box.size(0) == 0:
            return box
            
        # Extrair componentes: center_x, center_y, width, height
        center_x = box[:, 1] * img_size
        center_y = box[:, 2] * img_size
        width = box[:, 3] * img_size
        height = box[:, 4] * img_size
        
        # Calcular x1, y1, x2, y2
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _expand_box(self, box, ratio=2.0):
        """Expande uma bounding box mantendo seu centro"""
        # Calcula o centro e dimensões da box
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # Novas dimensões expandidas
        new_w = w * ratio
        new_h = h * ratio
        
        # Novas coordenadas mantendo o centro
        new_x1 = torch.clamp(cx - new_w / 2, min=0)
        new_y1 = torch.clamp(cy - new_h / 2, min=0) 
        new_x2 = torch.clamp(cx + new_w / 2, max=640)
        new_y2 = torch.clamp(cy + new_h / 2, max=640)
        
        # Retorna box expandida
        return torch.stack([new_x1, new_y1, new_x2, new_y2])

    def analyze_context_informativity(self, image, patch, boxes, top_k=3):
        """
        Analisa o quanto cada contexto candidato é informativo para a decisão de colagem.
        
        Args:
            image: Imagem [1,C,H,W]
            patch: Patch a ser colado [1,C,H,W]
            boxes: Lista de boxes candidatas [N,4] em formato [x1,y1,x2,y2]
            top_k: Número de melhores contextos a retornar
            
        Returns:
            Lista de dicts com scores e medidas de informatividade para cada box
        """
        if len(boxes) == 0:
            return []
        
        # Extrair features da imagem e patch
        with torch.no_grad():
            image_features = self.extract_features(image)
            patch_features = self.extract_patch_features(patch)
            
            # Converter boxes para tensor se necessário
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes, device=image.device)
            
            # Extrair features das regiões
            region_features = self.extract_roi_features(image_features, boxes)
            
            # Calcular similaridade com o patch
            similarities = F.cosine_similarity(region_features, patch_features)
            
            # Para cada região, calcular o impacto de sua presença na decisão
            informativity_scores = []
            for i in range(len(boxes)):
                # Extrair features de todas as regiões exceto esta
                other_indices = [j for j in range(len(boxes)) if j != i]
                if not other_indices:  # Se só tem um box
                    informativity_scores.append(1.0)
                    continue
                    
                other_features = region_features[other_indices]
                other_similarities = F.cosine_similarity(other_features, patch_features)
                
                # Quanto a inclusão desta região muda a classificação geral?
                with_region = torch.mean(similarities).item()
                without_region = torch.mean(other_similarities).item()
                
                # Score de informatividade: quanto maior o impacto, mais informativa
                informativity = abs(with_region - without_region) * 10.0  # Escala para 0-1
                informativity_scores.append(min(1.0, informativity))
            
            # Combinar com similaridade para score final
            combined_scores = similarities * 0.7 + torch.tensor(informativity_scores, device=similarities.device) * 0.3
            
            # Ordenar por score combinado
            sorted_indices = torch.argsort(combined_scores, descending=True)
            
            # Retornar top_k melhores
            results = []
            for i in range(min(top_k, len(sorted_indices))):
                idx = sorted_indices[i].item()
                x1, y1, x2, y2 = boxes[idx].tolist()
                results.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "similarity": float(similarities[idx]),
                    "informativity": float(informativity_scores[idx]),
                    "combined_score": float(combined_scores[idx])
                })
            
            return results
        
    def _generate_grid_candidates(self, image_size, grid_size=6, min_box_size=64, max_box_size=256):
        """
        Gera uma grade de boxes candidatas para avaliação quando nenhuma detecção é encontrada.
        
        Args:
            image_size: Tupla (H,W) do tamanho da imagem
            grid_size: Número de divisões da grade
            min_box_size: Tamanho mínimo da box
            max_box_size: Tamanho máximo da box
            
        Returns:
            Lista de boxes [x1,y1,x2,y2]
        """
        h, w = image_size
        boxes = []
        
        # Tamanhos variados de box
        sizes = np.linspace(min_box_size, max_box_size, 3)
        
        # Gerar centros em grade
        for i in range(grid_size):
            for j in range(grid_size):
                cx = w * (i + 0.5) / grid_size
                cy = h * (j + 0.5) / grid_size
                
                # Para cada centro, vários tamanhos
                for size in sizes:
                    half_size = size / 2
                    x1 = max(0, cx - half_size)
                    y1 = max(0, cy - half_size)
                    x2 = min(w, cx + half_size)
                    y2 = min(h, cy + half_size)
                    
                    # Adicionar apenas se a box tiver tamanho mínimo
                    if x2 - x1 >= min_box_size and y2 - y1 >= min_box_size:
                        boxes.append([x1, y1, x2, y2])
        
        return boxes
    
    # Substitua o método evaluate_anatomical_compatibility por esta versão aprimorada

    def evaluate_anatomical_compatibility(self, image, patch, box, domain_stats=None):
        """
        Versão aprimorada para avaliar a compatibilidade anatômica entre 
        um patch de lesão e uma região candidata.
        
        Args:
            image: Tensor da imagem alvo [C,H,W] ou [1,C,H,W]
            patch: Tensor do patch de lesão [C,H,W] ou [1,C,H,W]
            box: Lista [x1,y1,x2,y2] com coordenadas da região candidata
            domain_stats: Estatísticas do domínio (opcional)
            
        Returns:
            Score de compatibilidade (0-1) e detalhes de avaliação
        """
        # Garantir formato consistente
        if len(image.shape) == 4:
            image = image[0]
        if len(patch.shape) == 4:
            patch = patch[0]
                
        device = image.device
        
        # 1. Extrai características do patch
        patch_stats = {}
        
        # Características de cor
        patch_color_mean = patch.mean(dim=[1, 2])
        patch_color_std = torch.std(patch, dim=[1, 2])
        
        # Característica de iluminação
        patch_luminance = patch.mean(dim=0)  # [H, W]
        patch_lum_mean = patch_luminance.mean()
        patch_lum_std = patch_luminance.std()
        
        # Características de textura (aproximação por gradiente)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).float()
        
        patch_gray = patch.mean(dim=0, keepdim=True)  # [1, H, W]
        
        # Pad para convolução
        patch_padded = F.pad(patch_gray, (1, 1, 1, 1), mode='reflect')
        
        # Aplicar filtros Sobel manualmente
        patch_grad_x = F.conv2d(patch_padded.unsqueeze(0), 
                            sobel_x.view(1, 1, 3, 3), padding=0).squeeze()
        patch_grad_y = F.conv2d(patch_padded.unsqueeze(0), 
                            sobel_y.view(1, 1, 3, 3), padding=0).squeeze()
        
        patch_grad_mag = torch.sqrt(patch_grad_x**2 + patch_grad_y**2)
        patch_texture_energy = patch_grad_mag.mean()
        
        # Tamanho do patch
        patch_size = torch.tensor([patch.shape[1], patch.shape[2]]).float()
        
        # 2. Extrai região da imagem alvo
        x1, y1, x2, y2 = [int(coord) for coord in box]
        target_region = image[:, y1:y2, x1:x2]
        
        # Redimensionar para tamanho comparável
        if min(target_region.shape[1:]) > 20:  # Região grande o suficiente para resize
            target_region = F.interpolate(
                target_region.unsqueeze(0), 
                size=patch.shape[1:], 
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # 3. Calcular métricas básicas de compatibilidade
        
        # 3.1 Cor
        region_color_mean = target_region.mean(dim=[1, 2])
        color_diff = F.mse_loss(patch_color_mean, region_color_mean)
        color_score = torch.exp(-5.0 * color_diff).item()  # 1.0 = cores idênticas
        
        # 3.2 Iluminação
        region_luminance = target_region.mean(dim=0)
        region_lum_mean = region_luminance.mean()
        lum_diff = abs(patch_lum_mean - region_lum_mean)
        lum_score = torch.exp(-10.0 * lum_diff).item()  # 1.0 = iluminação idêntica
        
        # 3.3 Textura
        region_gray = target_region.mean(dim=0, keepdim=True)
        region_padded = F.pad(region_gray, (1, 1, 1, 1), mode='reflect')
        
        region_grad_x = F.conv2d(region_padded.unsqueeze(0), 
                            sobel_x.view(1, 1, 3, 3), padding=0).squeeze()
        region_grad_y = F.conv2d(region_padded.unsqueeze(0), 
                            sobel_y.view(1, 1, 3, 3), padding=0).squeeze()
        
        region_grad_mag = torch.sqrt(region_grad_x**2 + region_grad_y**2)
        region_texture_energy = region_grad_mag.mean()
        
        texture_diff = abs(patch_texture_energy - region_texture_energy)
        texture_score = torch.exp(-5.0 * texture_diff).item()
        
        # 4. Compatibilidade com estatísticas do domínio (versão aprimorada)
        domain_score = 1.0
        anatomical_constraints_score = 1.0
        pattern_score = 1.0
        adjacency_score = 1.0
        
        if domain_stats:
            # 4.1 Compatibilidade com distribuição espacial
            if 'density' in domain_stats:
                density_grid = np.array(domain_stats['density'])
                grid_size = density_grid.shape[0]
                
                # Normaliza coordenadas para o grid
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                norm_x = int(cx / image.shape[2] * grid_size)
                norm_y = int(cy / image.shape[1] * grid_size)
                
                # Limita aos índices válidos
                norm_x = min(max(0, norm_x), grid_size - 1)
                norm_y = min(max(0, norm_y), grid_size - 1)
                
                # Obtém score da distribuição de densidade
                density_score = density_grid[norm_y, norm_x]
                
                # Penaliza regiões fora da distribuição comum
                domain_score = max(0.2, min(1.0, density_score * 5))
            
            # 4.2 Compatibilidade com padrões de tamanho
            if 'mean_size' in domain_stats and 'std_size' in domain_stats:
                # CONVERSÃO PARA TENSORES NO MESMO DEVICE (corrige TypeError do torch.exp)
                ms = torch.as_tensor(domain_stats['mean_size'], device=device, dtype=torch.float32)
                ss = torch.as_tensor(domain_stats['std_size'],  device=device, dtype=torch.float32) + 1e-5

                bw = torch.tensor((x2 - x1) / image.shape[2], device=device, dtype=torch.float32)
                bh = torch.tensor((y2 - y1) / image.shape[1], device=device, dtype=torch.float32)

                size_diff_w = torch.abs(bw - ms[0]) / ss[0]
                size_diff_h = torch.abs(bh - ms[1]) / ss[1]
                size_diff = (size_diff_w + size_diff_h) / 2.0

                size_score = torch.exp(-size_diff).item()
                anatomical_constraints_score *= float(size_score)
            
            # 4.3 Compatibilidade com padrões de agrupamento
            if 'patterns' in domain_stats and 'positions' in domain_stats:
                # Coordenadas normalizadas do centro da box
                cx_norm = (x1 + x2) / (2 * image.shape[2])
                cy_norm = (y1 + y2) / (2 * image.shape[1])
                
                # Calcula distância às lesões conhecidas
                positions = np.array(domain_stats['positions'])
                if len(positions) > 0:
                    # Distâncias às lesões existentes
                    distances = np.sqrt((positions[:, 0] - cx_norm)**2 + 
                                        (positions[:, 1] - cy_norm)**2)
                    
                    # Encontra a distância mínima
                    min_distance = np.min(distances)
                    
                    # Se houver padrões de cluster identificados
                    if 'cluster_ratio' in domain_stats['patterns']:
                        cluster_ratio = domain_stats['patterns']['cluster_ratio']
                        
                        # Alta taxa de cluster favorece posições próximas a outras lesões
                        if cluster_ratio > 0.6:  # Muitos agrupamentos
                            # Próximo é bom, longe é ruim
                            pattern_score = np.exp(-5.0 * min_distance)
                        elif cluster_ratio < 0.3:  # Poucas lesões agrupadas
                            # Longe é bom, próximo é ruim
                            pattern_score = 1.0 - np.exp(-10.0 * min_distance)
                        # Entre 0.3 e 0.6, não favorece nem penaliza
                
            # 4.4 Compatibilidade com adjacência média
            if 'adjacency' in domain_stats and len(domain_stats['adjacency']) > 0:
                # Média de adjacência (distância entre lesões)
                mean_adjacency = np.mean(domain_stats['adjacency'])
                
                # Coordenadas normalizadas do centro da box
                cx_norm = (x1 + x2) / (2 * image.shape[2])
                cy_norm = (y1 + y2) / (2 * image.shape[1])
                
                # Calcula distância às lesões conhecidas
                positions = np.array(domain_stats['positions'])
                if len(positions) > 0:
                    # Distâncias às lesões existentes
                    distances = np.sqrt((positions[:, 0] - cx_norm)**2 + 
                                        (positions[:, 1] - cy_norm)**2)
                    
                    # Encontra a distância mínima
                    min_distance = np.min(distances)
                    
                    # Score baseado em quão próximo esta distância está da média de adjacência
                    adjacency_diff = abs(min_distance - mean_adjacency)
                    adjacency_score = np.exp(-5.0 * adjacency_diff)
        
        # 5. Calcular score final combinado - pesos ajustados
        visual_score = color_score * 0.3 + lum_score * 0.3 + texture_score * 0.2
        anatomical_score = (domain_score * 0.4 + 
                        anatomical_constraints_score * 0.3 + 
                        pattern_score * 0.15 + 
                        adjacency_score * 0.15)
        
        # Combina scores visuais e anatômicos
        combined_score = visual_score * 0.6 + anatomical_score * 0.4
        
        # Detalhes para debug e análise
        details = {
            "color_score": float(color_score),
            "luminance_score": float(lum_score),
            "texture_score": float(texture_score),
            "domain_score": float(domain_score),
            "anatomical_constraints": float(anatomical_constraints_score),
            "pattern_score": float(pattern_score),
            "adjacency_score": float(adjacency_score),
            "visual_score": float(visual_score),
            "anatomical_score": float(anatomical_score)
        }
        
        return float(combined_score), details
    
    def simulate_paste_result(self, image, patch, box, blend_mode='poisson', return_tensor=False):
        """
        Simula o resultado da colagem de um patch em uma imagem-alvo.
        
        Args:
            image: Tensor da imagem alvo [C,H,W] ou [1,C,H,W]
            patch: Tensor do patch a colar [C,H,W] ou [1,C,H,W]
            box: Lista [x1,y1,x2,y2] com coordenadas da região alvo
            blend_mode: Método de blending ('alpha', 'poisson', 'direct')
            return_tensor: Se True, retorna tensor; se False, retorna numpy array
            
        Returns:
            Imagem com patch colado
        """
        # Garantir formato correto
        if len(image.shape) == 4:
            image = image[0]
        if len(patch.shape) == 4:
            patch = patch[0]
    
        # Converter tensores para numpy para processamento
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()
    
        # Converter box para inteiros
        x1, y1, x2, y2 = [int(coord) for coord in box]
        target_h, target_w = y2 - y1, x2 - x1
    
        # Redimensionar patch para o tamanho da região-alvo
        patch_resized = cv2.resize(patch_np, (target_w, target_h))
    
        # Criar cópia da imagem para manipulação
        result = image_np.copy()
    
        if blend_mode == 'alpha':
            # Alpha blending simples
            alpha = 0.7  # Opacidade do patch
            result[y1:y2, x1:x2] = alpha * patch_resized + (1-alpha) * result[y1:y2, x1:x2]
    
        elif blend_mode == 'poisson':
            try:
                # Importar apenas quando necessário
                import cv2
                
                # Converte para valores de 0-255
                target = (image_np * 255).astype(np.uint8)
                source = (patch_resized * 255).astype(np.uint8)
                
                # Criar máscara (região central tem mais peso)
                mask = np.ones(source.shape[:2], dtype=np.uint8) * 255
                
                # Aplicar feather na máscara (borda suave)
                kernel_size = max(3, min(target_h, target_w) // 10)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.GaussianBlur(mask, (kernel_size*2+1, kernel_size*2+1), 0)
                
                # Aplicar Poisson blending
                center = (x1 + target_w//2, y1 + target_h//2)
                result = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
                result = result.astype(np.float32) / 255.0
                
            except Exception as e:
                print(f"Erro no Poisson blending: {e}")
                # Fallback para alpha blending
                alpha = 0.7
                result[y1:y2, x1:x2] = alpha * patch_resized + (1-alpha) * result[y1:y2, x1:x2]
    
        else:  # modo 'direct'
            # Simplesmente sobrepõe o patch
            result[y1:y2, x1:x2] = patch_resized
    
        # Retornar no formato apropriado
        if return_tensor:
            return torch.tensor(result).permute(2, 0, 1)
        return result
    
    def detect_similar_context_regions(self, image, reference_patch, top_k=5):
        """
        Detecta automaticamente regiões que têm contexto similar ao do patch de referência.
        
        Args:
            image: Tensor da imagem [1,C,H,W]
            reference_patch: Tensor do patch de lesão [1,C,H,W]
            top_k: Número de regiões similares a retornar
            
        Returns:
            Lista de boxes [x1,y1,x2,y2] com regiões de contexto similar
        """
        device = image.device
        
        # 1. Extrair características globais do patch
        with torch.no_grad():
            patch_features = self.extract_patch_features(reference_patch)
    
        # 2. Dividir a imagem em grade e analisar cada região
        grid_size = 8
        h, w = image.shape[2:]
        cell_h, cell_w = h // grid_size, w // grid_size
        
        # Tamanhos de janela variados para análise multi-escala
        window_sizes = [
            (cell_w, cell_h),
            (int(cell_w*1.5), int(cell_h*1.5)),
            (int(cell_w*2), int(cell_h*2))
        ]
        
        candidate_boxes = []
        candidate_scores = []
        
        # Evitar sobreposição excessiva
        min_iou_threshold = 0.5
        
        for window_w, window_h in window_sizes:
            for i in range(0, grid_size):
                for j in range(0, grid_size):
                    # Centro da célula
                    cx = (i + 0.5) * cell_w
                    cy = (j + 0.5) * cell_h
                    
                    # Coordenadas da janela
                    x1 = max(0, int(cx - window_w/2))
                    y1 = max(0, int(cy - window_h/2))
                    x2 = min(w, int(cx + window_w/2))
                    y2 = min(h, int(cy + window_h/2))
                    
                    # Se a janela é muito pequena, pular
                    if x2 - x1 < 32 or y2 - y1 < 32:
                        continue
                        
                    # Extrai região e características
                    box = torch.tensor([[x1, y1, x2, y2]], device=device)
                    
                    # Extrai features da imagem
                    image_features = self.extract_features(image)
                    region_features = self.extract_roi_features(image_features, box)
                    
                    # Calcula similaridade com o patch
                    similarity = F.cosine_similarity(region_features, patch_features).item()
                    
                    # Verificar IoU com boxes existentes para evitar redundância
                    add_box = True
                    for existing_box, existing_score in zip(candidate_boxes, candidate_scores):
                        iou = calculate_iou(
                            [x1, y1, x2, y2], 
                            [existing_box[0], existing_box[1], existing_box[2], existing_box[3]]
                        )
                        if iou > min_iou_threshold:
                            # Se já existe box similar e atual tem score maior, substitui
                            if similarity > existing_score:
                                existing_box[0], existing_box[1] = x1, y1
                                existing_box[2], existing_box[3] = x2, y2
                                existing_score = similarity
                            add_box = False
                            break
                    
                    if add_box:
                        candidate_boxes.append([x1, y1, x2, y2])
                        candidate_scores.append(similarity)
        
        # Ordenar e retornar as top_k melhores regiões
        if not candidate_boxes:
            return []
        
        sorted_indices = np.argsort(candidate_scores)[::-1][:top_k]
        best_boxes = [candidate_boxes[i] for i in sorted_indices]
        
        return best_boxes

    def extract_patch_features(self, patch):
        """
        Extrai características de um patch para comparação com regiões candidatas.
        
        Args:
            patch: Tensor do patch [1,C,H,W] ou [C,H,W]
        
        Returns:
            Tensor de características [1,D]
        """
        # Garante formato correto
        if len(patch.shape) == 3:
            patch = patch.unsqueeze(0)
    
        # CORREÇÃO ROBUSTA: Determina o dispositivo do modelo e move todos os dados para ele
        device = next(self.object_encoder.parameters()).device  # Usa o dispositivo do encoder
        patch = patch.to(device)  # Garante que o patch está no dispositivo correto
    
        # Extrai features usando o backbone do detector
        with torch.no_grad():
            features = self.extract_features(patch)
        
            # Aplica pooling global para reduzir para um vetor
            pooled_features = []
            for f in features:
                # Garante que o feature map está no dispositivo correto
                f = f.to(device)
                # Aplica average pooling global
                p = F.adaptive_avg_pool2d(f, (1, 1)).view(f.size(0), -1)
                pooled_features.append(p)
    
        # Concatena features de diferentes níveis
        if pooled_features:
            combined = torch.cat(pooled_features, dim=1).to(device)
        else:
            # Fallback para um tensor vazio com dimensão adequada
            combined = torch.zeros((1, 1024), device=device)
    
        # Aplica embedding de objeto (garantindo que está no dispositivo correto)
        combined = combined.to(device)  # Garantia extra
        self.object_encoder = self.object_encoder.to(device)  # Garante que o módulo está no dispositivo correto
        
        # Aplica embedding de objeto
        embedded = self.object_encoder(combined)
        
        # Normaliza
        return F.normalize(embedded, dim=1)

    def evaluate_region_quality(self, image, box, reference_lesions=None):
        """
        Avalia a qualidade de uma região para colagem de lesões, retornando um score de 0-1.
        
        Args:
            image: Imagem PIL ou tensor [C,H,W]
            box: Coordenadas [x1,y1,x2,y2] ou [cx,cy,w,h,is_yolo]
            reference_lesions: Lista opcional de features de lesões para comparação
            
        Returns:
            Score de qualidade (0-1) e dict com scores individuais
        """
        # Converter para tensor se for PIL
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
    
        if isinstance(image, Image.Image):
            # Converte PIL para tensor
            transform = transforms.ToTensor()
            image = transform(image).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
    
        # Padronizar formato da box para [x1,y1,x2,y2]
        if len(box) == 5 and box[4]:  # é YOLO
            cx, cy, w, h = box[:4]
            img_size = image.shape[-1]
            x1 = int((cx - w/2) * img_size)
            y1 = int((cy - h/2) * img_size)
            x2 = int((cx + w/2) * img_size)
            y2 = int((cy + h/2) * img_size)
        else:
            x1, y1, x2, y2 = [int(c) for c in box[:4]]
    
        # Garantir que a box está dentro dos limites da imagem
        img_h, img_w = image.shape[-2], image.shape[-1]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, {"error": "Box inválida"}
        
        # Extrair região
        device = image.device
        region = image[:, :, y1:y2, x1:x2]
        
        # 1. Critério de tamanho adequado
        min_size, max_size = 16, img_w/2
        size_score = 1.0
        region_size = max(x2-x1, y2-y1)
        
        if region_size < min_size:
            size_score = 0.1
        elif region_size > max_size:
            size_score = 0.5
    
        # 2. Critério de luminância
        with torch.no_grad():
            luminance = torch.mean(region, dim=1)  # Média sobre canais
            mean_luminance = torch.mean(luminance)
            
            # Muito escuro ou muito claro é ruim
            if mean_luminance < 0.1:  # Muito escuro
                lum_score = 0.2
            elif mean_luminance > 0.9:  # Muito claro
                lum_score = 0.5
            else:
                # Escala linear entre 0.1-0.9
                lum_score = 1.0 - 0.6 * abs((mean_luminance - 0.5) / 0.4)
    
        # 3. Critério de similaridade com lesões de referência
        if reference_lesions is not None and hasattr(reference_lesions, 'samples'):
            # Extrair características desta região
            with torch.no_grad():
                region_features = self.extract_roi_features(
                    self.extract_features(image), 
                    torch.tensor([[x1, y1, x2, y2]], device=device)
                )
                
                # Calcular similaridade com referências
                similarities = F.cosine_similarity(
                    region_features.unsqueeze(1), 
                    reference_lesions['samples'].to(device).unsqueeze(0), 
                    dim=2
                )
                max_similarity = torch.max(similarities).item()
                
                # Escala para score (0.5 para neutro, até 1.0 para muito similar)
                similarity_score = 0.5 + max(0, min(0.5, max_similarity - 0.3))
        else:
            similarity_score = 0.5  # Neutro se não há referências
        
        # Combina scores com pesos
        final_score = 0.3 * size_score + 0.3 * lum_score + 0.4 * similarity_score
        
        # Detalhes para debug
        details = {
            "size_score": float(size_score),
            "luminance_score": float(lum_score),
            "similarity_score": float(similarity_score) if reference_lesions is not None else None,
            "region_size": float(region_size),
            "mean_luminance": float(mean_luminance)
        }
        
        return float(final_score), details

def calculate_iou(box1, box2):
    """
    Calcula IoU (Intersection over Union) entre duas boxes
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Adicione no final do arquivo, antes da última linha

class RecommendationEvaluator:
    """
    Classe para auto-avaliação das recomendações de colagem
    """
    def __init__(self, recommender):
        """
        Inicializa o avaliador
        
        Args:
            recommender: Instância do PasteRecommender
        """
        self.recommender = recommender
        self.evaluation_history = []
        self.feedback_data = []
        
    def evaluate_recommendation(self, image, patch, recommendation, ground_truth=None):
        """
        Avalia a qualidade de uma recomendação
        
        Args:
            image: Imagem alvo
            patch: Patch a ser colado
            recommendation: Recomendação gerada pelo sistema
            ground_truth: Box real (opcional)
            
        Returns:
            Dicionário com métricas de avaliação
        """
        device = next(self.recommender.parameters()).device
        
        # Métricas a avaliar
        metrics = {}
        
        # 1. Avaliação básica de qualidade
        if 'combined_score' in recommendation:
            metrics['recommender_score'] = recommendation['combined_score']
        
        # 2. Simulação de colagem e avaliação de naturalidade
        if hasattr(self.recommender, 'simulate_paste_result'):
            box = recommendation['box']
            try:
                # Simula colagem
                simulated = self.recommender.simulate_paste_result(
                    image, patch, box, blend_mode='poisson', return_tensor=True
                ).to(device)
                
                # Avalia naturalidade usando as features do detector
                with torch.no_grad():
                    orig_features = self.recommender.extract_features(image)
                    sim_features = self.recommender.extract_features(simulated)
                    
                    # Calcula diferença nas features
                    feature_diffs = []
                    for of, sf in zip(orig_features, sim_features):
                        # Extrai features apenas da região modificada
                        x1, y1, x2, y2 = [int(b) for b in box]
                        
                        # Calcula coordenadas proporcionais no feature map
                        h_ratio = of.shape[-2] / image.shape[-2]
                        w_ratio = of.shape[-1] / image.shape[-1]
                        
                        f_x1 = max(0, int(x1 * w_ratio))
                        f_y1 = max(0, int(y1 * h_ratio))
                        f_x2 = min(of.shape[-1], int(x2 * w_ratio))
                        f_y2 = min(of.shape[-2], int(y2 * h_ratio))
                        
                        # Extrai região de interesse nas features
                        if f_x2 > f_x1 and f_y2 > f_y1:
                            of_region = of[:, :, f_y1:f_y2, f_x1:f_x2]
                            sf_region = sf[:, :, f_y1:f_y2, f_x1:f_x2]
                            
                            # Calcula diferença
                            if of_region.numel() > 0 and sf_region.numel() > 0:
                                diff = F.mse_loss(of_region, sf_region).item()
                                feature_diffs.append(diff)
                
                # Score de naturalidade (menor diferença = mais natural)
                if feature_diffs:
                    naturality_score = np.exp(-5.0 * np.mean(feature_diffs))
                    metrics['visual_naturality'] = float(naturality_score)
            except Exception as e:
                print(f"Erro na avaliação de naturalidade: {e}")
                metrics['visual_naturality'] = 0.5  # valor neutro
        
        # 3. Comparação com ground truth (se disponível)
        if ground_truth is not None:
            # Calcula IoU
            iou = calculate_iou(box, ground_truth)
            metrics['ground_truth_iou'] = float(iou)
            
            # Avaliação relativa à ground truth
            metrics['location_accuracy'] = 1.0 if iou > 0.5 else (iou * 2)
        
        # Adiciona ao histórico
        self.evaluation_history.append(metrics)
        
        return metrics
        
    def collect_feedback(self, image, patch, recommendation, feedback_score, feedback_type):
        """
        Coleta feedback (manual ou automático) para melhorar o sistema
        
        Args:
            image: Imagem alvo
            patch: Patch colado
            recommendation: Recomendação original
            feedback_score: Score de qualidade (0-1)
            feedback_type: Tipo de feedback (expert, automated, etc.)
        """
        import datetime 
        
        self.feedback_data.append({
            'recommendation': recommendation,
            'feedback_score': feedback_score,
            'feedback_type': feedback_type,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    def get_performance_metrics(self):
        """
        Calcula métricas agregadas de desempenho
        
        Returns:
            Dicionário com métricas de desempenho
        """
        if not self.evaluation_history:
            return {}
        
        metrics = {}
        
        # Extrai métricas de todas as avaliações
        for key in self.evaluation_history[0].keys():
            values = [e[key] for e in self.evaluation_history if key in e]
            if values:
                metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Adiciona métricas de feedback se disponíveis
        if self.feedback_data:
            feedback_scores = [f['feedback_score'] for f in self.feedback_data]
            metrics['user_feedback'] = {
                'mean': np.mean(feedback_scores),
                'std': np.std(feedback_scores),
                'count': len(feedback_scores)
            }
        
        return metrics
    
class SimplePasteRecommender:
    """
    Versão simplificada do recomendador que usa um detector YOLO + classificador binário.
    """
    def __init__(self, detector_path, embedding_size=256):
        # Detector para encontrar lesões
        self.detector = YOLO(detector_path)
        self.device = next(self.detector.model.parameters()).device
        
        # Classificador para avaliar compatibilidade região-patch
        self.classifier = SimpleContextClassifier(detector_path, embedding_size=embedding_size)
        self.classifier.to(self.device)
    
    def recommend_placements(self, target_img, patch, top_k=5):
        """
        Recomenda locais para colagem baseado na compatibilidade de contexto
        
        Args:
            target_img: Imagem alvo [B,3,H,W]
            patch: Patch de lesão [B,3,H,W]
            top_k: Número de recomendações a retornar
            
        Returns:
            Lista de recomendações com scores e boxes
        """
        device = self.device
        batch_size = target_img.shape[0]
        results = []
        
        # Garantir que tensores estão no device correto
        target_img = target_img.to(device)
        patch = patch.to(device)
        
        for b in range(batch_size):
            # 1. Gerar grade de candidatos uniforme
            candidate_regions = []
            candidate_boxes = []
            
            # Usar detector para encontrar regiões de interesse (se houver)
            detector_results = self.detector(target_img[b:b+1])
            if len(detector_results[0].boxes) > 0:
                # Expandir boxes detectadas como candidatos adicionais
                for det in detector_results[0].boxes.xyxy:
                    x1, y1, x2, y2 = det.tolist()
                    
                    # Box expandido
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    w, h = x2-x1, y2-y1
                    
                    for scale in [1.0, 1.5, 2.0]:
                        exp_w, exp_h = w*scale, h*scale
                        exp_x1 = max(0, cx - exp_w/2)
                        exp_y1 = max(0, cy - exp_h/2)
                        exp_x2 = min(target_img.shape[3], cx + exp_w/2)
                        exp_y2 = min(target_img.shape[2], cy + exp_h/2)
                        
                        # Extrair região
                        region = target_img[b:b+1, :, int(exp_y1):int(exp_y2), int(exp_x1):int(exp_x2)]
                        
                        # Verificar se a região é válida
                        if region.shape[2] < 16 or region.shape[3] < 16:
                            continue
                            
                        # Redimensionar para tamanho adequado
                        region = F.interpolate(region, size=(128, 128), mode='bilinear')
                        
                        candidate_regions.append(region)
                        candidate_boxes.append([exp_x1, exp_y1, exp_x2, exp_y2])
            
            # Adicionar grade uniforme de candidatos
            grid_boxes = self._generate_grid_candidates(target_img[b].shape[-2:], grid_size=6)
            
            for box in grid_boxes:
                x1, y1, x2, y2 = box
                
                # Extrair região
                region = target_img[b:b+1, :, int(y1):int(y2), int(x1):int(x2)]
                
                # Verificar se a região é válida
                if region.shape[2] < 16 or region.shape[3] < 16:
                    continue
                    
                # Redimensionar para tamanho adequado
                region = F.interpolate(region, size=(128, 128), mode='bilinear')
                
                candidate_regions.append(region)
                candidate_boxes.append([x1, y1, x2, y2])
            
            # 2. Para cada candidato, classificar compatibilidade
            scores = []
            for i, (region, box) in enumerate(zip(candidate_regions, candidate_boxes)):
                # Classificar compatibilidade (forward do modelo)
                with torch.no_grad():
                    score = self.classifier(region, patch[b:b+1]).item()
                
                # Outros atributos básicos da região
                region_lum = torch.mean(region).item()
                region_size = (box[2] - box[0]) * (box[3] - box[1]) / (target_img.shape[2] * target_img.shape[3])
                
                # Ajuste de score para evitar regiões muito escuras ou muito claras
                if region_lum < 0.05 or region_lum > 0.95:
                    score *= 0.5
                
                scores.append({
                    "box": box,
                    "compatibility_score": score,
                    "quality_score": min(1.0, 2.0 * region_size) if region_size > 0 else 0,
                    "combined_score": score,  # Simplificado para um único score
                    "rank": i
                })
            
            # Ordenar por score
            scores.sort(key=lambda x: x["compatibility_score"], reverse=True)
            results.append(scores[:top_k])
        
        # Aplanar resultados se batch_size=1
        if batch_size == 1:
            return results[0]
        
        return results
    
    def _generate_grid_candidates(self, image_size, grid_size=6):
        """
        Gera grade de boxes candidatas uniforme
        
        Args:
            image_size: Tamanho da imagem (H, W)
            grid_size: Número de divisões da grade
            
        Returns:
            Lista de boxes [x1, y1, x2, y2]
        """
        h, w = image_size
        boxes = []
        
        # Tamanhos para análise multi-escala (em % da menor dimensão)
        size_factors = [0.15, 0.2, 0.25]
        
        # Para cada célula da grade
        for i in range(grid_size):
            for j in range(grid_size):
                cx = w * (i + 0.5) / grid_size
                cy = h * (j + 0.5) / grid_size
                
                # Para cada tamanho
                for factor in size_factors:
                    size = int(min(h, w) * factor)
                    half_size = size / 2
                    
                    x1 = max(0, cx - half_size)
                    y1 = max(0, cy - half_size)
                    x2 = min(w, cx + half_size)
                    y2 = min(h, cy + half_size)
                    
                    # Adicionar apenas se a box tiver tamanho mínimo
                    if x2 - x1 >= 32 and y2 - y1 >= 32:
                        boxes.append([x1, y1, x2, y2])
        
        return boxes
    
    # Métodos compatíveis com o visualizador
    def visualize_recommendations(self, target_img, patch, recommendations=None, save_path=None, 
                                show_scores=True, show_simulated=False):
        """Interface compatível com o visualizador existente"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        
        if recommendations is None:
            recommendations = self.recommend_placements(target_img, patch)
            
        if not recommendations:
            print("Nenhuma recomendação encontrada")
            return None
            
        # Normalizar tensores para visualização
        if len(target_img.shape) == 4:
            target_img = target_img[0]
        if len(patch.shape) == 4:
            patch = patch[0]
        
        target_img_np = target_img.permute(1, 2, 0).cpu().numpy()
        target_img_np = np.clip(target_img_np, 0, 1)
        
        patch_np = patch.permute(1, 2, 0).cpu().numpy()
        patch_np = np.clip(patch_np, 0, 1)
        
        # Criar figura
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(target_img_np)
        
        # Mostrar patch como inset
        patch_ax = fig.add_axes([0.02, 0.02, 0.2, 0.2])
        patch_ax.imshow(patch_np)
        patch_ax.set_title("Patch")
        patch_ax.axis('off')
        
        # Cores para diferentes rankings
        colors = ['lime', 'cyan', 'yellow', 'magenta', 'white']
        
        # Desenhar recomendações
        for i, rec in enumerate(recommendations[:5]):
            x1, y1, x2, y2 = rec["box"]
            color = colors[i % len(colors)]
            
            # Retângulo
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Texto com scores
            if show_scores:
                score_text = f"#{i+1} Score: {rec['compatibility_score']:.2f}"
                ax.text(
                    x1, y1-5, score_text, color='white', fontsize=10, weight='bold',
                    bbox=dict(facecolor=color, alpha=0.7, pad=1)
                )
        
        ax.set_title("Recomendações de Colagem")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    # Método para simulação de colagem (compatibilidade com avaliação)
    def simulate_paste_result(self, image, patch, box, blend_mode='alpha', return_tensor=False):
        # Garantir formato correto
        if len(image.shape) == 4:
            image = image[0]
        if len(patch.shape) == 4:
            patch = patch[0]
    
        # Converter tensores para numpy para processamento
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()
    
        # Converter box para inteiros
        x1, y1, x2, y2 = [int(coord) for coord in box]
        target_h, target_w = y2 - y1, x2 - x1
    
        # Redimensionar patch para o tamanho da região-alvo
        import cv2
        patch_resized = cv2.resize(patch_np, (target_w, target_h))
    
        # Criar cópia da imagem para manipulação
        result = image_np.copy()
    
        # Alpha blending simples
        alpha = 0.7
        result[y1:y2, x1:x2] = alpha * patch_resized + (1-alpha) * result[y1:y2, x1:x2]
    
        # Retornar no formato apropriado
        if return_tensor:
            return torch.tensor(result).permute(2, 0, 1)
        return result