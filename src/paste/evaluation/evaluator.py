import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class RecommenderEvaluator:
    """Avalia o modelo AttentionRecommender com diagnóstico de contexto"""
    
    def __init__(self, model):
        self.model = model
        self.device = model.device
    
    def evaluate_recommender(self, val_dataset, num_samples=10, save_dir=None):
        """
        Avalia com diagnóstico visual de contexto vs objeto.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        device = self.device
        indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
        metrics = {
            'recommendation_scores': [],
            'context_activation': [],
            'object_activation': [],
            'context_to_object_ratio': []
        }
        
        self.model.eval()
        
        for idx in tqdm(indices, desc="Evaluating"):
            sample = val_dataset[idx]
            image = sample['image'].to(device)
            patches = sample['patches']  # ⚠️ Pode ser lista de dicts
            patch_boxes = sample['patch_boxes']
            segmentor_mask = sample.get('segmentor_target_mask', None)
            
            if len(patches) == 0:
                continue
            
            # ✅ CORREÇÃO: Extrair tensor do dicionário
            patch_dict = patches[0]
            if isinstance(patch_dict, dict):
                patch = patch_dict['object_only']
            else:
                patch = patch_dict
            
            orig_box = patch_boxes[0].tolist()
            
            with torch.no_grad():
                recs = self.model.recommend_placements(
                    image.unsqueeze(0), patch.unsqueeze(0), patch_box=orig_box, top_k=5
                )
                
                # Gera heatmap para diagnóstico
                pred_heatmap, _ = self.model(
                    image.unsqueeze(0), 
                    patch=patch.unsqueeze(0),
                    return_attention=True
                )
            
            if not recs:
                continue
            
            metrics['recommendation_scores'].extend([r['score'] for r in recs])
            
            # --- Diagnóstico ---
            if segmentor_mask is not None:
                if segmentor_mask.dim() == 4:
                    seg_mask = segmentor_mask[0, 0]
                elif segmentor_mask.dim() == 3:
                    seg_mask = segmentor_mask[0]
                else:
                    seg_mask = segmentor_mask
                
                seg_mask_binary = (seg_mask > 0.5).float().to(device)
                
                # Objeto = região original do patch
                obj_mask_binary = torch.zeros_like(seg_mask_binary)
                x1, y1, x2, y2 = map(int, orig_box)
                obj_mask_binary[y1:y2, x1:x2] = 1.0
                
                pred_heat_2d = pred_heatmap[0, 0]
                
                context_act = (pred_heat_2d * seg_mask_binary).sum() / (seg_mask_binary.sum() + 1e-6)
                object_act = (pred_heat_2d * obj_mask_binary).sum() / (obj_mask_binary.sum() + 1e-6)
                ratio = context_act / (object_act + 1e-6)
                
                metrics['context_activation'].append(float(context_act))
                metrics['object_activation'].append(float(object_act))
                metrics['context_to_object_ratio'].append(float(ratio))
        
        # Calcula estatísticas finais
        results = {}
        if metrics['recommendation_scores']:
            results['recommendation_scores'] = {
                'mean': float(np.mean(metrics['recommendation_scores'])),
                'std': float(np.std(metrics['recommendation_scores']))
            }
        
        if metrics['context_activation']:
            results['context_activation'] = {
                'mean': float(np.mean(metrics['context_activation'])),
                'std': float(np.std(metrics['context_activation']))
            }
            results['object_activation'] = {
                'mean': float(np.mean(metrics['object_activation'])),
                'std': float(np.std(metrics['object_activation']))
            }
            results['context_to_object_ratio'] = {
                'mean': float(np.mean(metrics['context_to_object_ratio'])),
                'std': float(np.std(metrics['context_to_object_ratio'])),
                'min': float(np.min(metrics['context_to_object_ratio'])),
                'max': float(np.max(metrics['context_to_object_ratio']))
            }
            
            print(f"\n*** DIAGNÓSTICO FINAL ***")
            print(f"Ativação CONTEXTO: {results['context_activation']['mean']:.4f} ± {results['context_activation']['std']:.4f}")
            print(f"Ativação OBJETO: {results['object_activation']['mean']:.4f} ± {results['object_activation']['std']:.4f}")
            print(f"Razão CONTEXTO/OBJETO: {results['context_to_object_ratio']['mean']:.2f}")
            if results['context_to_object_ratio']['mean'] > 1.0:
                print("✓ SUCESSO: Modelo aprendeu a preferir contexto!")
            else:
                print("✗ PROBLEMA: Modelo ainda prefere objeto.")
        
        if save_dir:
            with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_exogenous_patches(self, dataset, num_samples=15, save_dir=None):
        """Avalia com patches de outras imagens"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        device = self.device
        metrics = {'recommendation_scores': []}
        self.model.eval()
        
        for i in tqdm(range(num_samples), desc="Avaliando patches exógenos"):
            idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
            
            sample1 = dataset[idx1]
            sample2 = dataset[idx2]
            
            if not sample1['patches'] or not sample2['patches']:
                continue
            
            # ✅ CORREÇÃO: Extrair tensor do dicionário
            patch_dict = sample1['patches'][0]
            if isinstance(patch_dict, dict):
                patch = patch_dict['object_only'].to(device)
            else:
                patch = patch_dict.to(device)
            
            target_img = sample2['image'].to(device)
            
            with torch.no_grad():
                recs = self.model.recommend_placements(
                    target_img.unsqueeze(0),
                    patch.unsqueeze(0),
                    top_k=5
                )
            
            if recs:
                metrics['recommendation_scores'].extend([r['score'] for r in recs])
        
        results = {}
        if metrics['recommendation_scores']:
            scores = metrics['recommendation_scores']
            results['recommendation_scores'] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
        
        if save_dir:
            with open(os.path.join(save_dir, 'exogenous_metrics.json'), 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_segmentation_consistency(self, dataset, num_samples=20, save_dir=None):
        """Avalia consistência do mapa de calor com a segmentação"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        device = self.device
        metrics = {'iou_scores': [], 'dice_scores': []}
        self.model.eval()
        
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Avaliando consistência"):
            sample = dataset[i]
            image = sample['image'].to(device)
            patches = sample['patches']
            segmentor_mask = sample.get('segmentor_target_mask', None)
            
            if not patches or segmentor_mask is None:
                continue
            
            # ✅ CORREÇÃO: Extrair tensor do dicionário
            patch_dict = patches[0]
            if isinstance(patch_dict, dict):
                patch = patch_dict['object_only'].to(device)
            else:
                patch = patch_dict.to(device)
            
            # ✅ CORREÇÃO CRÍTICA: Mover segmentor_mask para o device
            if segmentor_mask is not None:
                segmentor_mask = segmentor_mask.to(device)
            
            with torch.no_grad():
                pred_heatmap, _ = self.model(
                    image.unsqueeze(0),
                    patch=patch.unsqueeze(0),
                    return_attention=True
                )
            
            # Calcula IoU e Dice
            if pred_heatmap.shape[-2:] != segmentor_mask.shape[-2:]:
                pred_heatmap = F.interpolate(
                    pred_heatmap,
                    size=segmentor_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # ✅ Agora ambos estão no mesmo device
            pred_binary = (pred_heatmap[0, 0] > 0.5).float()
            seg_binary = (segmentor_mask[0, 0] > 0.5).float()
            
            intersection = (pred_binary * seg_binary).sum()
            union = (pred_binary + seg_binary).clamp(0, 1).sum()
            iou = intersection / (union + 1e-6)
            
            dice = 2 * intersection / (pred_binary.sum() + seg_binary.sum() + 1e-6)
            
            metrics['iou_scores'].append(float(iou))
            metrics['dice_scores'].append(float(dice))
        
        results = {}
        if metrics['iou_scores']:
            results['iou'] = {
                'mean': float(np.mean(metrics['iou_scores'])),
                'std': float(np.std(metrics['iou_scores']))
            }
            results['dice'] = {
                'mean': float(np.mean(metrics['dice_scores'])),
                'std': float(np.std(metrics['dice_scores']))
            }
        
        if save_dir:
            with open(os.path.join(save_dir, 'consistency_metrics.json'), 'w') as f:
                json.dump(results, f, indent=2)
        
        return results