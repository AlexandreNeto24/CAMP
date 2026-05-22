"""
Dataset para carregar imagens sintéticas guardadas em HDF5.

Uso:
    synth_ds = SyntheticHDF5Dataset(hdf5_path='data/augmented/brain/our_approach/1x/brain_1x.h5')
    loader = DataLoader(synth_ds, batch_size=32)
"""

import h5py
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SyntheticHDF5Dataset(Dataset):
    """
    Dataset especializado para carregar imagens sintéticas guardadas em HDF5.
    
    Estrutura esperada do HDF5:
    ├─ sample_0/
    │  ├─ image [H, W, 3] (RGB, uint8)
    │  ├─ mask [H, W] (uint8)
    │  └─ attrs: {class, confidence, augmentation, fold, technique, multiplier}
    ├─ sample_1/
    │  └─ ...
    """
    
    def __init__(self, hdf5_path, image_size=(512, 512), transform=None):
        """
        Args:
            hdf5_path (str or Path): Caminho para o ficheiro HDF5
            image_size (tuple): Tamanho para redimensionar imagens (H, W)
            transform (callable, optional): Transformações opcionais
        """
        self.hdf5_path = Path(hdf5_path)
        self.image_size = image_size
        self.transform = transform
        
        # Contar número de samples
        with h5py.File(self.hdf5_path, 'r') as f:
            self.num_samples = len([k for k in f.keys() if k.startswith('sample_')])
        
        # Transforms padrão (redimensionamento)
        self.resize = transforms.Resize(self.image_size)
        self.to_tensor = transforms.ToTensor()
        
        if self.num_samples == 0:
            raise ValueError(f"Nenhum sample encontrado em {self.hdf5_path}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Carrega uma amostra do HDF5.
        
        Returns:
            dict: {
                'image': tensor [C, H, W],
                'mask': tensor [1, H, W],
                'metadata': dict com atributos
            }
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            sample_key = f'sample_{idx}'
            
            if sample_key not in f:
                raise KeyError(f"Sample {sample_key} not found in {self.hdf5_path}")
            
            sample = f[sample_key]
            
            # Carrega image e mask como arrays numpy
            image_np = np.array(sample['image'], dtype=np.uint8)
            mask_np = np.array(sample['mask'], dtype=np.uint8)
            
            # Converte para PIL
            image_pil = Image.fromarray(image_np, mode='RGB')
            mask_pil = Image.fromarray(mask_np, mode='L')
            
            # Redimensiona
            image_pil = self.resize(image_pil)
            mask_pil = self.resize(mask_pil)
            
            # Aplica transformações customizadas se fornecidas
            if self.transform:
                image_pil = self.transform(image_pil)
            
            # Converte para tensor
            image_tensor = self.to_tensor(image_pil)  # [C, H, W]
            mask_array = np.array(mask_pil, dtype=np.float32) / 255.0
            mask_tensor = self.to_tensor(mask_pil).unsqueeze(0)  # [1, H, W]
            
            # Extrai metadados
            metadata = dict(sample.attrs)
            
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'metadata': metadata,
                'source': 'synthetic'
            }


class CombinedDataset(Dataset):
    """
    Dataset que combina dados reais com dados sintéticos.
    
    Uso:
        real_ds = AttentionDataset(root_path='data/brain/brain')
        synth_ds = SyntheticHDF5Dataset(hdf5_path='data/augmented/brain/our_approach/1x/brain_1x.h5')
        
        combined = CombinedDataset([real_ds, synth_ds])
        loader = DataLoader(combined, batch_size=32)
    """
    
    def __init__(self, datasets, weights=None):
        """
        Args:
            datasets (list): Lista de datasets a combinar
            weights (list, optional): Pesos para cada dataset (probabilidade de selecção)
        """
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])
        self.total_size = self.cumulative_sizes[-1]
        
        if weights is None:
            # Igual probabilidade para cada dataset
            self.weights = [1.0 / len(datasets)] * len(datasets)
        else:
            assert len(weights) == len(datasets), "Número de weights deve corresponder ao número de datasets"
            self.weights = [w / sum(weights) for w in weights]  # Normaliza
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Encontra qual dataset tem este índice
        dataset_idx = 0
        for cumsum in self.cumulative_sizes:
            if idx < cumsum:
                break
            dataset_idx += 1
        
        # Calcula índice local dentro do dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][local_idx]
