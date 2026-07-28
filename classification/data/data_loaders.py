from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json

import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG')


def build_image_transform(size: int = 512):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


def load_json(path: Path) -> dict:
    with open(path, 'r') as handle:
        return json.load(handle)


def list_image_files(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []

    files = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(folder.glob(pattern))
    return sorted(set(files))


def _select_paths_by_indices(paths: Sequence[Path], indices: Sequence[int], label_name: str) -> List[Path]:
    selected_paths = []
    for index in indices:
        if index < 0 or index >= len(paths):
            raise IndexError(
                f'Index {index} is out of range for {label_name} with {len(paths)} files.'
            )
        selected_paths.append(paths[index])
    return selected_paths


def resolve_fold_indices(splits_data: dict, fold_names: Sequence[str]) -> Tuple[List[int], List[int]]:
    instance_indices: List[int] = []
    normal_indices: List[int] = []

    for fold_name in fold_names:
        fold_data = splits_data['folds'][fold_name]
        instance_key = 'instance_idxs' if 'instance_idxs' in fold_data else 'train_instance_indices'
        normal_key = 'normal_idxs' if 'normal_idxs' in fold_data else 'train_normal_indices'
        instance_indices.extend(fold_data.get(instance_key, []))
        normal_indices.extend(fold_data.get(normal_key, []))

    return sorted(set(instance_indices)), sorted(set(normal_indices))


def build_local_fold_selection(dataset_root: Path, splits_data: dict, fold_names: Sequence[str]) -> dict:
    dataset_root = Path(dataset_root)
    class_name = dataset_root.name

    instance_images = list_image_files(dataset_root / class_name / 'images')
    normal_images = list_image_files(dataset_root / 'normal')

    instance_indices, normal_indices = resolve_fold_indices(splits_data, fold_names)

    return {
        'instance_indices': instance_indices,
        'normal_indices': normal_indices,
        'instance_images': _select_paths_by_indices(instance_images, instance_indices, 'instance images'),
        'normal_images': _select_paths_by_indices(normal_images, normal_indices, 'normal images'),
    }


def build_strategy_selection(dataset_root: Path, strategy_metadata: dict, kfold_splits: dict) -> dict:
    train_folds = strategy_metadata['train_folds']
    val_fold = strategy_metadata['val_fold']
    test_fold = strategy_metadata['test_fold']

    return {
        'strategy_id': strategy_metadata['strategy_id'],
        'train_folds': train_folds,
        'val_fold': val_fold,
        'test_fold': test_fold,
        'train': build_local_fold_selection(dataset_root, kfold_splits, train_folds),
        'val': build_local_fold_selection(dataset_root, kfold_splits, [val_fold]),
        'test': build_local_fold_selection(dataset_root, kfold_splits, [test_fold]),
    }


class LocalBinaryClassificationDataset(Dataset):
    def __init__(self, image_paths, label, image_size=512, transform=None, source='local'):
        self.samples = [(Path(path), label) for path in image_paths]
        self.transform = transform or build_image_transform(image_size)
        self.source = source

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': str(image_path),
            'source': self.source,
        }


class SyntheticTarClassificationDataset(Dataset):
    def __init__(
        self,
        tar_path,
        image_size=512,
        strategy_id=None,
        train_folds=None,
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.transform = transform or build_image_transform(image_size)
        self.samples = []

        synthetic_iter = wds.WebDataset(str(tar_path), shardshuffle=False).decode('pil')
        allowed_folds = set(train_folds or [])

        for sample in synthetic_iter:
            metadata = sample.get('json', {}) or {}
            sample_strategy = metadata.get('strategy')
            sample_role = metadata.get('strategy_role')
            sample_train_folds = set(metadata.get('train_folds', []))

            if strategy_id is not None and sample_strategy != strategy_id:
                continue
            if allowed_folds and not sample_train_folds.intersection(allowed_folds):
                continue
            if sample_role is not None and sample_role != 'train':
                continue

            self.samples.append({
                'image': sample['jpg'],
                'path': metadata.get('source_id') or metadata.get('target_id') or sample.get('__key__', 'synthetic'),
                'source': 'synthetic',
            })

            if max_samples is not None and len(self.samples) >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = self.transform(image.convert('RGB'))

        return {
            'image': image,
            'label': torch.tensor(1, dtype=torch.long),
            'path': sample['path'],
            'source': sample['source'],
        }


def make_local_split_dataset(label_name: str, selection: dict, image_size=512, transform=None):
    if label_name == 'positive':
        image_paths = selection['instance_images']
        label = 1
    else:
        image_paths = selection['normal_images']
        label = 0
    return LocalBinaryClassificationDataset(image_paths, label=label, image_size=image_size, transform=transform)


def classification_collate(batch):
    return {
        'image': torch.stack([item['image'] for item in batch], dim=0),
        'label': torch.stack([item['label'] for item in batch], dim=0),
        'path': [item.get('path', '') for item in batch],
        'source': [item.get('source', '') for item in batch],
    }


# def build_classification_datasets(
#     dataset_root: Path,
#     strategy_metadata: dict,
#     kfold_splits: dict,
#     synthetic_tar_path: Path,
#     image_size: int = 512,
#     max_synthetic_samples: Optional[int] = None,
# ):
#     fold_selection = build_strategy_selection(dataset_root, strategy_metadata, kfold_splits)

#     train_selection = fold_selection['train']
#     val_selection = fold_selection['val']
#     test_selection = fold_selection['test']

#     train_real_positive = make_local_split_dataset('positive', train_selection, image_size=image_size)
#     train_real_negative = make_local_split_dataset('negative', train_selection, image_size=image_size)
#     val_positive = make_local_split_dataset('positive', val_selection, image_size=image_size)
#     val_negative = make_local_split_dataset('negative', val_selection, image_size=image_size)
#     test_positive = make_local_split_dataset('positive', test_selection, image_size=image_size)
#     test_negative = make_local_split_dataset('negative', test_selection, image_size=image_size)

#     train_synthetic = SyntheticTarClassificationDataset(
#         synthetic_tar_path,
#         image_size=image_size,
#         strategy_id=fold_selection['strategy_id'],
#         train_folds=fold_selection['train_folds'],
#         max_samples=max_synthetic_samples,
#     )

#     datasets = {
#         'fold_selection': fold_selection,
#         'train_real_positive': train_real_positive,
#         'train_real_negative': train_real_negative,
#         'train_synthetic': train_synthetic,
#         'val_positive': val_positive,
#         'val_negative': val_negative,
#         'test_positive': test_positive,
#         'test_negative': test_negative,
#         'train': ConcatDataset([train_real_negative, train_real_positive, train_synthetic]),
#         'val': ConcatDataset([val_negative, val_positive]),
#         'test': ConcatDataset([test_negative, test_positive]),
#     }

#     return datasets


# def build_classification_dataloaders(
#     dataset_root: Path,
#     strategy_metadata: dict,
#     kfold_splits: dict,
#     synthetic_tar_path: Path,
#     image_size: int = 512,
#     batch_size: int = 32,
#     num_workers: int = 0,
#     max_synthetic_samples: Optional[int] = None,
#     pin_memory: Optional[bool] = None,
# ):
#     datasets = build_classification_datasets(
#         dataset_root=dataset_root,
#         strategy_metadata=strategy_metadata,
#         kfold_splits=kfold_splits,
#         synthetic_tar_path=synthetic_tar_path,
#         image_size=image_size,
#         max_synthetic_samples=max_synthetic_samples,
#     )

#     if pin_memory is None:
#         pin_memory = torch.cuda.is_available()

#     loaders = {
#         'train': DataLoader(
#             datasets['train'],
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             collate_fn=classification_collate,
#         ),
#         'val': DataLoader(
#             datasets['val'],
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             collate_fn=classification_collate,
#         ),
#         'test': DataLoader(
#             datasets['test'],
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             collate_fn=classification_collate,
#         ),
#     }

#     return datasets, loaders


def build_classification_datasets(
    dataset_root: Path,
    strategy_metadata: dict,
    kfold_splits: dict,
    synthetic_tar_path: Path,
    image_size: int = 512,
    multiplier: str = '1x',  # <-- NOVO ARGUMENTO
    max_synthetic_samples: Optional[int] = None,
):
    fold_selection = build_strategy_selection(dataset_root, strategy_metadata, kfold_splits)

    train_selection = fold_selection['train']
    val_selection = fold_selection['val']
    test_selection = fold_selection['test']

    train_real_positive = make_local_split_dataset('positive', train_selection, image_size=image_size)
    train_real_negative = make_local_split_dataset('negative', train_selection, image_size=image_size)
    val_positive = make_local_split_dataset('positive', val_selection, image_size=image_size)
    val_negative = make_local_split_dataset('negative', val_selection, image_size=image_size)
    test_positive = make_local_split_dataset('positive', test_selection, image_size=image_size)
    test_negative = make_local_split_dataset('negative', test_selection, image_size=image_size)

    train_synthetic = SyntheticTarClassificationDataset(
        synthetic_tar_path,
        image_size=image_size,
        strategy_id=fold_selection['strategy_id'],
        train_folds=fold_selection['train_folds'],
        max_samples=max_synthetic_samples,
    )

    # ─── LÓGICA DO 0x ───
    # Começamos com as negativas reais (que são sempre precisas)
    train_components = [train_real_negative]
    
    # Só adicionamos as POSITIVAS REAIS se NÃO for a experiência 0x
    if multiplier != '0x':
        train_components.append(train_real_positive)
        
    # Adicionamos as positivas sintéticas
    train_components.append(train_synthetic)

    datasets = {
        'fold_selection': fold_selection,
        'train_real_positive': train_real_positive,
        'train_real_negative': train_real_negative,
        'train_synthetic': train_synthetic,
        'val_positive': val_positive,
        'val_negative': val_negative,
        'test_positive': test_positive,
        'test_negative': test_negative,
        'train': ConcatDataset(train_components), # <-- ARRAY CONDICIONAL
        'val': ConcatDataset([val_negative, val_positive]),
        'test': ConcatDataset([test_negative, test_positive]),
    }

    return datasets


def build_classification_dataloaders(
    dataset_root: Path,
    strategy_metadata: dict,
    kfold_splits: dict,
    synthetic_tar_path: Path,
    image_size: int = 512,
    batch_size: int = 32,
    multiplier: str = '1x',  # <-- NOVO ARGUMENTO
    num_workers: int = 0,
    max_synthetic_samples: Optional[int] = None,
    pin_memory: Optional[bool] = None,
):
    datasets = build_classification_datasets(
        dataset_root=dataset_root,
        strategy_metadata=strategy_metadata,
        kfold_splits=kfold_splits,
        synthetic_tar_path=synthetic_tar_path,
        image_size=image_size,
        multiplier=multiplier, # <-- PASSA A VARIÁVEL
        max_synthetic_samples=max_synthetic_samples,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=classification_collate,
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=classification_collate,
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=classification_collate,
        ),
    }

    return datasets, loaders
