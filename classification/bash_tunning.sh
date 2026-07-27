source ~/anaconda3/etc/profile.d/conda.sh
conda init
conda activate cim_yolo

## RESNET34
# metaplasia
python classification/tune_hyperarameters.py --dataset 'metaplasia' --model-architecture 'resnet34' --augmentation-technique 'our_approach' --multiplier '1x' --n-trials 50

# polyps
python classification/tune_hyperarameters.py --dataset 'polyps' --model-architecture 'resnet34' --augmentation-technique 'our_approach' --multiplier '1x' --n-trials 50

# breast 
python classification/tune_hyperarameters.py --dataset 'breast' --model-architecture 'resnet34' --augmentation-technique 'our_approach' --multiplier '1x' --n-trials 50

# brain
python classification/tune_hyperarameters.py --dataset 'brain' --model-architecture 'resnet34' --augmentation-technique 'our_approach' --multiplier '1x' --n-trials 50

## VIT-B/16
# metaplasia
python classification/tune_hyperarameters.py --dataset 'metaplasia' --model-architecture 'vit_b_16' --augmentation-technique 'our_approach' --multiplier '1x' --n-trials 50

# polyps
python classification/tune_hyperarameters.py --dataset 'polyps' --model-architecture 'vit_b_16' --augmentation-technique 'our_approach' --multiplier '1x' --n-trials 50

# breast 
python classification/tune_hyperarameters.py --dataset 'breast' --model-architecture 'vit_b_16' --augmentation-technique 'our_approach' --multiplier '1x' --n-trials 50

# brain
python classification/tune_hyperarameters.py --dataset 'brain' --model-architecture 'vit_b_16' --augmentation-technique 'our_approach' --multiplier '1x' --n-trials 50