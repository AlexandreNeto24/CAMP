#!/bin/bash

# Ativar ambiente
source ~/anaconda3/etc/profile.d/conda.sh
conda init
conda activate cim_yolo

# echo "Starting classification..."

echo "Starting classification of metaplasia dataset with our_approach augmentation technique and 1x multiplier..."
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '1x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004
echo "Starting classification of metaplasia dataset with our_approach augmentation technique and 2x multiplier..."
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '2x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004
echo "Starting classification of metaplasia dataset with our_approach augmentation technique and 3x multiplier..."
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '3x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004

# echo "Starting classification of metaplasia dataset with copypaste augmentation technique and 1x multiplier..."
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '1x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004
#  echo "Starting classification of metaplasia dataset with copypaste augmentation technique and 2x multiplier..."
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '2x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004
#  echo "Starting classification of metaplasia dataset with copypaste augmentation technique and 3x multiplier..."
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '3x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004

# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '1x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '2x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '3x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004

# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '1x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '2x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '3x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004

python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '1x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '2x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '3x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001

# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '1x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '2x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '3x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001

# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '1x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '2x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '3x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001

# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '1x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '2x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '3x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001



python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '1x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '2x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '3x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001

# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '1x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '2x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '3x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001

# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '1x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '2x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '3x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001

# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '1x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '2x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '3x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.0001


python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '1x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '2x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '3x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001

# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '1x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '2x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '3x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001

# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '1x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '2x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '3x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001

# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '1x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '2x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '3x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.0001 --lr 0.00001

# echo "Starting classification..."

echo "Starting classification of metaplasia dataset with our_approach augmentation technique and 1x multiplier..."
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '1x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.00004 --model-architecture 'vit_b_16'
echo "Starting classification of metaplasia dataset with our_approach augmentation technique and 2x multiplier..."
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '2x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.00004 --model-architecture 'vit_b_16'
echo "Starting classification of metaplasia dataset with our_approach augmentation technique and 3x multiplier..."
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '3x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.00004 --model-architecture 'vit_b_16'

# echo "Starting classification of metaplasia dataset with copypaste augmentation technique and 1x multiplier..."
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '1x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.00004 --model-architecture 'vit_b_16'
#  echo "Starting classification of metaplasia dataset with copypaste augmentation technique and 2x multiplier..."
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '2x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.00004 --model-architecture 'vit_b_16'
#  echo "Starting classification of metaplasia dataset with copypaste augmentation technique and 3x multiplier..."
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '3x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.00004 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '1x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.00004 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '2x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.00004 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '3x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.00004 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '1x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '2x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '3x' --dataset 'metaplasia' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.001 --lr 0.0004 --model-architecture 'vit_b_16'

python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '1x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '2x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '3x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '1x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '2x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'   
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '3x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '1x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '2x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '3x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '1x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '2x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '3x' --dataset 'polyps' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.004 --lr 0.00001 --model-architecture 'vit_b_16'



python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '1x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '2x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '3x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '1x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '2x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '3x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '1x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '2x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '3x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '1x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '2x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '3x' --dataset 'breast' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.003 --lr 0.000007 --model-architecture 'vit_b_16'


python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '1x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '2x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'
python classification/main_classification.py --augmentation-technique 'our_approach' --multiplier '3x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '1x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'copypaste' --multiplier '2x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-Ptechnique 'copypaste' --multiplier '3x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '1x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '2x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'cutmix' --multiplier '3x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'

# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '1x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'
# python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '2x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'
python classification/main_classification.py --augmentation-technique 'saliencymix' --multiplier '3x' --dataset 'brain' --save-gradcam-samples 100 --warmup-epochs 7 --num-epochs 20 --warmup-lr 0.002 --lr 0.00003 --model-architecture 'vit_b_16'

