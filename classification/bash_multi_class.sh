#!/bin/bash

# Ativar ambiente
source ~/anaconda3/etc/profile.d/conda.sh
conda init
conda activate cim_yolo

# echo "Starting classification..."
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique cutmix --multiplier 1x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique cutmix --multiplier 2x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique cutmix --multiplier 3x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique cutmix --multiplier 0x --save-gradcam-samples 150 

# echo "Starting classification..."
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique saliencymix --multiplier 1x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique saliencymix --multiplier 2x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique saliencymix --multiplier 3x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique saliencymix --multiplier 0x --save-gradcam-samples 150

# echo "Starting classification..."
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique copypaste --multiplier 1x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique copypaste --multiplier 2x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique copypaste --multiplier 3x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique copypaste --multiplier 0x --save-gradcam-samples 150

# echo "Starting classification..."
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique our_approach --multiplier 1x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique our_approach --multiplier 2x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique our_approach --multiplier 3x --save-gradcam-samples 150 
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique our_approach --multiplier 0x --save-gradcam-samples 150

# echo "Starting classification..."
PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique cutmix --multiplier 1x --save-gradcam-samples 150 --model-architecture vit_b_16
PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique cutmix --multiplier 2x --save-gradcam-samples 150 --model-architecture vit_b_16
PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique cutmix --multiplier 3x --save-gradcam-samples 150 --model-architecture vit_b_16
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique cutmix --multiplier 0x --save-gradcam-samples 150 --model-architecture vit_b_16

# echo "Starting classification..."
PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique saliencymix --multiplier 1x --save-gradcam-samples 150 --model-architecture vit_b_16
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique saliencymix --multiplier 2x --save-gradcam-samples 150 --model-architecture vit_b_16
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique saliencymix --multiplier 3x --save-gradcam-samples 150 --model-architecture vit_b_16
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique saliencymix --multiplier 0x --save-gradcam-samples 150 --model-architecture vit_b_16

# echo "Starting classification..."
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique copypaste --multiplier 1x --save-gradcam-samples 150 --model-architecture vit_b_16
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique copypaste --multiplier 2x --save-gradcam-samples 150 --model-architecture vit_b_16
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique copypaste --multiplier 3x --save-gradcam-samples 150 --model-architecture vit_b_16
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique copypaste --multiplier 0x --save-gradcam-samples 150 --model-architecture vit_b_16

# echo "Starting classification..."
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique our_approach --multiplier 1x --save-gradcam-samples 150 --model-architecture vit_b_16
# PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique our_approach --multiplier 2x --save-gradcam-samples 150 --model-architecture vit_b_16
PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique our_approach --multiplier 3x --save-gradcam-samples 150 --model-architecture vit_b_16
PYTHONPATH=. python classification/main_multi_class.py --dataset multiclass --augmentation-technique our_approach --multiplier 0x --save-gradcam-samples 150 --model-architecture vit_b_16

