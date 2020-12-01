# GAT: ImageNet100

Run the following command to train a ResNet-18 network using Guided Adversarial Training:

`CUDA_VISIBLE_DEVICES=0 python GAT-ImageNet100.py --lr 0.1 --batch-size 64 -a resnet18 "<dataset_path>" --lce 1 --Bval 4 --Feps 8 --l2_reg 20 --mul 7 --epochs 120 --gpu 0 --EXP_NAME "ImageNet100_ResNet18_GAT" ` 

The file ImageNet100_classes.txt contains the subset of 100 class Ids that we use from ImageNet. The directory "<dataset_path>" is expected to contain the ImageNet-100 images, split across three folders: 'train', 'val' and 'test'.
