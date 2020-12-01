# GAT: CIFAR-10

Run the following command to train a ResNet-18 network using Guided Adversarial Training:

`CUDA_VISIBLE_DEVICES=0 python GAT_cifar10.py -EXP_NAME 'CIFAR10_ResNet18_GAT' -MAX_EPOCHS 100 -l2_reg 10.0 -mul 4 -B_val 4.0 -Feps 8.0 -b_size 64 -lr_factor 10.0` 
