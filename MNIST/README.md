# GAT: MNIST

Run the following command to train a network with a modified LeNet architecture using Guided Adversarial Training:

`CUDA_VISIBLE_DEVICES=0 python GAT-MNIST.py -EXP_NAME 'MNIST_MLeNet_GAT' -MAX_EPOCHS 50 -l_ce 1 -l2_reg 15 -mul 3 -Feps 0.3 -B_val 0.3 -b_size 64 -lr_factor 5` 
