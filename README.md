# Guided Adversarial Attack for Evaluating and Enhancing Adversarial Defenses

This repository contains code for the implementation of our paper titled "Guided Adversarial Attack for Evaluating and Enhancing Adversarial Defenses", accepted for a Spotlight presentation at NeurIPS 2020. Our paper is available on arXiv [here](https://arxiv.org/abs/2011.14969).

The proposed Guided Adversarial Margin Attack (GAMA) utilizes function mapping of the clean image to guide the generation of adversaries, thereby resulting in stronger attacks.
The following plot shows the Robust Accuracy (%) of different attacks against multiple random restarts. Evaluations are performed on TRADES WideResNet-34 model [1] for CIFAR-10, PGD-AT ResNet-50 model [2] for ImageNet (first 1000 samples), and TRADES SmallCNN model [1] for MNIST.

<p align="center">
    <img src="https://github.com/GaurangSriramanan/GAMA-GAT/blob/main/GAMA_accuracy_vs_rr.pdf" width="800"\>
</p>


# Environment Settings
+ Python: 3.6.9
+ PyTorch: 1.3.1
+ TorchVision: 0.4.2
+ Numpy: 1.19.0

[1] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. I. Jordan. Theoretically principled trade-off between robustness and accuracy. In International Conference on Machine Learning (ICML), 2019.

[2] L. Engstrom, A. Ilyas, H. Salman, S. Santurkar, and D. Tsipras. Robustness (python library), 2019. [link](https://github.com/MadryLab/robustness)
