# Guided Adversarial Attack for Evaluating and Enhancing Adversarial Defenses

This repository contains code for the implementation of our paper titled "Guided Adversarial Attack for Evaluating and Enhancing Adversarial Defenses", accepted for a Spotlight presentation at NeurIPS 2020. Our paper is available on arXiv [here](https://arxiv.org/abs/2011.14969).


 - We propose **Guided Adversarial Margin Attack(GAMA)**, which achieves state-of-the-art performance across multiple defenses for a single attack and across multiple random restarts.
 - We introduce a multi-targeted variant GAMA-MT, which achieves improved performance compared to methods that utilize multiple targeted attacks to improve attack strength [1]. 
 - We demonstrate that Projected Gradient Descent based optimization (GAMA-PGD) leads to stronger attacks when a large number of steps (100) can be used, thereby making it suitable for defense evaluation; whereas, Frank-Wolfe based optimization (GAMA-FW) leads to stronger attacks when the number of steps used for attack are severely restricted (10), thereby making it useful for adversary generation during multi-step adversarial training.
 - We propose **Guided Adversarial Training (GAT)**, which achieves state-of-the-art results amongst existing single-step adversarial defenses. We demonstrate that the proposed defense can scale to large network sizes and to large scale datasets such as ImageNet-100.
    
 # Guided Adversarial Margin Attack 

The proposed Guided Adversarial Margin Attack (GAMA) utilizes function mapping of the clean image to guide the generation of adversaries, thereby resulting in stronger attacks.
The following plot shows the Robust Accuracy (%) of different attacks against multiple random restarts. Evaluations are performed on TRADES WideResNet-34 model [2] for CIFAR-10, PGD-AT ResNet-50 model [3] for ImageNet (first 1000 samples), and TRADES SmallCNN model [2] for MNIST.

<p align="left">
    <img src="https://github.com/GaurangSriramanan/GAMA-GAT/blob/main/GAMA_Robustness_vs_RR.PNG" width="1000"\>
</p>


# Environment Settings
+ Python: 3.6.9
+ PyTorch: 1.3.1
+ TorchVision: 0.4.2
+ Numpy: 1.19.0

[1] S. Gowal, J. Uesato, C. Qin, P.-S. Huang, T. Mann, and P. Kohli. An alternative surrogate loss for pgd-based adversarial testing. arXiv preprint arXiv:1910.09338, 2019

[2] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. I. Jordan. Theoretically principled trade-off between robustness and accuracy. In International Conference on Machine Learning (ICML), 2019.

[3] L. Engstrom, A. Ilyas, H. Salman, S. Santurkar, and D. Tsipras. Robustness (python library), 2019. [link](https://github.com/MadryLab/robustness)
