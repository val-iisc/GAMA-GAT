Run the following command to evaluate a ResNet-18 network with Guided Adversarial Margin Attack (GAMA) using Projected Gradient Descent:

`CUDA_VISIBLE_DEVICES=0 python GAMA_PGD.py -EXP_NAME "GAMA_PGD_EVAL" -MODEL "<FILE with MODEL WEIGHTS>"` 

Run the following command to evaluate a ResNet-18 network with Guided Adversarial Margin Attack (GAMA) using Frank-Wolfe Optimization:

`CUDA_VISIBLE_DEVICES=0 python GAMA_FW.py -EXP_NAME "GAMA_FW_EVAL" -MODEL "<FILE with MODEL WEIGHTS>"` 


For the evaluation of Networks with other architectures, the arch.py file can be modified as required.
