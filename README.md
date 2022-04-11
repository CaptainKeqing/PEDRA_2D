# PEDRA_2D
 2D SIMULATION adapted from original PEDRA repository (https://github.com/aqeelanwar/PEDRA).
This project does not connect to AIRSIM, instead does the simulation and training in 2D.
Currently, this project implements the novel FCQN (fully convolutional Q-network) architecture
described in 
 
"Deep Reinforcement Learning based Automatic
 Exploration for Navigation in Unknown
 Environment"
 by Haoran Li, Qichao Zhang, Dongbin Zhao Senior Member, IEEE (https://arxiv.org/pdf/2007.11808.pdf)

Bayesian Hilbert Maps taken from (https://github.com/RansML/Bayesian_Hilbert_Maps)
# Installing
It is advisable to create a new virtual environment and install the necessary dependencies.
```
cd PEDRA_2D
pip install –r pedra_2d_requirements_cpu.txt
```

# Demonstration
Change the directories in infer_2D to your own and run with the latest weight (drone_2D_10000)
given in the results/weights directory.

# Training
Use main2D