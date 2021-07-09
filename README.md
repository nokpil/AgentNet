# AgentNet
Pytorch implementation of AgentNet, which is designed for reveal hidden interactions and predict future dynamics of the unknown complex system.
Paper : https://arxiv.org/abs/2001.02539

## model
- AgentNet architectures for each model, Cellular Automata (CA), Vicsek model (VC), Active Ornstein-Uhlenbeck Particle(AOUP), and Chimney Swift flock (CS). 

## src
- Main source code includes utilities and loss functions.

## train
- Training code for each model.
- Currently in multi-GPU settings (with DistributedDataParallel).

## data_generation
- Generates training/test dataset for the model systems.
- Produced dataset needs to be placed in './data/{CA, Vicsek, AOUP}', respectively for further training.

## data_generation_cs
 - This code needs pre-processed system data from the original chimney swift trajectory data (Evangelista, D. J. et al., Three-dimensional trajectories and network analyses of group behaviour within chimney swift flocks during approaches to the roost, 2017, https://royalsocietypublishing.org/doi/full/10.1098/rspb.2016.2602). 
- The pre-procssed data is uploaded at (https://zenodo.org/record/5084183#.YOft42gzaUl), and needs to be placed in './data/Flock/system'
- The produced dataset needs to be placed in './data/Flock/dataset'for further training.

## Figures
- Supplementary figures for manuscript. 
- Additional attention analysis for different variables (for AOUP and CS).
