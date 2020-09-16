# AgentNet
Pytorch implementation of AgentNet, which is designed for reveal hidden interactions and predict future dynamics of the unknown complex system.

## model
- AgentNet architectures for each model, Cellular Automata (CA), Vicsek model (VC), Active Ornstein-Uhlenbeck Particle(AOUP), and Chimney Swift flock (CS). 

## src
- Main source code includes utilities and loss functions

## train
- Training code for each model.
- Currently in multi-GPU settings (with DistributedDataParallel)

## Figures
- Supplementary figures for manuscript. 
- Additional attention analysis for different variables (for AOUP and CS).
