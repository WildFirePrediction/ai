# Rules for Agents

## Project Overview

This project is a wildfire spread prediction algorithm implemented by reinforcement learning (A3C+\alpha)

- Timeline : Due 29th November
- Objective : 70+% IoU

## Hardware restrictions

- CPU : AMD Ryzen9 9950x (16 physical cores, 32 logical)
- RAM : DDR5 64GB (5600MHz)
- GPU : NVIDIA GeForce RTX 5070 (GDDR7 12GB)
- SSD : Samsung 990PRO 2TB (29% Used)

## Development environment

- OS : Ubuntu 24.04.3 LTS
- IDE : Neovim (Or PyCharm)
- Env : Terminal (zsh, tmux) 

## Location

- Project root directory : ~/code/WildfirePrediction

## Immutable rules

- Before creating, read every documentation file (.md) first
- Do not create new documentation files (.md, .txt, etc) unless the user instructs to
- Take into account of hardware restrictions, calculate whether the model fits inside memory during training 
- Whenever something failes, look up official documentation for reference 
- Never run the full training, always give the user the command or guide to run full training
- When the user don't specify wandb integration, the default is yes
- Attach this file(AGENTS.md) context eveytime you act(plan, create a response, test, etc), to enforce the rules more effectively
- Always ask questions to the user to clarify every aspect of the act

## Code Styles

- Concise and readable (Extensive and descriptive code comments)
- Use reusable patterns if applicable
- Specify lots of default arguments which are proven to be effective before (i.e. minimum-episode-length 4, num-workers 4, lr, gamma, etc)
- Integrate GPU compute power (CUDA) as much as possible

## Workflow

- Plan -> Implement -> Test -> Debug -> Evaluate
