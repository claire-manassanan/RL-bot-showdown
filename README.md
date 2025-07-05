# Reinforcement Learning for Pokémon Showndown using poke-env and StableBaselines3
### About the project
This is a group university project in the subject "Research in Data Science" performed by me and [my classmate](https://github.com/Pynochio). We implemented the Double Battle using the [poke-env](https://github.com/hsahovic/poke-env) to build the agent and
applied the algorithm from [stable baselines 3](https://github.com/DLR-RM/stable-baselines3) to our agent.  
___
### What you need to install
- python ver. 3.9 or above
- node.js
- stablebaselines3
- poke-env  
To start, just follow these steps. First, install the libraries:
```bash
pip install -r requirements.txt
```
Start your own local server by typing this command:
```bash
node pokemon-showdown start --no-security
```
  
**⚠Warning:** the number of iterations in `A2C_full_network.py` script leave it with 1M iterations of training, it might crash you computer. And if it doesn't, it will take around 3-4 hours depend on your spec.
Then, just simply run:
```bash
python3 A2C_full_network.py
```
  
To evaluate run:
```bash
python3 Evaluate.py
```
This will test and show the number of winning round at the end of the evaluation.
