# Reinforcement Learning Quickstart

To get setup with the repository,
```bash
git clone https://github.com/datares/rl-quickstart.git && cd rl-quickstart
```

To setup the development environment

```bash
conda create -n python=3.9 rl
conda activate rl
pip install -r requirements.txt
```

The finally to train the agents
```bash
python main.py
```


## Viewing Results
At the end of training, a window should open to display the trained agent.

To run tensorboard, run the following in a new terminal window
```bash
tensorboard --logdir=logs
```

Videos are also saved to the `video` directory, which has `.mp4`s of the testing results.
