## Requirements

Conda environments can be found in `/conda_ymls` (Note: Installation with M1 macs and newer might be inconsistent. Some required packages seem to not be compatible yet.)

Overall, the main packages required to run the code are the following:

- `python=3.10.12`
- `pytorch=2.2.0`
- `mlagents=1.0.0`
- `mlagents-envs=1.0.0`
- `numpy=1.21.2`

## Running the code

The main scripts consist of `train.py` and `evaluate.py` for training and evaluation respectively. Passing the `-h` flag with either should give all the information necessary.

An example:
```python evaluate.py --gpu --model_base_dir models/sampleModels --exp_name baseline --epoch 5000 --run_for 200```
This command runs and loads the models found under models/sampleModels/baseline/5000_*_actor on the GPU for 200 steps.

The files are all named after <epoch>_<agent>_<actor/critic>. So when you want to load a different model, simply browse the sample model files and take note of the epoch and which directory it is saved under.

Similarly, for training, we can do:
```python train.py --gpu --no_graphics --epochs 5000 --exp_name reaction_time_10 --reaction_time 10```
to train for 5000 epochs on agents with a reaction time of 10 frames
