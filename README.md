# WAVE - Waveform Analysis and Velocity Estimation

WAVE is a deep learning project that explores the reconstruction of subsurface velocity maps from seismic shot-gather data.

The project is based on Full Waveform Inversion (FWI), where the goal is to estimate subsurface velocity structures from observed seismic wave data. In this repository, we treat the problem as a supervised learning task and use a VelocityGAN-style model trained on the OpenFWI FlatVel-A benchmark dataset.

## Problem Statement

Full Waveform Inversion is used to estimate subsurface properties, such as seismic velocity, from recorded seismic waveforms. Traditional FWI methods can be accurate, but they are computationally expensive and depend on careful physical modeling.

This project explores a data-driven approach where a neural network learns the mapping directly from examples:

```text
seismic shot-gather data -> velocity map
```

Given:

* **Input:** seismic shot-gather data
  `(5 sources x 1 component x 1000 time steps x 70 receivers)`

* **Output:** 2D velocity map
  `(70 x 70 spatial grid)`


## Dataset

This project uses the **OpenFWI FlatVel-A** dataset.

OpenFWI FlatVel-A is a synthetic benchmark dataset for Full Waveform Inversion. It contains flat-layered velocity models and corresponding seismic shot-gather data generated from those models.

Dataset source:
[https://openfwi-lanl.github.io/](https://openfwi-lanl.github.io/)

The dataset needs to be downloaded separately from the official OpenFWI source.

Dataset details used in this project:

* **Dataset:** OpenFWI FlatVel-A
* **Input:** seismic shot-gather data
* **Target:** subsurface velocity map
* **Velocity models:** synthetic flat-layered structures
* **Velocity range:** approximately 1500 to 4500 m/s


## Approach

We use a VelocityGAN-style setup for seismic velocity reconstruction.

The main components are:

* **Generator:** a CNN encoder-decoder model that maps seismic input to a velocity map
* **Discriminator:** a CNN model that classifies velocity maps as real or generated
* **Loss:** a combination of reconstruction loss (L1) and adversarial loss (BCE)

The training is supervised because ground-truth velocity maps are available for the seismic inputs.


## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/WAVE.git
cd WAVE
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the FlatVel-A dataset from the official OpenFWI website:

[https://openfwi-lanl.github.io/](https://openfwi-lanl.github.io/)

After downloading, place the dataset locally and update the `data_root` field in `config.yaml`.

Expected Structure:

```text
data/
└── FlatVel-A/
    ├── train/
    │   ├── seismic/
    │   └── velocity/
    └── val/
        ├── seismic/
        └── velocity/
```


## Training

```bash
python src/train.py
```

Training settings can be changed in `config.yaml`. Checkpoints are saved in the `checkpoints/` directory.

## Evaluation

```bash
python src/evaluate.py
```

Computes L1 loss, MSE, and optionally SSIM on the validation set. Requires a saved checkpoint; update `checkpoint_path` in `config.yaml`.


## Visualization

```bash
python src/visualize.py
```

Saves comparison plots to the `outputs/` directory. Each plot includes the seismic input, ground-truth velocity map, and predicted velocity map.

## Limitations

This project is limited to the OpenFWI FlatVel-A dataset, which contains synthetic flat-layered velocity models. 
