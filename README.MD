# Project Repository

This repository contains source code for exploratory data analysis and graph neural network classifiers. The project is organized into three main directories: `eda_src`, `classifier_src`, and `classifier_src_soc`.

## Directory Structure

  * **eda\_src**: Contains code and notebooks for exploratory data analysis.
  * **classifier\_src**: Contains baseline and lightweight implementations suitable for consumer hardware.
  * **classifier\_src\_soc**: Contains training scripts optimized for the SoC cluster using SLURM.

## Environment Setup

### Exploratory Data Analysis (`eda_src`)

Running the code in the `eda_src` directory requires a Python environment that satisfies the requirements file located within that directory.

```bash
pip install -r eda_src/requirements.txt
```

### Classifiers (`classifier_src` and `classifier_src_soc`)

Running code in these directories requires a specific Conda environment. Follow the steps below to set it up:

1.  Create the environment using the configuration file found in the root directory:

    ```bash
    conda env update --file env_new.yml --prune
    ```

2.  Activate the environment.

3.  Explicitly install PyTorch, DGL, and KaggleHub using the following commands (these must be installed manually as the configuration file does not set them up correctly):

    ```bash
    conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install -c dglteam/label/th21_cu121 dgl
    pip install kagglehub
    ```

## Usage

### Exploratory Data Analysis

The `eda_src` directory contains the main visualization notebooks and helper Python modules required to generate visualizations.

### Local Training (`classifier_src`)

This directory contains baseline and lightweight implementations designed to run on consumer hardware with low end GPU (minimum 6GB VRAM recommended).

  * **baseline.ipynb**: Baseline implementation.
  * **convnet.ipynb**: Lightweight GCN implementation.
  * **modules/**: Helper functions used by the notebooks.

### Cluster Training (`classifier_src_soc`)

This directory contains Python training scripts re-written for the SoC cluster using SLURM for faster training.

**Available Models:**
There are 6 training scripts available covering the following architectures, both with and without Random Walk Positional Encoding:

  * GCN (Graph Convolutional Network)
  * GINE (Graph Isomorphism Network with Edge features)
  * GT (Graph Transformer)

**Running Jobs:**

  * SLURM batch job scripts are provided for each training script.
  * **Note:** The first training job submission will take longer than usual as the dataset must be downloaded.
  * **Access:** Refer to `classifier_src_soc/README.md` for instructions on setting up access to the SoC cluster.

**Logs:**
The `classifier_src_soc/logs` directory contains training logs from previous runs used to generate the results in the project report.