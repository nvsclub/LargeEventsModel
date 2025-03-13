# A Scalable Approach for Unified Large Events Models in Soccer

This repository contains the implementation of the research presented in the paper "A Scalable Approach for Unified Large Events Models in Soccer" by Tiago Mendes-Neves, Luís Meireles, and João Mendes-Moreira from Faculdade de Engenharia da Universidade do Porto and LIAAD - INESC TEC, Portugal.

## Abstract

Large Events Models (LEMs) are a class of models designed to predict and analyze the sequence of events in soccer matches, capturing the complex dynamics of the game. The original LEM framework, based on a chain of classifiers, faced challenges such as synchronization, scalability issues, and limited context utilization. This paper proposes a unified and scalable approach to model soccer events using a tabular autoregressive model. Our models demonstrate significant improvements over the original LEM, achieving higher accuracy in event prediction and better simulation quality, while also offering greater flexibility and scalability. The unified LEM framework enables a wide range of applications in soccer analytics that we display in this paper, including real-time match outcome prediction, player performance analysis, and game simulation, serving as a general solution for many problems in the field.

## Demo
[Watch the demonstration video](https://youtu.be/IjThR71EZ0Y)

## Project Structure

The project consists of three main scripts:

1. **Preprocess Data to LEM.py**: Handles the preprocessing of soccer event data into the LEM standard format.
2. **Train Tabular LEMs.py**: Trains various neural network architectures (MLPs) on the preprocessed data.
3. **Benchmark Tabular LEMs.py**: Evaluates model performance through comprehensive benchmarking.

The remaining notebooks contain analysis and application examples.

## Usage
### 0. Data
This implementation is built for Wyscout V3 data, which should be organized in the following structure:
- competitions.csv
- seasons.csv
- matches.csv
- seasons/events/{season_id}.feather

### 1. Preprocessing Data

```bash
python "0000 Preprocess Data to LEM.py" \
  --data_dir /path/to/wyscout/data \
  --output_dir /path/to/processed/data \
  --seq_lengths 1 3 5 7 9
```

This script performs three main tasks:
- Converts raw data to LEM standard format
- Preprocesses data for tabular models
- Preprocesses data for time series models (commented out by default)

### 2. Training Models

```bash
python "0001 Train Tabular LEMs.py" \
  --mode [survey|full] \
  --data_dir /path/to/processed/data \
  --output_dir /path/to/model/output \
  --seq_lengths 1 3 5 7 9
```

The script supports two training modes:
- **survey**: Quick training to compare different architectures
- **full**: Complete training of selected architectures

### 3. Benchmarking Models

```bash
python "0003 Benchmark Tabular LEMs.py" \
  --data_dir /path/to/data \
  --model_dir /path/to/model/files \
  --output_dir /path/to/benchmark/results \
  --seq_len 3 \
  --n_sims 10000
```

This script performs comprehensive benchmarking including:
- Model performance metrics (accuracy, F1-score)
- Distribution analysis of predictions vs real data
- Simulation analysis for game outcomes
- Visualization of results

## Papers

If you use this code or find it helpful for your research, read:

```
@article{mendesneves2024,
  author = {Mendes-Neves, Tiago and Meireles, Luís and Mendes-Moreira, João},
  title = {Towards a foundation large events model for soccer},
  journal = {Machine Learning},
  volume = {113},
  number = {11},
  pages = {8687-8709},
  year = {2024},
  doi = {10.1007/s10994-024-06606-y},
  url = {https://doi.org/10.1007/s10994-024-06606-y},
  issn = {1573-0565},
}

@misc{mendesneves2024estimatingplayerperformancedifferent,
      title={Estimating Player Performance in Different Contexts Using Fine-tuned Large Events Models}, 
      author={Tiago Mendes-Neves and Luís Meireles and João Mendes-Moreira},
      year={2024},
      eprint={2402.06815},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.06815}, 
}

```

## License
GNU Affero General Public License (AGPL)
The AGPL mandates that modified source code must be made openly available when the software is distributed or used as a network service.