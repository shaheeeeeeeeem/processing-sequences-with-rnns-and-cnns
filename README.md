# Processing Sequences with RNNs and CNNs

This repository contains my notes, experiments, and implementations while I
work through a chapter on processing sequential data with recurrent and
convolutional neural networks.

> **Work in progress:** I am updating this repository as I continue learning
> through the chapter. Some notebook cells may be incomplete or contain saved
> errors while I test and revise the code.

## Current Progress

The current notebook focuses on time-series forecasting with daily Chicago
Transit Authority (CTA) ridership data.

Topics explored so far:

- Loading, cleaning, and visualizing time-series data with pandas
- Weekly and yearly seasonal-naive forecasting baselines
- Rolling averages and differencing
- ARIMA forecasting with `statsmodels`
- Creating a sliding-window `Dataset` and `DataLoader` in PyTorch
- Training a linear forecasting baseline
- Building a simple recurrent neural network with `torch.nn.RNN`
- Evaluating forecasts with mean absolute error

Planned as the chapter progresses:

- Deeper and more capable recurrent architectures
- LSTM and GRU models
- One-dimensional convolutional networks for sequences
- Additional experiments, comparisons, and notes

## Repository Structure

```text
.
|-- notebooks/
|   |-- RNNs.ipynb
|   `-- datasets/
|       |-- ridership.tgz
|       `-- ridership/
|           `-- CTA_-_Ridership_-_Daily_Boarding_Totals.csv
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Getting Started

1. Clone the repository and enter the project directory.

   ```bash
   git clone https://github.com/shaheeeeeeeeem/processing-sequences-with-rnns-and-cnns.git
   cd processing-sequences-with-rnns-and-cnns
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies.

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start Jupyter and open the notebook.

   ```bash
   jupyter lab
   ```

Run `RNNs.ipynb` from inside the `notebooks` directory so its relative dataset
paths resolve correctly. The notebook can also download and extract the
ridership archive if the local dataset is missing.

## Dataset

The notebook uses the **CTA Ridership: Daily Boarding Totals** dataset. A copy
is included for reproducibility, and the notebook's download helper retrieves
the archive from the data repository accompanying Aurélien Géron's machine
learning examples:

<https://github.com/ageron/data/raw/main/ridership.tgz>

## Notes

- The notebook automatically selects CUDA, Apple MPS, or CPU depending on
  availability.
- Exact package versions are intentionally not pinned yet because this is an
  active learning workspace.
- Results and implementations may change as I revisit earlier sections.
