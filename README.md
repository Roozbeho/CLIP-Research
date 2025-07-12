# CLIP Research: Zero-Shot vs. Linear Probe Classification

This project provides a comprehensive evaluation and comparison of two classification techniques using pre-trained CLIP models: **Zero-Shot Classification** and **Linear Probe**. The experiments are conducted on the MNIST dataset of handwritten digits, exploring how different text prompts and evaluation methods impact model performance.

The entire codebase is designed with a clean, modular architecture and is controlled with a flexible command-line interface.

* **for the full report, take look at docs/ directory.**

##  Features

* **Modular Architecture**: The codebase is cleanly separated into modules for data loading, model logic, configuration, and visualization, making it easy to read, maintain, and extend.
* **Command-Line Interface**: All experiment parameters (model name, batch size, GPU usage, etc.) are configurable via command-line arguments, eliminating the need to edit source code.
* **Dual Evaluation Modes**: Implements and directly compares Zero-Shot and Linear Probe classification pipelines.
* **Prompt Engineering Analysis**: Tests a wide variety of text prompts to analyze their influence on zero-shot accuracy.
* **Performance Optimized**: Includes custom data processing pipelines that significantly reduce evaluation time (from \~40 minutes to \~5 minutes).
* **Automated Visualization**: Generates and saves bar charts comparing the performance of different models and prompts for easy analysis.

##  Project Structure

```
clip_research/
├── main.py              # Main entry point to run experiments
├── dataset.py           # Defines the custom MNISTDataSet class
├── data_loader.py       # Prepares and creates the DataLoaders
├── zero_shot.py         # Logic for the Zero-Shot classification model
├── linear_probe.py      # Logic for the Linear Probe evaluation model
│
├── utils/
│   ├── config.py        # Manages project configuration via a @dataclass
│   └── visualization.py # Handles the generation of result charts
│
└── requirements.txt     # Lists all project dependencies
```

##  Setup and Installation

To get started, clone the repository and install the required dependencies.

### 1. Clone the repository:

```bash
git clone https://github.com/Roozbeho/CLIP-Research
cd clip_research
```

### 2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate 
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

##  Usage

The main script `main.py` is controlled via command-line arguments.

### Basic Execution

To run the full evaluation pipeline (Zero-Shot and Linear Probe) with default settings:

```bash
python main.py
```

### Generating Visual Reports

To run the evaluation and save a PNG chart of the results:

```bash
python main.py --visualize
```

This will save a file named `evaluation_report.png` in your project directory.

### Example with Custom Parameters

To run with a different CLIP model, change the batch size, and disable CUDA:

```bash
python main.py --model-name ViT-B/16 --batch-size 500 --no-cuda
```

### View All Options

To see a full list of available command-line arguments:

```bash
python main.py --help
```
