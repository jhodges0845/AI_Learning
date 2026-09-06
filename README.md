# AI Learning Lab

A hands-on Python repository for learning machine learning by building, testing, and organizing algorithms into a reusable project structure.

Rather than treating machine learning as a collection of isolated notebooks, this project is structured around **supervised learning, unsupervised learning, neural networks, shared utilities, and repeatable demos**. The goal is to understand both the algorithms themselves and how ML code can be organized like maintainable software.

## Current Focus

The most complete implementation today is a reusable **linear regression workflow** that covers the full model lifecycle:

- Generate synthetic training data
- Split data into training and test sets
- Train a regression model
- Evaluate predictions
- Report R², MSE, and RMSE metrics
- Visualize the fitted model

The root `main.py` provides a runnable demonstration of that workflow.

## Learning Roadmap

The repository is organized to grow through several areas of machine learning.

### Supervised Learning

**Regression**
- ✅ Linear regression
- ⏳ Polynomial regression

**Classification**
- ⏳ Logistic regression
- ⏳ Decision trees

### Unsupervised Learning

Planned experiments and implementations for clustering, dimensionality reduction, and related techniques.

### Neural Networks

A dedicated area for progressing from traditional machine-learning algorithms into neural-network concepts and frameworks.

### Notebooks

Jupyter notebooks are used for experimentation, visualization, and deeper exploration alongside the reusable Python modules.

## Project Structure

```text
AI_Learning/
├── main.py
├── supervised/
│   ├── regression/
│   │   ├── linear_regression.py
│   │   └── polynomial_regression.py
│   └── classification/
│       ├── logistic_regression.py
│       └── decision_tree.py
├── unsupervised/
├── neural_networks/
├── notebooks/
├── utils/
├── requirements.txt
└── README.md
```

Some modules are intentionally present as placeholders for future learning milestones. The repository is a living learning project rather than a finished ML framework.

## Why This Repository Exists

My primary background is software engineering, systems integration, automation, and architecture. This project is my workspace for building a stronger understanding of machine learning from the implementation level upward.

The emphasis is not only on calling ML libraries, but also on understanding:

- How training and evaluation pipelines fit together
- How model quality is measured
- How data preparation affects results
- How ML experiments can become reusable software components
- How traditional machine learning connects to neural networks and modern AI systems

## Technology

The current environment includes:

- Python
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- JupyterLab
- TensorFlow
- XGBoost

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

macOS / Linux:

```bash
source venv/bin/activate
```

Windows PowerShell / Command Prompt:

```bash
venv\Scripts\activate
```

Git Bash on Windows:

```bash
source venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the current demo

```bash
python main.py
```

The demo trains and evaluates the current linear regression implementation and displays the resulting visualization.

## Jupyter Setup

With the virtual environment activated:

```bash
pip install ipykernel
python -m ipykernel install --user --name=ai-learning
jupyter lab
```

Select the `ai-learning` kernel from Jupyter when working with notebooks.

## Project Status

This repository is intentionally iterative. New algorithms will be added as I work through them, with an emphasis on keeping implementations organized, understandable, and reusable instead of accumulating disconnected experiments.

## Next Milestones

- Complete polynomial regression
- Add logistic regression
- Add decision-tree classification
- Expand model evaluation examples
- Add unsupervised-learning experiments
- Begin neural-network implementations
- Add tests around reusable model components
