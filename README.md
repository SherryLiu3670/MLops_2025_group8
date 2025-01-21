# Fruit and Vegetable Classification

## Goal
This is the project description for Group 8 in the 02476 Machine Learning Operations course offered at DTU. The goal of this project is to design a machine learning lifecycle for fruit and vegetable vision-based classification. Throughout the project, in addition to basic model development, we will focus on utilizing machine learning operations (MLOps) concepts to maintain the model and support its deployment to a cloud-based processor.

## Framework
The initial plan is to use PyTorch as the framework to keep things simple. We will progressively improve the quality of the project by incorporating various tools available in the PyTorch ecosystem.

## Data
The dataset we have chosen is the Fruits and Vegetables Image Recognition Dataset from Kaggle, which consists of 36 different classes of fruits and vegetables with a total of 4,320 images. The food items included are:

- **Fruits:** Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango  
- **Vegetables:** Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant  

**Dataset link:** [Fruits and Vegetables Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

## Model
The core task in this project is image classification, and we plan to approach this by leveraging convolutional neural networks (CNNs). We will begin by designing our own models using basic CNN layers, such as convolutional, pooling, and fully connected layers, to gain an intuitive understanding of the dataset and the task at hand. These initial models will act as baselines and help us explore the challenges in the dataset, such as class imbalance or differences in image quality.

Once the baseline models are developed, we will move on to testing more advanced and pre-trained models. Specifically, we aim to explore:

- **ResNet (Residual Networks):** Known for their ability to address the vanishing gradient problem through skip connections, making them highly effective for deep networks.  
- **MobileNet:** Optimized for lightweight applications, it can serve as an excellent candidate for deployment in resource-constrained environments.

To ensure a systematic comparison, each model will be evaluated using standard metrics such as accuracy, precision, recall, and F1-score. We will also explore performance across different subsets of the dataset to check for robustness, such as performance on specific fruit or vegetable classes.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


## Invoke Commands

This project uses [Invoke](http://www.pyinvoke.org/) for task automation. Below are the available commands and their descriptions. Please ensure to have Invoke installed before running any commands.

### Prerequisites

To use `invoke`, you need to have Python installed. Then, install Invoke using the following command:
```bash
pip install invoke
```

### Available Commands

| Command                  | Description                                               |
|--------------------------|-----------------------------------------------------------|
| `invoke create-environment` | Creates a new conda environment for the project.          |
| `invoke requirements`    | Installs project requirements.                            |
| `invoke dev-requirements`| Installs development requirements.                        |
| `invoke preprocess-data` | Preprocesses the dataset.                                 |
| `invoke train`           | Trains the machine learning model.                       |
| `invoke test-model`      | Tests the machine learning model.                        |
| `invoke test`            | Runs the test suite and generates a coverage report.      |
| `invoke docker-build`    | Builds the Docker images for training and API.            |
| `invoke build-docs`      | Builds the project documentation.                        |
| `invoke serve-docs`      | Serves the project documentation locally.                |

### Usage

Run a command using the following syntax:
```bash
invoke <command>
```

#### Description

1. To create the conda environment:
   ```bash
   invoke create-environment
   ```

2. To install the project requirements:
   ```bash
   invoke requirements
   ```

3. To train the model:
   ```bash
   invoke train
   ```

4. To build Docker images:
   ```bash
   invoke docker-build
   ```

### Notes

- Use `invoke --list` to see all available tasks:
  ```bash
  invoke --list
  ```

- For detailed usage of a specific command, use `invoke --help <command>`:
  ```bash
  invoke --help train
  ```

This section will help collaborators and users understand and efficiently use the automation tasks provided in this project.
```
