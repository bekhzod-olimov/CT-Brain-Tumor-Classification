# CT-Brain-Tumor-Classification

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify computed tomography (CT) brain tumors. The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)) and the dataset for training is [A Refined Brain Tumor Image Dataset with Grayscale Normalization and Zoom](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256). The project in [Kaggle](https://www.kaggle.com/) can be found [here](https://www.kaggle.com/code/killa92/pytorch-resnet-18-99-9-accuracy).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/bekhzod-olimov/BrainTumorClassification.git
```

2. Create conda environment from yml file using the following script:
```python
conda env create -f environment.yml
```
Then activate the environment using the following command:
```python
conda activate speed
```

3. Train the AI model using the following script:
```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```
