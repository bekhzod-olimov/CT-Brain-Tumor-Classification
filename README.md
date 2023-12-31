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

3. Data Visualization

![image](https://github.com/bekhzod-olimov/CT-Brain-Tumor-Classification/assets/50166164/dfc56a5a-3fc6-41de-8aae-d90df0d1db49)

4. Train the AI model using the following script:
```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```

The training parameters can be changed using the following information:

![image](https://github.com/bekhzod-olimov/BrainTumorClassification/assets/50166164/f41f2ada-bb1f-4fb7-b34b-f27134c5e015)

The training process progress:

![image](https://github.com/bekhzod-olimov/BrainTumorClassification/assets/50166164/e23d91c3-94f3-48ff-9709-3d0a795e0be2)

5. Learning curves:
   
Use [DrawLearningCurves](https://github.com/bekhzod-olimov/JellyfishClassifier/blob/80393cea3cdf497533f915d88481a3513b6cbcf7/main.py#L56C6-L56C6) class to plot and save learning curves.

* Train and validation loss curves:
  
![loss_learning_curves](https://github.com/bekhzod-olimov/CT-Brain-Tumor-Classification/assets/50166164/875bb60b-ec1a-473c-be6f-79ceacbe52ef)

* Train and validation accuracy curves:
  
![acc_learning_curves](https://github.com/bekhzod-olimov/CT-Brain-Tumor-Classification/assets/50166164/362d4984-852e-4688-85a3-9051bc507f46)

6. Conduct inference using the trained model:
```python
python inference.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```

The inference progress:

![image](https://github.com/bekhzod-olimov/BrainTumorClassification/assets/50166164/3e42d081-738f-431e-a993-ee5455849a26)

7. Inference Results (Predictions):

![brain_preds](https://github.com/bekhzod-olimov/CT-Brain-Tumor-Classification/assets/50166164/db6702ad-1c1c-4fdc-8d08-c35740393a70)
