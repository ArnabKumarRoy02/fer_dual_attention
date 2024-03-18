# Dual Direction Attention Mixed Feature Network for Facial Emotion Recognition

This repository contains the code for the paper [A Dual-Direction Attention Mixed Feature Network for Facial Expression Recognition](https://doi.org/10.3390/electronics12173595). The code is based on the PyTorch library and is tested with Python 3.8. The code is for research purposes only.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Usage](#usage)
- [License](#license)


## Introduction

Facial expression recognition (FER) is a challenging task in computer vision and has many applications in human-computer interaction, human emotion analysis, and psychological research. In this paper, the authors proposed a novel Dual-Direction Attention Mixed Feature Network (DDAMFN) for FER. The proposed DDAMFN is designed to capture the spatial and channel-wise attention information from the input features. The DDAMFN consists of a dual-direction attention module (DDAM) and a mixed feature module (MFM). The DDAM is designed to capture the spatial and channel-wise attention information from the input features. The MFM is designed to capture the mixed feature information from the input features. The proposed DDAMFN is evaluated on three benchmark datasets, i.e., CK+, JAFFE, and FER2013. But this repository extended the benchmark with FER+ dataset. The experimental results demonstrate that the proposed DDAMFN outperforms the state-of-the-art methods on all the datasets.


## Installation

The code is based on the PyTorch library and is tested with Python 3.8. To use the code in this repository, run the following command:

```bash
git clone https://github.com/ArnabKumarRoy02/fer_dual_attention.git
```


## Dataset

The code in this repository is evaluated on the FER+ and CK+ dataset.

[FER+](https://github.com/microsoft/FERPlus.git): The FER+ dataset is an extension of the original FER dataset, where the images have been re-labelled into one of 8 emotion types: neutral, happiness, surprise, sadness, anger, disgust, fear, and contempt.

[CK+](https://www.kaggle.com/datasets/davilsena/ckdataset): The CK+ dataset contains 327 grayscale images of size 640x480 pixels. The images are divided into seven classes: angry, disgust, fear, happy, sad, surprise, and neutral.

But the paper has also been evaluated on other datasets as well.


## Methodology

The proposed Dual-Direction Attention Mixed Feature Network (DDAMFN) consists of a dual-direction attention module (DDAM) and a mixed feature module (MFM). The DDAM is designed to capture spatial and channel-wise attention information from the input features. It helps the model focus on important regions and channels for facial expression recognition. The MFM, on the other hand, captures mixed feature information from the input features, combining both low-level and high-level features. This allows the model to learn discriminative representations for facial expression recognition.


## Usage

To train the model, run the following command:

```bash
python train.py
```

To test the model, run the following command:

```bash
python test.py
```

Make sure to change the dataset path in the `train.py` and `test.py` files.

You can also check the checkpoints of the pretrained model in [this](checkpoints/) folder.


## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
