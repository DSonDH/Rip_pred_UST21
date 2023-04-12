# Rip current prediction project
Study objective: find best modeling methods with interpreting model results.

## License
This project is released under the [Apache 2.0 license](LICENSE).


## Get started
mainly inplemented using Pytorch framework.

### Experimented model
* Prophet (nonparametric regression model)
* Machine Learning Models
    - Random Forest
    - Extreme Gradient Boosting
* MLP
    - MLP vanilla
    - Simple Linear
    - LightTS
* 1DCNN
    - Simple 1DCNN
    - SCINET
* RNN
    - LSTM
* Transformer
    - Transformer
    - Informer


### Explainable rip prediction model
regression coefficient (for Prophet)
SHAP: https://shap.readthedocs.io/en/latest/index.html
shapelets : https://tslearn.readthedocs.io/en/stable/user_guide/shapelets.html#id5
transformer visualization : 
    1. https://github.com/hila-chefer/Transformer-Explainability
    2. Temporal Fusion Transformers (TFT)
       https://www.youtube.com/watch?v=RuQQE1dBXkE


### Requirements
Install the required package first:
```
cd rippred
conda create -n rippred python=3.8
conda activate rippred
pip install -r requirements.txt
```

### Dataset preparation
We conduct the experiments on NIA rip current time-series datasets

The data directory structure is shown as follows. 
```
datasets/
└── NIA/
    └── obs_qc_100p
        ├── DC-total.csv
        ├── HD-total.csv
        ├── JM-total.csv
        ├── NS-total.csv
        └── SJ-total.csv
```

### how to run
```
python main.py --train_epochs 200 --batch_size 32 --patience 30 --lr 1e-3 --devices 0
```
