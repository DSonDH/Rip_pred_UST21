# SCINet

## SCINet License
This project is released under the [Apache 2.0 license](LICENSE).


## Get started

### Experimented model
* SCINet
* MLP
* LSTM
* Transformer


### Explainable rip prediction model
T.B.D.


### Requirements
Install the required package first:
```
cd SCINet
conda create -n scinet python=3.8
conda activate scinet
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
python run_NIA.py --train_epochs 200 --batch_size 32 --patience 30 --lr 1e-3 --devices 0
```