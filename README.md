# Collaborative Filtering

This project is part of the Computational Intelligence Lab course (2019) at ETH.

The project website can be found [here](https://inclass.kaggle.com/c/cil-collab-filtering-2019).

Team: 	sasglentame

| Name  | Email |
| ------------- | ------------- |
| Sotiris Anagnostidis  | sanagnos@student.ethz.ch  |
| Adamos Solomou  | solomoua@student.ethz.ch  |
| Ioannis Sachinoglou  | saioanni@student.ethz.ch  |
| ----  | sanagnos@student.ethz.ch  |



## Project structure

    .
    ├── data                               # should contain files data_train.csv  sampleSubmission.csv
    ├── experiment_results                 # results from experiments
        ├── graphs                         # directory for saving graphs
        ├── preprocessed                   # directory for saving .csv
        ├── raw                            # directory for saving outputs
    ├── report                              
        ├── report.pdf                     # Final report
    ├── src 
        ├── experiments                    # experiment scripts
    ├── requirements.txt
    └── README.md
    

## Getting Started

To run on cluster:

 ```
 # create environment
 python -m venv "cil-2019"
 
 # activate environment 
 source cil-2019/bin/activate
 
 # or use existing environment
 # source $HOME/.local/bin/virtualenvwrapper.sh
 # workon "cil-2019"
 
 # install dependencies 
 pip install --user -r requirements.txt
 
 # load modules
 module load python_gpu/3.6.4
  ```
  
To replicate final submission: 
```
python main.py 
```

To replicate experiments:
```
cd src/experiments # You should be in the experiment directory for the experiments to run
python <experiment> 
```

To run cross-validation test:
```
python cross_validation.py --model <model> [--<parameters> <value>]
```

Valid models names are the following: 

```autoencoder```, ```bsgd``` , ```svd_shrinkage```, ```ncf```

## Documentation
[Report](https://github.com/adamsolomou/recommender/blob/master/report/report.pdf) 
