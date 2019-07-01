# Collaborative Filtering

This project is part of the Computation Intelligence Lab course (2019) at ETH.

Team: 	sasglentame

| Name  | Email |
| ------------- | ------------- |
| Adamos Solomou  | solomoua@student.ethz.ch  |
| Anagnostidis Sotiris  | sanagnos@student.ethz.ch  |
| Yannis Sach  | saioanni@student.ethz.ch  |
| ----  | sanagnos@student.ethz.ch  |



## Project structure

    .
    ├── data                               # should contain files data_train.csv  sampleSubmission.csv
    ├── src                                                     
    ├── report                              
    │   ├── report.pdf                      # Final report.
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

TODO

To run cross-validation test:

TODO

TODOS:
 - argpartser on main
 - argparser on cross_validation
 - create table A_train for cross_validation
