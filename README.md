# recommender
Collaborative Filtering for Movie Recommendation 

## To run experiments

* run `setup.sh`. This script just creates the required directories in the current working directory
* Each `run_*.sh` script runs the corresponding experiment.
* The difference between `run_grid.sh` and `run_grid2.sh` is that the second breaks the lambda1 parameter into many processes. So run `run_grid2.sh` preferably.

## New project structure

* `experiment_results/` where all the results of the experiments are stored
* `experiment_results/raw` where cluster outputs are stored
* `experiment_results/preprocessed` where the csvs are stored
* `experiment_results/graphs` where the graphs in pdfs are stored
* `model_results/` where important submissions are stored
