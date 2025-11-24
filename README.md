# MAP Benchmarks with SPNs

This repo contains code to carry out experiments comparing different approximate
MAP inference methods against each other, on the same input SPN model so that
all performance differences are down to the methods themselves and not the
circuits they are working with being better or worse fitted to the data.

## How to install and run

- Initialise a conda/virtual environment
- Run pip install requirements.txt
- Run the experiment pipeline with run_all.sh

The run_all.sh script runs three files in sequence that together constitute the
full benchmark experiment pipeline:

- make_queries_and_evids.py creates .map files in each dataset folder that
  contain randomly chosen query variables, evidence variables and evidence assignments for each
  MAP query to test the methods with
- benchmark.py loads the MAP queries and spns for each dataset, runs them through
  the specified methods, and saves the max probabilities and runtimes
- plot_results.py loads the results, aggregates them and re-formats them, also
  producing a results summary.

For the experiment pipeline, you need data for the experiments contained in a
folder in the root directory with the following structure:

pyspn-may312020/
├── run_all.sh
├── 20-datasets
│ ├── accidents
│ ├── ad
│ └── etc ...

You can download the 20 datasets,from this repo: https://gitlab.com/pgm-usp/learned-spns

You can download the 'small_datasets' collection from this version of the above repo:
https://gitlab.com/pgm-usp/learned-spns/-/tree/0242367e3fa014d2dc86c8200196153e5a86c87a
(at least a version of the repo around this time)

You then need to specify the query & evidence proportion parameters. For a full
description of the input arguments for each python script in the run_all pipeline,
see the python files.
