# PYSPN

This is a python implementation of my [SPN library](https://gitlab.com/marcheing/spn), which is a C++ implementation based on [RenatoGeh's SPN library](https://github.com/RenatoGeh/gospn/)

## Usage
Instead of command-line options, this program uses a "spn.json" file on the config folder. The following keys are mandatory:

* "log-path" With the value being the filename where the info log will be written.
* "random" Currently a hash with one configuration: "seed", which must be an integer
* "actions" with a list of actions and their parameters.

### Actions currently implemented

* "spn" to learn an SPN With the following parameters:
  * "dataset": With the name of the .arff or .data file
  * "learning-data-proportion": A float between 0 and 1 for the percentage of the dataset to be used to learn (the rest is stored for classification actions)
  * "kclusters": Number of kclusters to use on learning
  * "pval": The pval for the g-test
  * "name": A string to use for reference to the other actions

* "export" to export an spn to a spn file:
  * "spn": The name of the SPN to export
  * "filename": The name of the file to be written

* "load" to import a previously exported spn file:
  * "filename": The name of the file to be read
  * "name": The SPN name to use for reference to the other actions

* "graph" to export a .dot representation of the SPN:
  * "spn": The name of the SPN to export
  * "filename": The name of the .dot file to be written"

* "classify" to execute a simple classification test
  * "spn": The name of the SPN to classify

#### Map Actions

With the key "map", the program will expect a hash with the map algorithm and its configuration

* "spn": The name of the SPN to use
* "algorithm": The name of the map algorithm from this list:
  * "naive": Test every combination of values
  * "max-product": The max-product algorithm (Poon2011)
  * "argmax-product": The argmax-product algorithm (Conaty2017)
  * "marginal-checking": The Max-search algorithm (Mei2018) with the marginal checking heuristic
  * "forward-checking": The Max-search algorithm (Mei2018) with the forward checking heuristic
  * "branch-and-bound": (WIP) Implementation of the branch and bound algorithm with Lagrangian relaxation


##### Article Script

In order to reproduce the results found on the article, run

```bash
python article_script.py PATH_TO_SPNS
```

Where the PATH_TO_SPNS should be a folder with spns divided into subfolders.
Each SPN subfolder should have:
- A .spn file with the format accepted by this library
- A .query file with the contents: N QUERY_VARIABLE_ID_1 QUERY_VARIABLE_ID_2 ... QUERY_VARIABLE_ID_N
- A .evid file with the contents: N EVIDENCE_VARIABLE_ID_1 VALUE_FOR_EVIDENCE_VARIABLE_1 EVIDENCE_VARIABLE_ID_2 VALUE_FOR_EVIDENCE_VARIABLE_2 ... EVIDENCE_VARIABLE_ID_N VALUE_FOR_EVIDENCE_VARIABLE_N

Any variable id not described in either file will be included as a marginalized variable for the MAP algorithms.
For a good example of this structure, the [learned-spns repository](https://gitlab.com/pgm-usp/learned-spns) was the one used on the artice.
