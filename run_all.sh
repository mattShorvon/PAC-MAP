# 0.1q 0.1e
DATETIME=$(date +"%d-%m-%Y %H:%M:%S")
python experiment_scripts/make_queries_and_evids.py -dp 20-datasets -all -q 0.1 -e 0.1
python experiment_scripts/benchmark.py -m MP AMP MS PACMAP --no-learn --data-path 20-datasets -all -q 0.1 -e 0.1 -dt "$DATETIME"
python experiment_scripts/plot_results.py -q 0.1 -e 0.1 -dn 20-datasets -dt "$DATETIME"
# python experiment_scripts/make_queries_and_evids.py -dp small_datasets -all -q 0.1 -e 0.1
# python experiment_scripts/benchmark.py -m MP AMP MS HBP PACMAP --no-learn --data-path small_datasets -all -q 0.1 -e 0.1 -dt "$DATETIME"
# python experiment_scripts/plot_results.py -q 0.1 -e 0.1 -dn small_datasets -dt "$DATETIME"

# 0.25q 0.25e
# DATETIME=$(date +"%d-%m-%Y %H:%M:%S")
# python experiment_scripts/make_queries_and_evids.py -dp 20-datasets -all -q 0.25 -e 0.25
# python experiment_scripts/benchmark.py -m MP AMP MS PACMAP --no-learn --data-path 20-datasets -all -q 0.25 -e 0.25 -dt "$DATETIME"
# python experiment_scripts/plot_results.py -q 0.25 -e 0.25 -dn 20-datasets -dt "$DATETIME"
# python experiment_scripts/make_queries_and_evids.py -dp small_datasets -all -q 0.25 -e 0.25
# python experiment_scripts/benchmark.py -m MP AMP MS HBP PACMAP --no-learn --data-path small_datasets -all -q 0.25 -e 0.25 -dt "$DATETIME"
# python experiment_scripts/plot_results.py -q 0.25 -e 0.25 -dn small_datasets -dt "$DATETIME"

# 0.4q 0.4e
# DATETIME=$(date +"%d-%m-%Y %H:%M:%S")
# python experiment_scripts/make_queries_and_evids.py -dp 20-datasets -all -q 0.4 -e 0.4
# python experiment_scripts/benchmark.py -m MP AMP MS PACMAP --no-learn --data-path 20-datasets -all -q 0.4 -e 0.4 -dt "$DATETIME"
# python experiment_scripts/plot_results.py -q 0.4 -e 0.4 -dn 20-datasets -dt "$DATETIME"
# python experiment_scripts/make_queries_and_evids.py -dp small_datasets -all -q 0.4 -e 0.4
# python experiment_scripts/benchmark.py -m MP AMP MS HBP PACMAP --no-learn --data-path small_datasets -all -q 0.4 -e 0.4 -dt "$DATETIME"
# python experiment_scripts/plot_results.py -q 0.4 -e 0.4 -dn small_datasets -dt "$DATETIME"