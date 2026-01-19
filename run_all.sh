# 0.1q 0.9e
DATETIME=$(date +"%d-%m-%Y %H:%M:%S")
python experiment_scripts/make_queries_and_evids.py -dp 20-datasets -all -q 0.1 -e 0.9
python experiment_scripts/benchmark.py -m AMP IND MP PACMAP PACMAP-H --no-learn --data-path 20-datasets -all -q 0.1 -e 0.9 -dt "$DATETIME"
python experiment_scripts/plot_results.py -q 0.1 -e 0.9 -dn 20-datasets -dt "$DATETIME"

# 0.25q 0.75e
# DATETIME=$(date +"%d-%m-%Y %H:%M:%S")
python experiment_scripts/make_queries_and_evids.py -dp 20-datasets -all -q 0.25 -e 0.75
python experiment_scripts/benchmark.py -m AMP IND MP PACMAP PACMAP-H --no-learn --data-path 20-datasets -all -q 0.25 -e 0.75 -dt "$DATETIME"
python experiment_scripts/plot_results.py -q 0.25 -e 0.75 -dn 20-datasets -dt "$DATETIME"

# 0.5q 0.5e
# DATETIME=$(date +"%d-%m-%Y %H:%M:%S")
python experiment_scripts/make_queries_and_evids.py -dp 20-datasets -all -q 0.5 -e 0.5
python experiment_scripts/benchmark.py -m AMP IND MP PACMAP PACMAP-H --no-learn --data-path 20-datasets -all -q 0.4 -e 0.6 -dt "$DATETIME"
python experiment_scripts/plot_results.py -q 0.4 -e 0.6 -dn 20-datasets -dt "$DATETIME"