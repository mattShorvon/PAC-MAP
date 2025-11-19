# 0.1q 0.1e
DATETIME=$(date +"%d-%m-%Y %H:%M:%S")
# python make_queries_and_evids.py -dp 20-datasets -all -q 0.1 -e 0.1
# python benchmark.py -m MP AMP MS --no-learn --data-path 20-datasets -all -q 0.1 -e 0.1 -dt $DATETIME
# python plot_results.py -q 0.1 -e 0.1 -dn 20-datasets -dt $DATETIME
python make_queries_and_evids.py -dp small_datasets -all -q 0.1 -e 0.1
python benchmark.py -m MP AMP MS HBP --no-learn --data-path small_datasets -all -q 0.1 -e 0.1 -dt "$DATETIME"
python plot_results.py -q 0.1 -e 0.1 -dn small_datasets -dt "$DATETIME"