# 0.1q 0.1e
python make_queries_and_evids.py -dp 20-datasets -all -q 0.1 -e 0.1
python benchmark.py -m MP AMP MS --no-learn --data-path 20-datasets -all -q 0.1 -e 0.1
python make_queries_and_evids.py -dp small_datasets -all -q 0.1 -e 0.1
python benchmark.py -m MP AMP MS HBP --no-learn --data-path small_datasets -all -q 0.1 -e 0.1