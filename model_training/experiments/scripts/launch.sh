CMD="python experiments/train.py \
    --config experiments/configs/train_config.py:gc_bc_offline_bridge \
    --bridgedata_config experiments/configs/data_config.py:all \
    "

# take in command line args too
$CMD $@
