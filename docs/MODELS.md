# Important Caveat
**Please note that I found an indexing bug after publishing my work on arXiv and the nuScenes leaderboard and this fix is reflected in the code, so your results might be slightly different from my reported values.**

# Train models
In order to train models and validate them, run the following command line:
```bash
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS ./tools/nusc_shasta/train.py --launcher pytorch --seed 1 \
--config configs/nusc/OBJECT_TYPE.py --work_dir work_dir/RUN_NAME --project_name shasta --group_name RUN_NAME
```

An example of this for the object type car with 8 GPUs is in trainval.sh, which you can run. The train.py file invokes the validate.py file to validate the performance at each epoch.

# Evaluating models

If you want to reproduce my results, you can download my pre-trained models and run either the official_val.sh or official_test.sh shell script.
```bash
wget http://download.cs.stanford.edu/juno/ShaSTA/models.zip
unzip models.zip
chmod +x official_SPLIT.sh
./official_SPLIT.sh
```

If you want to only run evaluation to fine-tune parameters not required for training the model (e.g. track confidence refinement parameters), then you would use the eval.py file to load your pre-trained model and produce your tracks as such. 


# Download tracking results
If you would like to use the tracks generated for my paper for any extensions, you can download them below:
```bash
wget http://download.cs.stanford.edu/juno/ShaSTA/results.zip
unzip results.zip
```


