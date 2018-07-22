# Deep Episodic Memory: Encoding, Recalling, and Predicting Episodic Experiences for Robot Action Execution

## Train

``` bash
python ./train.py --gpu 0
# or
mpirun -np 3 python train.py --multi-gpu
```

After running this script, trained data and log files are generated at `results` directory.

![graph](img/log.png)

## Visualize

``` bash
python ./predict.py predict --gpu 0 results/model_iter_1000
```

This script generates images where the outputs generated from trained network are visualized into `predict` directory.

## Summary

``` bash
python ./predict.py summary --gpu 0 results/model_iter_1000
```

![summary](img/summary.png)
