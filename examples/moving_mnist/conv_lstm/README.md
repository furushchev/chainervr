# Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting


## Train

``` bash
python ./train.py --gpu 0
```

After running this script, trained data and log files are generated at `results` directory.

## Visualize

``` bash
python ./predict.py --gpu 0 results/model_iter_100000
```

This script generates images where the outputs generated from trained network are visualized into `predict` directory.
