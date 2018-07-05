moving_mnist
============

### Unsupervised Learning of Video Representations using LSTMs

#### Training

``` bash
python ./train_lstm.py --gpu 0 --batch-size 4 --in-episodes 5 --out-episodes 5 --snapshot-interval 1000
```

After running this script, trained data and log files are generated at `lstm_results` directory.

#### Visualization

``` bash
python ./predict_lstm.py --gpu 0 --in-episode 5 --out-episode 5 lstm_results/model_iter_100000
```

This script generates images where the outputs generated from trained network are visualized into `lstm_predict` directory.


### Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

#### Training


``` bash
python ./train_conv_lstm.py --gpu 0 --in-episodes 5 --out-episodes 5 --batch-size 4
```

After running this script, trained data and log files are generated at `conv_lstm_results` directory.

#### Visualization

``` bash
python ./predict_conv_lstm.py --gpu 1 --in-episode 5 --out-episode 5 conv_lstm_results/model_iter_100000
```

This script generates images where the outputs generated from trained network are visualized into `conv_lstm_predict` directory.
