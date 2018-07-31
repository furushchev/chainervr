# Chainer Video Representation

Chainer implementation of Networks for Learning Video Representations

## Contents

### Unsupervised Learning of Video Representations using LSTMs

Located at `models/unsupervised_videos`.

```
Srivastava, Nitish, Elman Mansimov, and Ruslan Salakhudinov.
Unsupervised learning of video representations using lstms."
International conference on machine learning. 2015.
```

See https://github.com/emansim/unsupervised-videos for the original implementation.


### Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

Located at `models/conv_lstm`.

```
Xingjian, S. H. I., et al.
"Convolutional LSTM network: A machine learning approach for precipitation nowcasting."
Advances in neural information processing systems. 2015.
```

### Deep Episodic Memory: Encoding, Recalling, and Predicting Episodic Experiences for Robot Action Execution

Located at `models/deep_episodic_memory`.

```
Rothfuss, Jonas, et al.
"Deep Episodic Memory: Encoding, Recalling, and Predicting Episodic Experiences for Robot Action Execution."
arXiv preprint arXiv:1801.04134 (2018).
```

## Install

1. Clone this repository
2. Install this package using `pip`

``` bash
cd chainervr
pip install .
```

3. (Optional) If you plan to use with GPU, please install appropriate `cupy` package.

``` bash
pip install cupy-cuda91  # for CUDA 9.1
# or
pip install cupy-cuda92  # for CUDA 9.2
# and so on.
```


## Examples

- [Reconstruction and Prediction of Moving Mnist Dataset](examples/moving_mnist)

## Awesome References

- [Lecture CS231n at Stanford Univ.](http://cs231n.github.io/convolutional-networks/)
- [Padding of ConvNet in Tensorflow](https://www.tensorflow.org/api_guides/python/nn#convolution)

## Author

Yuki Furuta <<furushchev@jsk.imi.i.u-tokyo.ac.jp>>
