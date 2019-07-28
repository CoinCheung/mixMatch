

get cifar-10 dataset: 
```
    $ wget - http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
```


1. use exponential moving average (EMA) to update model parameters.

2. though softmax has negative impact on the training with mse loss, the paper still use mse loss (from softmax predictions of unlabeled data to guessed label) to train the model.

3. it is better to warmup the balance factor between the labeled loss and the unlabeled loss. The official repository let the factor improve from 0.0x100 to 1.0x100 during the whole 1024 epoches. Maybe it is better to slowly increase the contribution of the unlabeled loss.
