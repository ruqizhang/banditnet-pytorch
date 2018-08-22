# Banditnet
A Pytorch implementation of ["Deep Learning with Logged Bandit Feedback"](http://www.cs.cornell.edu/people/tj/publications/joachims_etal_18a.pdf).
# Dependencies
* Python 2
* Pytoch 0.3.0

# Prepare Dataset
Download CIFAR 10 Python version from (https://www.cs.toronto.edu/~kriz/cifar.html) to `/data`.
Unpack the file by running
```
tar xvfz cifar-10-python.tar.gz
```

# Usage
To run banditnet with `$\lambda = 0.9$`
```
python main_ips.py --l 0.9
```
# Result
* 50000 number of Bandit-Feedback Examples with Resnet 18: 89% Accuracy



