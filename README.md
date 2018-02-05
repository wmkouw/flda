# Feature-level domain adaptation

This repository contains MATLAB code accompanying the paper:

 [Kouw, WM, Krijthe, JH, Loog, M, & van der Maaten, LJP (2016). Feature-level
domain adaptation. Journal of Machine Learning Research, 17 (171), 1-32.](http://www.jmlr.org/papers/v17/15-206.html).

 For a cleaner implementation of flda as well as a translation into Python, see my library on transfer learners and domain-adaptive classifiers: [libTLDA](https://github.com/wmkouw/libTLDA).

## Installation
Clone the repository (bash):
```shell
git clone https://github.com/wmkouw/flda
```
Installation consists of adding the repository to your path (matlab):
```
addpath(genpath('./flda'))
```

### Dependencies
Flda depends on  [minFunc](http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) and [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/).
First download and extract them (bash):
```
wget http://www.cs.ubc.ca/~schmidtm/Software/minFunc_2012.zip -O minFunc.zip
unzip minFunc.zip

wget http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip -O libSVM.zip
unzip libSVM.zip
```

Then add them to your path (matlab):
```
addpath(genpath('./minFunc_2012'))
addpath(genpath('./libSVM-3.22'))
```

## Usage
Repo contains the following folders:
- __experiment-*__: contains scripts for running experiments reported in the paper.
- __data__: contains the digits, spam, office, imdb and amazon data sets.
- __util__: contains utility functions and algorithms.

To start an experiment, call the corresponding experiment function (matlab):
```
cd experiment-amazon/
run_daexp_amazon('flda_log_b')
```
Options for classifiers are:
- 'flda_log_b': flda with logistic loss and blankout transfer model
- 'flda_log_d': flda with logistic loss and dropout transfer model
- 'flda_qd_b': flda with quadratic loss and blankout transfer model
- 'flda_qd_d': flda with quadratic loss and dropout transfer model
- 'gfk_knn': geodesic flow kernel with a k-nearest-neighbour classifier
- 'tca_svm': transfer component analysis with a support vector machine
- 'sa_svm': subspace alignment with a support vector machine
- 'kmm': kernel mean matching with importance-weighted logistic regression
- 'scl': structural correspondence learning with logistic regression


### Contact
Bugs, comments and questions can be submitted to the issues tracker.
