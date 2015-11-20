# flda 
Feature-level domain adaptation

This repository contains 4 algorithms for feature-level domain adaptation. We implemented this approach using two loss functions (quadratic and logistic) and two transfer distributions (dropout and blankout).

- flda-qd-drop: quadratic loss and dropout transfer distribution
- flda-qd-blank: quadratic loss and blankout transfer distribution
- flda-log-drop: logistic loss and dropout transfer distribution
- flda-log-blank: logistic loss and blankout transfer distribution

Usage: Trains a linear domain adaptive classifier that can be applied to the target domain data to make predictions. <br>
Input: <br>
    - Source domain data:           XQ      (M features x NQ samples) <br>
    - Target domain data:           XP      (M features x NP samples) <br>
    - Source domain labels:         yQ      (NQ x 1) <br>
Output: <br>
    - Linear classifier:            W       (M features + 1 bias)
    - Transfer model parameters:    theta   (M x 1)

We are publishing a paper on this approach, which will be linked here when in press.

Dependencies:
 - MinFunc (http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
 
Datasets are available upon request (w.m.kouw at tudelft dot nl) 

