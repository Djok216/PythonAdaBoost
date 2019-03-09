# PythonAdaBoost
AdaBoost python implementation for 2d points with two labels (red and blue).

# Data format
```
NumberOfBluePoints
x1 y1
...
xn yn
NumberOfRedPoints
x1 y1
...
xm ym
```

# How to run example
```
python adaboost2d.py data/ex59.in withPlots
```
withPlots is an optional parameter (in case you want to see the distribution weight and the decision stump after every iteration).