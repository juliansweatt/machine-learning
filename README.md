# Project 1
## Files
* [writeup.pdf](./writeup.pdf): PDF document which answers questions from the project requirements.
* [common.py](./common.py): Contains methods for generating random gaussian data sets and other common methods used throughout.
* [svm.py](./svm.py): Contains SVM class implementation.
* [svmDriver.py](./svmDriver.py): Driver for the SVM. Used to call the SVM from command line.
* [svmDriver.py](./regressionDriver.py): Driver for the Regression. Used to call the regressions from command line.
* [regression.py](./regression.py): Contains the linear and logistic regression implementation.

## Dependencies
* Python 3
* Python Packages
    * numpy
    * matplotlib
    * math
    * cvxpy

## General Execution Notice
As with all scripts, depending on the operating system being used, execution may vary slightly. When running as a script directly (for example: `svmDriver.py`) it may be necessary to preface the script with a path (`./svmDriver.py`) or grant execution permission to the script (for example: `chmod 777 svmDriver.py`). To avoid this problem, I prefaced script executions with the appropriate Python version `python3`, for example `python3 svmDriver.py`.

## SVM
### Run Instructions
The SVM is implemented in [svm.py](./svm.py), however, use [svmDriver.py](./svmDriver.py) to run the SVM from command line.

For the default SVM (C=1, plot graphically, do not print report), execute `python3 svmDriver.py`. The optional flag `--c` allows a custom constant c value to be used, for example, `python3 svmDriver --c=10`. The optional flag `--noPlot` will stop the plot from appearing graphically, for example `python3 svmDriver.py --noPlot`. The optional flag `--printReport` will print basic SVM information (information requested in question 1b) to the console. The optional flag `--brute` should only be used if a report is being printed, to calculate leave-one-out error by brute force retraining (will greatly increase run-time). 

For a full breakdown of the driver parameters, run `svmDriver.py -h` for help.

### Execution Steps
These are the steps used to generate the data discussed in the write up.

1. python3 svmDriver.py --printReport --brute
    * This command will generate an SVM where C=1, print vital information to console, and calculate the leave-one-out error by brute-force retraining.
    * Output is Figure 1 in the report.
2. python3 svmDriver.py --printReport --c=0
3. python3 svmDriver.py --printReport --c=10
4. python3 svmDriver.py --printReport --c=100
4. python3 svmDriver.py --printReport --c=1000

## Regression
### Run Instructions
The regression is implemented in [regression.py](./regression.py), however, use [regressionDriver.py](./regressionDriver.py) to run the regression from command line.

For the linear regression, execute `python3 regressionDriver.py linear`. For the logistic regression, execute `python3 regressionDriver.py logistic`.

For a full breakdown of the driver parameters, run `regressionDriver.py -h` for help.

### Execution Steps
These are the steps used to generate the data discussed in the write up.

1. python3 regressionDriver.py linear --loo
2. python3 regressionDriver.py logistic  --loo