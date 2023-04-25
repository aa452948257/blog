---
layout: post
title:  "Debugging Pandas Rolling Correlation Function Performance Code"
author: Dabai Wang
avatar: https://en.gravatar.com/userimage/234480055/86956a413a70581209f3acae5eadac48.png?size=200
image: /assets/images/pandas-performance-analysis/img.png
excerpt: "In this article, we explore performance testing and debugging of Pandas Rolling Correlation function code."
date: 2023-04-19
category: ARTICLES
---

## Introduction
In this article, we explore performance testing and debugging of Pandas Rolling Correlation function code. 
The Rolling correlation function is a very powerful Pandas function that allows rolling correlation calculations to be performed on a DataFrame or Series. 
However, we have found that the `DataFrame.rolling().agg('corr')` function is very slow when dealing with a large number of columns. 
Even when performing rolling operations on a DataFrame with empty rows but non-empty columns, it can still take a significant amount of time.
More details is shown in [Xorbits/issues](https://github.com/xprobe-inc/xorbits/issues/316).

We analyze the impact of different input formats on the performance of `DataFrame.rolling(window=30).agg('corr')` and 
use the `py-spy` package to visualize the time consumption of different modules through a heat map.

## Performance Testing

We use [Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) as test data, that record Historical daily prices and volumes of all U.S. stocks and ETFs.

### All dataset
```python
import pandas as pd
import time
return_matrix = pd.read_csv("../test_csv/returns.csv", index_col=0, parse_dates=True)
start_time = time.time()
roll = return_matrix.rolling(window=30).agg('corr')
end_time = time.time()
print("Execution time:", end_time - start_time)
```
Execution time: 105.7209062576294
### Half subset on axis=0
```python
import pandas as pd
import time
return_matrix = pd.read_csv("../test_csv/returns.csv", index_col=0, parse_dates=True)
return_matrix = return_matrix.iloc[: len(return_matrix) // 2]
start_time = time.time()
roll = return_matrix.rolling(window=30).agg('corr')
end_time = time.time()
print("Execution time:", end_time - start_time)
```
Execution time: 97.95154094696045

### Half subset on axis=1
```python
import pandas as pd
import time
return_matrix = pd.read_csv("../test_csv/returns.csv", index_col=0, parse_dates=True)
return_matrix = return_matrix[return_matrix.columns[: 500]]
start_time = time.time()
roll = return_matrix.rolling(window=30).agg('corr')
end_time = time.time()
print("Execution time:", end_time - start_time)
```
Execution time: 26.112659215927124

### Empty Dataset
```python
import pandas as pd
import time
return_matrix = pd.read_csv("../test_csv/returns.csv", index_col=0, parse_dates=True)
return_matrix = return_matrix.drop(return_matrix.index)
start_time = time.time()
roll = return_matrix.rolling(window=30).agg('corr')
end_time = time.time()
print("Execution time:", end_time - start_time)
```
Execution time: 89.09344792366028

From the four experiments above, using the entire dataset, half the dataset, the entire dataset (half the columns), and an empty dataset (all columns), it can be seen that computation time is dependent on the number of columns.

We use the `py-spy` package to visualize the time consumption of different modules as follows:
![](../assets/images/pandas-performance-analysis/img_1.png)

The main execution time is concentrated in the following 
[function](https://github.com/pandas-dev/pandas/blob/249d93e4abc59639983eb3e8fccac8382592d457/pandas/core/window/rolling.py#L557)
```python
def flex_binary_moment(arg1, arg2, f, pairwise=False):
    results = defaultdict(dict)
    for i in range(len(arg1.columns)):
        for j in range(len(arg2.columns)):
            if j < i and arg2 is arg1:
                # Symmetric case
                results[i][j] = results[j][i]
            else:
                results[i][j] = f(
                    *prep_binary(arg1.iloc[:, i], arg2.iloc[:, j])
                )
```

`flex_binary_moment` is used to calculate the rolling correlation between every two columns of Series. `results[i][j]` represents the rolling correlation between the i-th and j-th columns of Series. The two time-consuming parts of flex_binary_moment are `f()` (35.5%) and `prep_binary`(57.2%). `f()` calculates the rolling correlation, while `prep_binary` masks out rows with NaN values in the two columns of Series.

Some people may wonder why the time taken to calculate the rolling correlation is shorter than that of prep_binary. Let's take a look at the code for the two parts below.

### prep_binary
```python
def prep_binary(arg1, arg2):
    # mask out values, this also makes a common index...
    X = arg1 + 0 * arg2
    Y = arg2 + 0 * arg1

    return X, Y
```
Here is the prep_binary function executing a flame graph, which shows that a significant amount of time is spent on calculating X and Y.
![](../assets/images/pandas-performance-analysis/img_2.png)

### rolling correlation
```python
def corr_func(x, y):
    x_array = self._prep_values(x)
    y_array = self._prep_values(y)
    window_indexer = self._get_window_indexer()
    min_periods = (
        self.min_periods
        if self.min_periods is not None
        else window_indexer.window_size
    )
    start, end = window_indexer.get_window_bounds(
        num_values=len(x_array),
        min_periods=min_periods,
        center=self.center,
        closed=self.closed,
        step=self.step,
    )
    self._check_window_bounds(start, end, len(x_array))

    with np.errstate(all="ignore"):
        mean_x_y = window_aggregations.roll_mean(
            x_array * y_array, start, end, min_periods
        )
        mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
        mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
        count_x_y = window_aggregations.roll_sum(
            notna(x_array + y_array).astype(np.float64), start, end, 0
        )
        x_var = window_aggregations.roll_var(
            x_array, start, end, min_periods, ddof
        )
        y_var = window_aggregations.roll_var(
            y_array, start, end, min_periods, ddof
        )
        numerator = (mean_x_y - mean_x * mean_y) * (
            count_x_y / (count_x_y - ddof)
        )
        denominator = (x_var * y_var) ** 0.5
        result = numerator / denominator
    return Series(result, index=x.index, name=x.name)
```
The key here is that Pandas convert x_array and y_array to numpy arrays for Cython routines.
```python
x_array = _prep_values(x)
y_array = _prep_values(y)
```

## Solution
We propose a simple optimization method from a parallel perspective, and as analyzed earlier, 
reducing the columns significantly helps to reduce the computation time.
Assuming the original DataFrame has a shape of (365, N) and a parallelism degree of m, 
we can generate m Sub-DataFrames with a shape of (365, N/m). 
The parallelism degree can be set to the number of cores of the processor. 
To simplify, we ignore the communication time between different cores and broadcast the original DataFrame 
to each core before the calculation.

Without adding any parallel strategy, the number of computations is:

```math
C_1 = N^2
```

With any parallel strategy, since each core can operate simultaneously, the number of computations is:

```math
C_2 = (N/m)^2 + (N/m)^2 * (m - 1)
```
In C2, the first term is the required computation for the current Sub-DataFrame, and the second term is the required computation for the current Sub-DataFrame and other Sub-DataFrames. 
As can be seen, the computation is reduced after parallelization.

## Conclusion
In this article, we discovered the performance issue with `DataFrame.rolling().agg('corr')` 
and used the `py-spy` module to generate a flame graph that visualizes the time consumption of different modules. 
We analyzed that the root cause of the performance issue was the index alignment operation of two arrays in the `prep_binary` function. 
We also propose a simple optimization method from a parallel perspective.
We hope this blog can provide some help for developers who are dedicated to optimizing large-scale data computation with Pandas.
