# Debugging Pandas Rolling Correlation Function Performance Code

## Introduction
In this article, we will explore performance testing and debugging of Pandas Rolling Correlation function code. 
The Rolling correlation function is a very powerful Pandas function that allows rolling correlation calculations to be performed on a DataFrame or Series. 
However, we have found that the `DataFrame.rolling().agg('corr')` function is very slow when dealing with a large number of columns. 
Even when performing rolling operations on a DataFrame with empty rows but non-empty columns, it can still take a significant amount of time.

In this article, we analyze the impact of different input formats on the performance of `DataFrame.rolling(window=30).agg('corr')` and 
use the `py-spy` package to visualize the time consumption of different modules through a heat map.
