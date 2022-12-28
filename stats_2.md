Mean, Median,Mode
--------------------------------------------

For a nominal level(that can't be ordered), you can only use the mode to find the most frequent value. But avoid using it for Category(might be used for some cases),Ratio as repeating values are rare

For an ordinal level or ranked data, you can also use the median to find the value in the middle of your data set. 

For interval or ratio levels, in addition to the mode and median, you can use the mean to find the average value.
Outlier will effect on the mean


# Skew
----------

skewness is a degree of asymmetry observed in a probability distribution that deviates from the symmetrical normal distribution (bell
curve) in a given set of data.

Normal distribution - Mean, Median, Mode all are same

Skewed distributions - One side has a more spread out and longer tail with fewer scores at one end than the other 

For data from skewed distributions, the median is better than the mean because it isn’t influenced by extremely large values.

In a positively(right) skewed distribution, there’s a cluster of lower scores and a spread out tail on the right and the central tendency of your dataset is on the lower end of possible scores.

In a positively skewed distribution, mode < median < mean.

In a negatively(left) skewed distribution, there’s a cluster of higher scores and a spread out tail on the left
and the central tendency of your dataset is towards the higher end of possible scores.

In a negatively skewed distribution, mean < median < mode.

### perarson's 2nd coefficient : Mean - Mode ~ 3*(Mean - Median) , it's immune to high-low value of  mode

#### If the skewness is between -0.5 & 0.5, the data are nearly symmetrical.

#### If the skewness is between -1 & -0.5 (negative skewed) or between 0.5 & 1(positive skewed), the data are slightly skewed.

### If the skewness is lower than -1 (negative skewed) or greater than 1 (positive skewed), the data are extremely skewed.


# Kurtosis
------------

Kurtosis refers to the degree of presence of outliers in the distribution.


# Z-test
---------------
A z-test is a statistical test used to determine whether two population means are different when the variances are known and the sample size is large. In z-test mean of the population is compared.

Ho: Sample mean is same as the population mean(Null hypothesis)

Ha: Sample mean is not same as the population mean(Alternate hypothesis)

z = (x — μ) / (σ / √n),


### If z value is less than critical value accept null hypothesis else reject null hypothesis.

### The decision to reject the null hypothesis could be incorrect, it is known as Type I error.

### The decision to retain the null hypothesis could be incorrect, it is know as Type II error.


# T-test

In t-test the mean of the two given samples are compared. A t-test is used when the population parameters (mean and standard deviation) are not known.

# Paired T-Test

Tests for the difference between two variables from the same population( pre- and post test score). 

# Independent T-test
### (two sample / student’s t-test)

It's a statistical test that determines whether there is a statistically significant difference between the means in two unrelated groups.For example -comparing boys and girls in a population.

# One sample t-test

The mean of a single group is compared with a given mean. For example-to check the increase and decrease in sales if the average sales is given.

t = (x1 — x2) / (σ / √n1 + σ / √n2),

# Anova test

It checks if the means of two or more groups are significantly different from each other. 

Ho: All pairs of samples are same i.e. all sample means are equal

Ha: At least one pair of samples is significantly different

F= ((SSE1 — SSE2)/m)/ SSE2/n-k, where

SSE = residual sum of squares

m = number of restrictions

k = number of independent variables

#### One-Way ANOVA  

when there're three or more separate, non-overlapping groups or data sets that you want to compare. It's similar to an independent-samples t-test, just with more group

#### Within-groups (Repeated measures) ANOVA   

when you've eparate data sets that are provided by the same people over time . It's similar to a dependent-samples t-test, just with more data sets. This is most often used when you are doing a longitudinal study that tracks the same people across time.


#### Factorial ANOVA 

is used when You are interested in the interaction between two or more variables/factors/dimensions. Essentially, you are comparing the means of the various combinations of factors.

One thing that is important to note about ANOVAs is that because there are more than two groups that are being compared, follow-up (or post-hoc) tests are often required to further interpret the data.

# Non parametric statistical test are used when data is not normally distributed.

# Chi-square test( χ2 test)

chi-square test is used to compare two categorical variables. 

Ho: Variable x and Variable y are independent

Ha: Variable x and Variable y are not independent.

χ2 = (o-e)^2/e
where o=observed , e=expected.



