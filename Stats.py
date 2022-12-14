########   Statistical Analysis we should look for to understand the relationship among the features ########
##################################################################################################################

Descriptive Statistics
========================
allow you to characterize your data based on its properties. 
There are four major types of descriptive statistics:

1. Measures of Frequency -  Count, Percent, Frequency
=========================
   Shows how often something occurs
->   Use this when you want to show how often a response is given

2. Measures of Central Tendency - Mean, Median, and Mode
================================
   Locates the distribution by various points
->   Use this when you want to show how an average or most commonly indicated response

3. Measures of Dispersion or Variation - Range, Variance, Standard Deviation
=======================================
   Identifies the spread of scores by stating intervals
   Range = High/Low points ,  size of the distribution of values
   Variance or Standard Deviation = difference between observed score and mean
   Kurtosis = This measures whether or not the tails of a given distribution 
   contain extreme values (also known as outliers). If a tail lacks outliers, we 
   can say that it has low kurtosis. If a dataset has a lot of outliers, we can say 
   it has high kurtosis.
   Skewness = This is a measure of a dataset’s symmetry. If you were to plot a bell-curve 
   and the right-hand tail was longer and fatter, we would call this positive skewness. 
   If the left-hand tail is longer and fatter, we call this negative skewness. 

->   Use this when you want to show how "spread out" the data are. It is helpful to 
   know when your data are so spread out that it affects the mean

4. Measures of Position - Percentile Ranks, Quartile Ranks
========================
   Describes how scores fall in relation to one another. Relies on standardized scores
->  Use this when you need to compare scores to a normalized score (e.g., a national norm)


Inferential Statistics
=======================
This may include making comparisons across time, comparing different groups, or trying to make predictions based on data 
that has been collected. 

1. t-tests : statistical test that can be used to compare means
=====================================================================
1.1 One-sample t-test :  used to compare your data to the mean of some known population.
-------------------------- 

->Thus, use a one-sample t-test when: You have one data set or one mean that you are interested in You know the mean of 
the population (the entire population, not a sample!) you wish to compare your mean to

1.2 Independent-samples t-test :  used to compare data from two separate, non-related samples.

->Thus, use an independent-samples t-test when:You have two separate, non-overlapping groups or data sets that you want to compare. 
That is, different people provided the data for each group.

1.3 Dependent samples t-test : used to compare data from related groups or the same people over time. 
--------------------------------
This is most often used when you have a pretest/posttest setup.


->Thus, use a dependent-samples t-test when:You have two separate data sets that are provided by the same people, 
just at different times (e.g. pre/post)

2. ANOVA (Analysis of Variance)
=========================================
a statistical test that is also used to compare means. The difference between a t-test and an ANOVA is that a t-test can only 
compare two means at a time, whereas with an ANOVA, you can compare multiple means at the same time. ANOVAs also allow you 
to compare the effects of different factors on the same measure. 


2.1 One-way ANOVA  : used to compare three or more groups/levels along the same dimension. It is similar to 
---------------------
an independent-samples t-test, just with more groups.

->Thus, use a one-way ANOVA when:You have three or more separate, non-overlapping groups or data sets that you want to compare.

2.2 Within-groups (Repeated measures) ANOVA : used to compare data from related groups or the same people over time.
---------------------------------------------
This is similar to a dependent-samples t-test, just with more data sets. This is most often used when you are doing a 
longitudinal study that tracks the same people across time.


->Thus, use a within-groups ANOVA when: You have separate data sets that are provided by the same people over time

2.3 Factorial ANOVA : used when you have two or more variables/factors/dimensions,and you want to explore whether there are 
------------------------
interactions between these factors.Essentially, you are comparing the means of the various combinations of factors.


->Thus, use a factorial ANOVA when: You are interested in the interaction between two or more variables/factors/dimensions
One thing that is important to note about ANOVAs is that because there are more than two groups that are being compared, 
follow-up (or post-hoc) tests are often required to further interpret the data. 


3. Regression : allows you to make a prediction about an outcome (or criterion) variable based on knowledge of some predictor variable.
==================       
Then you would determine the contribution of the predictor variable to the outcome variable.

->Thus, use regression when:You want to be able to make a prediction about an outcome given what you already know 
about some related factor.

4. Correlation
==================
5. Confidence intervals  : using different distribution to fit the data
=============================
6. Hypothesis testing : Null & alternate hypothesis form
========================== 
Footer
