ADF Statistic: -6.918528
p-value: 0.000000
Critical Values:
    1%: -3.483
    5%: -2.885
    10%: -2.579
Stationary ---> p <= 0.5 & 5% > ADF stats

ar_model = AutoReg(train, lags=2).fit()


ma_model = ARMA(train, order=(0,1)).fit()

AIC is an estimate of a constant plus the relative distance between the unknown true likelihood function of the data and the fitted likelihood function of the model, whereas BIC is an estimate of a function of the posterior probability of a model being true, under a certain Bayesian setup. 

BIC considers Type I and Type II errors to be about equally undesirable, while AIC considers Type II errors to be more undesirable than Type I errors unless n is very small. 

Number of obs – This is the number of observations that were used in the analysis.  This number may be smaller than the total number of observations in your data set if you have missing values for any of the variables used in the logistic regression.  Stata uses a listwise deletion by default, which means that if there is a missing value for any variable in the logistic regression, the entire case will be excluded from the analysis.

LR chi2(3)d      =   This is minus two (i.e., -2) times the difference between the starting and ending log likelihood.  The number in the parenthesis indicates the number of degrees of freedom.  In this model, there are three predictors, so there are three degrees of freedom.

Prob > chi2 – This is the probability of obtaining the chi-square statistic given that the null hypothesis is true.  In other words, this is the probability of obtaining this chi-square statistic (71.05) if there is in fact no effect of the independent variables, taken together, on the dependent variable.  This is, of course, the p-value, which is compared to a critical value, perhaps .05 or .01 to determine if the overall model is statistically significant.  In this case, the model is statistically significant because the p-value is less than .000.

Std. Err. – These are the standard errors associated with the coefficients.  The standard error is used for testing whether the parameter is significantly different from 0; by dividing the parameter estimate by the standard error you obtain a z-value (see the column with z-values and p-values). The standard errors can also be used to form a confidence interval for the parameter, as shown in the last two columns of this table.

------------------------------------------------------------------------------
     honcompg|     Coef.h  Std. Err.i     zj   P>|z|j    [95% Conf. Interval]k
-------------+----------------------------------------------------------------
      female |   1.482498   .4473993     3.31   0.001     .6056111    2.359384
        read |   .1035361   .0257662     4.02   0.000     .0530354    .1540369
     science |   .0947902   .0304537     3.11   0.002      .035102    .1544784
       _cons |   -12.7772    1.97586    -6.47   0.000    -16.64982   -8.904589
------------------------------------------------------------------------------

 z and P>|z| – These columns provide the z-value and 2-tailed p-value used in testing the null hypothesis that the coefficient (parameter) is 0.   If you use a 2-tailed test, then you would compare each p-value to your preselected value of alpha.  Coefficients having p-values less than alpha are statistically significant.  For example, if you chose alpha to be 0.05, coefficients having a p-value of 0.05 or less would be statistically significant (i.e., you can reject the null hypothesis and say that the coefficient is significantly different from 0).   If you use a 1-tailed test (i.e., you predict that the parameter will go in a particular direction), then you can divide the p-value by 2 before comparing it to your preselected alpha level.  With a 2-tailed test and alpha of 0.05, you may reject the null hypothesis that the coefficient for female is equal to 0.  The coefficient of 1.482498 is significantly greater than 0. The coefficient for read is .1035361 significantly different from 0 using alpha of 0.05 because its p-value is 0.000, which is smaller than 0.05. The coefficient for science is .0947902 significantly different from 0 using alpha of 0.05 because its p-value is 0.000, which is smaller than 0.05.


 Why the Math work? What it actually does(geometrically)? By doing that what can we achieve/avoid?  