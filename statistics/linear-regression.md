It's about finding a linear line that gives the best representation of the dataset.

---

Let's say we have an equation of a regression line: y = a + bx,<br>
where y represents the house price,<br>
x represents the house size.

`R^2` tells us how much the house size explains the house price.

R^2 is a number between 0 and 1.

The closer to 1, the more the variable explains.

For example, if the R^2 is 0.6, we can say that the house size explains 60% of the house price.

---

We can have multiple variables like this: y = a + bx + cz,<br>
where z represents the age of the house (a new variable).

It's still a linear line in 3 dimensions.

The dimension goes up as we add more variables.

---

You may think that adding variables without any consideration never hurts the estimation because the coefficient would be 0 if the variable is completely unrelated.

However, it's still not a good idea to add variables without proper reasons because there is a random chance that the variable might show a correlation with the target by coincidence.

In other words, adding variables without proper reasoning would make the estimation optimistic (overfitted).

---

A high R^2 does not mean the estimation is good because the sample size might be too small.

To determine the significance of R^2, we also need to check the p-value.

---

A `residual` means the distance between the regression line and the actual data point.

We square each residual to make the number always positive.

The regression line minimizes the `sum of squared residuals`.

The method of finding the regression line is called `least squares`.
