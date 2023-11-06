It's a way to convert discrete variables into numbers.

## With mean values

Let's say we have the following data.

| Political Party   | Support Universal Basic Income (UBI) |
| ----------------- | ------------------------------------ |
| Democrat          | 1 (represents support)               |
| Republican        | 0 (represents not support)           |
| Green             | 1                                    |
| Republican        | 0                                    |
| Republican        | 0                                    |
| Republican        | 1                                    |
| Democrat          | 1                                    |
| Democrat          | 0                                    |

We want to predict whether the person supports UBI or not based on their supporting political party.

Let's say the algorithm we use expects numbers for the input.

We can convert the political parties into numbers using the target encoding with mean values.

---

There are 3 Democrats, and 2 of them support UBI.

So, we use 0.67 (=2/3) to represent Democrats.

Likewise, there are 4 Republican, and 1 of them support UBI.

So, we use 0.25 (=1/4) to represent Democrats.

Lastly, we have 1 Green party supporter, and the person supports UBI.

So, we use 1 (=1/1) to represent Green.

---

Thus, we can transform the original table like this:

| Political Party   | Support Universal Basic Income (UBI) |
| ----------------- | ------------------------------------ |
| 0.67              | 1 (represents support)               |
| 0.25              | 0 (represents not support)           |
| 1                 | 1                                    |
| 0.25              | 0                                    |
| 0.25              | 0                                    |
| 0.25              | 1                                    |
| 0.67              | 1                                    |
| 0.67              | 0                                    |

## With weighted mean values

However, the sample data shown in the `With mean values` section has an issue.

There is only one data for Green, which makes us less confident about Green value.

A better approach would be using weight mean.

Here is the formula for the weighted mean:

```math
\begin{align}
\text{weighted mean} = \frac{n \cdot \text{option mean} + m \cdot \text{overall mean}}{n + m} \text{, where} \\
\\
n = \text{weight for option mean (usually the number of rows)} \\
m = \text{weight for overall mean (user-defined)}
\end{align}
```

Since m is a hyperparameter, let's just say m = 2.

The overall mean is 0.5 (=4/8) because there are 8 people, and 4 of them support UBI.

n for Democrats is 3 because there are 3 Democrats in the sample data.

The option mean for Democrats is 0.67 (=2/3), as we got in the previous section.

So, let's calculate the weighted mean for all parties.

```math
\begin{align}
\text{weighted mean for Democrat} = \frac{3 \cdot 0.67 + 2 \cdot 0.5}{3 + 2} = 0.60 \\
\text{weighted mean for Republican} = \frac{4 \cdot 0.25 + 2 \cdot 0.5}{4 + 2} = 0.33 \\
\text{weighted mean for Green} = \frac{1 \cdot 1 + 2 \cdot 0.5}{1 + 2} = 0.67 \\
\end{align}
```

Thus, we can transform the original table like this:

| Political Party   | Support Universal Basic Income (UBI) |
| ----------------- | ------------------------------------ |
| 0.60              | 1 (represents support)               |
| 0.33              | 0 (represents not support)           |
| 0.67              | 1                                    |
| 0.33              | 0                                    |
| 0.33              | 0                                    |
| 0.33              | 1                                    |
| 0.60              | 1                                    |
| 0.60              | 0                                    |

By the way, we can think of the overall mean as our best guess when there is no data.

Some people call this approach `Bayesian Mean Encoding`.

## K-Fold Target Encoding

However, the above method has an issue.

We transformed the data using the target values, which led to data leakage.

To avoid that, we can use the k-fold target encoding.

K refers to the number of subsets we choose to create.

For example, let's use 2 for K.

We have two tables as follows.

Subset 1:
| Political Party   | Support Universal Basic Income (UBI) |
| ----------------- | ------------------------------------ |
| Democrat          | 1 (represents support)               |
| Republican        | 0 (represents not support)           |
| Green             | 1                                    |
| Republican        | 0                                    |

Subset 2:
| Political Party   | Support Universal Basic Income (UBI) |
| ----------------- | ------------------------------------ |
| Republican        | 0                                    |
| Republican        | 1                                    |
| Democrat          | 1                                    |
| Democrat          | 0                                    |

---

To convert the political parties into numbers in Subset 1, we have to use the target values in Subset 2.

Here is the calculation for Subset 1:
```math
\begin{align}
\text{weighted mean for Democrat} = \frac{2 \cdot \frac{1}{2} + 2 \cdot \frac{2}{4}}{2 + 2} = 0.5 \\
\text{weighted mean for Republican} = \frac{2 \cdot \frac{1}{2} + 2 \cdot \frac{2}{4}}{2 + 2} = 0.5 \\
\text{weighted mean for Green} = \frac{0 \cdot 0 + 2 \cdot \frac{2}{4}}{0 + 2} = 0.5 \\
\end{align}
```

Actually, all the numbers are coming from Subset 2 in this case.

We ignore everything in Subset 1 during the calculation.

Since there is no Green party supporter in Subset 2, n and the option mean are 0.

As a result, Subset 1 becomes like this:

| Political Party   | Support Universal Basic Income (UBI) |
| ----------------- | ------------------------------------ |
| 0.5               | 1 (represents support)               |
| 0.5               | 0 (represents not support)           |
| 0.5               | 1                                    |
| 0.5               | 0                                    |

---

Likewise, let's encode Subset 2.

Here is the calculation for Subset 2:
```math
\begin{align}
\text{weighted mean for Democrat} = \frac{1 \cdot \frac{1}{1} + 2 \cdot \frac{2}{4}}{1 + 2} = 0.67 \\
\text{weighted mean for Republican} = \frac{2 \cdot \frac{0}{2} + 2 \cdot \frac{2}{4}}{2 + 2} = 0.25 \\
\end{align}
```

Again, we actually use Subset 1 and ignore Subset 2 during the calculation.

As a result, Subset 2 becomes like this:

| Political Party   | Support Universal Basic Income (UBI) |
| ----------------- | ------------------------------------ |
| 0.25              | 1                                    |
| 0.25              | 0                                    |
| 0.67              | 1                                    |
| 0.67              | 0                                    |

---

Lastly, we combine the subsets like this:

| Political Party   | Support Universal Basic Income (UBI) |
| ----------------- | ------------------------------------ |
| 0.5               | 1 (represents support)               |
| 0.5               | 0 (represents not support)           |
| 0.5               | 1                                    |
| 0.5               | 0                                    |
| 0.25              | 1                                    |
| 0.25              | 0                                    |
| 0.67              | 1                                    |
| 0.67              | 0                                    |

---

By the way, we end up using different numbers to represent the same party.

That is okay because the political party becomes a continuous variable from a discrete one.

## Leave-One-Out Target Encoding

This approach is the same as `K-Fold Target Encoding`, except we make two subsets per calculation.

One subset has only one row, and another has all the rest.

To convert one row, we use the rest of the rows (excluding own) to calculate the number.
