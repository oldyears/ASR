**Solution 1:**

Let $V_{t}^{j}$ be the probability of the most probable path of the symbol sequence

$x_1,x_2,...,x_t$ ending in state j, then:
$$
V^{j}_{t+1}=b_{j}(x_{t+1})\underset{i}{max}(V^{i}_{t}a_{ij})
$$
For the matrix $V^j_t$ , where $j\in S$ and $1\le t\le n$.

Iteration:$V_t^j = b_j(x_t)\underset{i}{max}(V^i_{i-1}*a_{ij})$ for all states $i,j\in S,t\ge 2$.

So we have the initial table as following:

Initialization:

 $V^j_1=b_j(x_1)p(q_1=j)$, $V_1^j=\frac{b_j(A)}{state}$

$V_2^j=b_j(C)\underset{i}{max}(V^i_1a_{ij})$

$V_3^j=b_j(B)\underset{i}{max}(V^i_3a_{ij})$

From the title, we can get the information of status transition matrix and states generated matrix:

$A=\begin{pmatrix}0.2&0.3&0.5\\0.2&0.2&0.6\\0&0.2&0.8\end{pmatrix}$,$B=\begin{pmatrix}0.7&0.2&0.1\\0.3&0.4&0.3\\0&0.1&0.9\end{pmatrix}$.

|         |         A         |  C   |  B   |  A   |  C   |
| :-----: | :---------------: | :--: | :--: | :--: | :--: |
|  good   | $\frac{1}{3}*0.7$ |      |      |      |      |
| neutral | $\frac{1}{3}*0.3$ |      |      |      |      |
|   bad   |  $\frac{1}{3}*0$  |      |      |      |      |

and then, further we have

|         |  A   |       C        |  B   |  A   |  C   |
| :-----: | :--: | :------------: | :--: | :--: | :--: |
|  good   | 0.23 | $0.23*0.2*0.1$ |      |      |      |
| neutral | 0.1  | $0.3*0.23*0.3$ |      |      |      |
|   bad   |  0   | $0.9*0.23*0.5$ |      |      |      |

and then, further we have

|         |  A   |   C    |        B         |  A   |  C   |
| ------- | :--: | :----: | :--------------: | :--: | :--: |
| good    | 0.23 | 0.0046 | $0.2*0.0207*0.2$ |      |      |
| neutral | 0.1  | 0.0207 | $0.4*0.1035*0.2$ |      |      |
| bad     |  0   | 0.1035 | $0.1*0.1035*0.8$ |      |      |

and then, further we have

|         |  A   |   C    |    B     |         A         |  C   |
| :-----: | :--: | :----: | :------: | :---------------: | :--: |
|  good   | 0.23 | 0.0046 | 0.000828 | $0.7*0.00828*0.2$ |      |
| neutral | 0.1  | 0.0207 | 0.00828  | $0.3*0.00828*0.2$ |      |
|   bad   |  0   | 0.1035 | 0.00828  |         0         |      |

and then, further we have

|         |  A   |   C    |    B     |     A     |          C          |
| :-----: | :--: | :----: | :------: | :-------: | :-----------------: |
|  good   | 0.23 | 0.0046 | 0.000828 | 0.0011592 | $0.1*0.0011592*0.2$ |
| neutral | 0.1  | 0.0207 | 0.00828  | 0.0004968 | $0.3*0.0011592*0.3$ |
|   bad   |  0   | 0.1035 | 0.00828  |     0     | $0.9*0.0011592*0.5$ |

so the result table is following:

|         | A        | C          | B           | A             | C              |
| ------- | -------- | ---------- | ----------- | ------------- | -------------- |
| good    | **0.23** | 0.0046     | 0.000828    | **0.0011592** | 0.000023184    |
| neutral | 0.1      | 0.0207     | **0.00828** | 0.0004968     | 0.000104328    |
| bad     | 0        | **0.1035** | 0.00828     | 0             | **0.00052164** |

So most probable mood curve is following:

|    Day     | Monday | Tuesday | Wednesday | Thursday | Friday |
| :--------: | :----: | :-----: | :-------: | :------: | :----: |
| Assignment |   A    |    C    |     B     |    A     |   C    |
|    Mood    |  good  |   bad   |  neutral  |   good   |  bad   |





**Solution 2:**

For each data point x, we can introduce a latent variable $Y_i\in {1,2,...,m}$ denoting the component that point belongs to. For the E-step, we compute the posterior over the classes and have to normalize:
$$
V_j(x_i)=\frac{\pi_jf_L(x_i;\mu_j,\beta_j)}{\sum^m_{l=1}\pi_{l}f_L( x_l;\mu_l,\beta_l)}.
$$
In the M-step, we optimize:
$$
\sum^{n}_{i=1}\sum^{m}_{j=1}V_j(x_i)\log{P(x_i,y_i=j)}=\sum^{n}_{i=1}\sum^{m}_{j=1}V_j(x_i)\log{\pi_jf_L(x_i;\mu_j,\beta_j)}\\=\sum^{n}_{i=1}\sum^{m}_{j=1}V_j(x_i)(\log{\pi_j}+\log{\frac{1}{2\beta_j}e^{-\frac{1}{\beta_j}|x-\mu|}})\\
=\sum^{n}_{i=1}\sum^{m}_{j=1}V_j(x_i)(\log{\pi_j}-\frac{1}{\beta_j}|x_i-\mu_j|)+C
\tag{1}
$$
And then, we add a Lagrange multiplier $\lambda$ to make that $\sum^m_{j=1}\pi_j=1$ and get the Lagrangian:
$$
L(\pi,\mu,\lambda)=\sum^{n}_{i=1}\sum^{m}_{j=1}V_j(x_i)(\log{\pi_j}-\frac{1}{\beta_j}|x_i-\mu_j|)+\lambda(\sum^m_{j=1}\pi_j-1).
$$
Exactly as in the previous problem, by setting the gradient with respect to $\pi_j$ to zero, we get
$$
\frac{\partial}{\partial\pi_j}L(\pi,\mu,\lambda)=\frac{\sum^n_{i=1}V_j(x_i)}{\pi_j}+\lambda=0\\
\Longrightarrow \pi_j=\frac{\sum^n_{i=1}V_j(x_i)}{-\lambda}
$$
The multiplier is again equal to $\lambda=-n$. If we want yo maximize  (1) with respect to the variables $\mu_j$, we have to solve m separate optimization problems, one for each $\mu_j$, These m problems have the following form:
$$
\underset{\mu_j}{maximize}-\sum^n_{i=1}\frac{V_j(x_i)}{\beta_j}|x_i-\mu_j|
$$
These are one-dimensional convex optimization problems. While one can try soling this via an iterative process lick subgradient descent, a direct approach is also possible if we observe that function is piecewise linear and the breakpoints ara $x_1,x_2,...x_n$. Hence,the optimum must be attained at one of these n points and we can simply set $\mu_j$ to the point $x_i$ with the largest objective value.