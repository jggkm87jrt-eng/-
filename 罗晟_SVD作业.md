# 线性代数与统计学习作业

姓名：罗晟　学号：2026710209　班级：物理学QX261

## 1. 奇异值分解：推导与例子


### 1.1 需要解决什么问题

一般的数据矩阵不一定是方阵，不能直接对它做通常的特征分解。SVD 可以处理任意实矩阵：

$$
A_{m\times n}=U_{m\times m}\Sigma_{m\times n}V^{\mathsf T}_{n\times n}.
$$

$U,V$ 为正交矩阵，$\Sigma$ 的非零元素只在主对角线上，是按从大到小排列的非负奇异值。非零奇异值的个数等于 $A$ 的秩。

对输入向量，先作用 $V^{\mathsf T}$，再作用 $\Sigma$，最后作用 $U$。正交变换改变方向、保持长度，$\Sigma$ 沿不同方向缩放。正交变换也可能包含反射。

### 1.2 为什么要计算 A 的转置乘 A

$A^{\mathsf T}A$ 是对称半正定矩阵，因为

$$
x^{\mathsf T}A^{\mathsf T}Ax=\|Ax\|^2\ge0.
$$

因此可以找一组标准正交特征向量：

$$
A^{\mathsf T}Av_i=\lambda_i v_i,\qquad \lambda_i\ge0.
$$

对于 $\lambda_i>0$，令

$$
\sigma_i=\sqrt{\lambda_i},\qquad u_i=\frac{Av_i}{\sigma_i}.
$$

这样 $Av_i=\sigma_i u_i$，并且

$$
u_i^{\mathsf T}u_j
=\frac{v_i^{\mathsf T}A^{\mathsf T}Av_j}{\sigma_i\sigma_j}
=\frac{\lambda_jv_i^{\mathsf T}v_j}{\sigma_i\sigma_j}
=\delta_{ij}.
$$

也就是说，得到的左奇异向量彼此正交且长度为 1。如果奇异值是零，不能使用除法公式，需要在相应零空间补齐正交基。

### 1.3 自设计例子

取

$$
A=\begin{bmatrix}1&1\\1&0\\0&1\end{bmatrix}.
$$

选择这个矩阵是因为它是长方形、各行非零，但列内积简单，适合把分解过程完整手算出来。它不同于课件中已经给出的对角型例子。

先求列之间的内积：两列的平方长度都是 2，互相的内积是 1，因此

$$
A^{\mathsf T}A=\begin{bmatrix}2&1\\1&2\end{bmatrix}.
$$

特征方程为

$$
(2-\lambda)^2-1=(\lambda-3)(\lambda-1)=0.
$$

于是

$$
\lambda_1=3,\quad\lambda_2=1,
\qquad\sigma_1=\sqrt3,\quad\sigma_2=1.
$$

对 $\lambda_1=3$，特征向量两个分量相等；对 $\lambda_2=1$，两个分量互为相反数。归一化后：

$$
v_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},
\qquad v_2=\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix}.
$$

再计算左奇异向量：

$$
u_1=\frac{Av_1}{\sqrt3}
=\frac1{\sqrt6}\begin{bmatrix}2\\1\\1\end{bmatrix},
\qquad
u_2=Av_2=\frac1{\sqrt2}\begin{bmatrix}0\\1\\-1\end{bmatrix}.
$$

完整的 $U$ 需要三列，第三列取在 $\ker(A^{\mathsf T})$ 中。设 $u_3=(a,b,c)^{\mathsf T}$，则

$$
A^{\mathsf T}u_3=0\Rightarrow a+b=0,\ a+c=0.
$$

归一化后可取 $u_3=(-1,1,1)^{\mathsf T}/\sqrt3$。最终

$$
U=\begin{bmatrix}
2/\sqrt6&0&-1/\sqrt3\\
1/\sqrt6&1/\sqrt2&1/\sqrt3\\
1/\sqrt6&-1/\sqrt2&1/\sqrt3
\end{bmatrix},
\quad\Sigma=\begin{bmatrix}\sqrt3&0\\0&1\\0&0\end{bmatrix},
\quad V^{\mathsf T}=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}.
$$

尺寸为 $(3\times3)(3\times2)(2\times2)=3\times2$。

### 1.4 怎样验证

先检查 $U^{\mathsf T}U=I_3$、$V^{\mathsf T}V=I_2$。再将非零项展开：

$$
\sigma_1u_1v_1^{\mathsf T}
=\begin{bmatrix}1&1\\1/2&1/2\\1/2&1/2\end{bmatrix},
\qquad
\sigma_2u_2v_2^{\mathsf T}
=\begin{bmatrix}0&0\\1/2&-1/2\\-1/2&1/2\end{bmatrix}.
$$

相加正好等于 $A$。只保留第一项得到秩一近似，误差为

$$
\|A-A_1\|_F=1,
\qquad\frac{\sigma_1^2}{\sigma_1^2+\sigma_2^2}=\frac34.
$$

这里的 75% 指平方 Frobenius 范数能量。根据截断 SVD 的最优低秩近似性质，该秩一近似在 Frobenius 范数意义下最优。这个例子用于解释降维，不意味着这个小矩阵真的能节省存储。

### 1.5 程序对应关系

```python
A = np.array([[1., 1.], [1., 0.], [0., 1.]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
reconstructed = U @ np.diag(s) @ Vt
error = np.linalg.norm(A - reconstructed, "fro")
```

经济型分解省去了不参与重构的第三列，此时 $U$ 为 $3\times2$。本次计算得到奇异值约为 $(1.732051,1)$，重构误差约为 $6.80\times10^{-16}$。这是浮点精度下的正确重构。

不同实现可能把一对左右奇异向量同时变号，重构不变。手算通过 $A^{\mathsf T}A$ 理解原理，程序直接调用 SVD；这能避免主动形成 $A^{\mathsf T}A$ 带来的精度损失。

参考：教师提供的 SVD 课件；[NumPy SVD 文档](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)。

---

## 2. 贝叶斯推断：谷分辨项目中的建模案例


### 2.1 项目问题与案例边界

当前项目研究带隙石墨烯在激光驱动下的谷分辨激发与发射。一个相关问题是：两个谷的残余导带布居是否相同，有限观测能把这种不对称确定到什么程度？

本例为这个问题提出理想化统计观测方案。以下 70/30、65/35 都是人为设计的模拟计数，不来自论文或实际实验。没有经过实际数据验证的探测方案，不应当作为已完成的实验结果报告。

### 2.2 先把物理量定义清楚

对固定的材料和激光参数，记两个谷区域的残余导带布居积分为

$$
N_K=\int_{\Omega_K}n_c(\mathbf k,t_{\rm end})\,d^2\mathbf k,
\qquad
N_{K'}=\int_{\Omega_{K'}}n_c(\mathbf k,t_{\rm end})\,d^2\mathbf k.
$$

两个积分使用一致归一化，谷区域要预先规定。定义

$$
\theta=\frac{N_K}{N_K+N_{K'}},\qquad
P_v=\frac{N_K-N_{K'}}{N_K+N_{K'}}=2\theta-1.
$$

总布居必须非零。$\theta$ 是 $K$ 谷比例，$P_v$ 是本例的谷极化率，正值表示 $K$ 谷占优。这些量是本例的建模定义，不宣称与论文所有归一化细节完全一致。

### 2.3 从观测方式推导似然

假设能在固定驱动条件下，收集近似独立、谷标签可辨认的载流子事件；两个谷效率相同，暂不考虑背景和误分类。

对每个事件定义

$$
X_i=1\text{ 表示 }K\text{ 谷},\qquad
X_i=0\text{ 表示 }K'\text{ 谷}.
$$

假定 $X_i\mid\theta$ 独立同分布，服从参数为 $\theta$ 的伯努利分布。则 $n$ 个事件中 $K$ 谷计数 $k=\sum_iX_i$ 的似然为

$$
p(k\mid n,\theta)=\binom nk\theta^k(1-\theta)^{n-k}.
$$

这里先说明观测方式，再选择二项模型。数值计算中的动量网格点不是独立随机事件，不能把网格数当成这里的 $n$。发光或 HHG 强度也不是天然的谷标签载流子计数。

### 2.4 先验与后验

贝叶斯公式为

$$
p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)},
\qquad
p(D)=\int p(D\mid\theta)p(\theta)\,d\theta.
$$

先验说明观测前对参数的认识，似然说明给定参数时数据出现的概率，后验说明结合数据后对参数的认识。

采用 Beta 先验：

$$
p(\theta)=\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},
\qquad 0\le\theta\le1.
$$

本例设 $\alpha=\beta=2$，得到关于两个谷对称、略偏向中间概率的先验。这个选择便于展示共轭更新，不是由当前项目数据拟合得到的结论；它也不是均匀先验。

将先验和似然相乘，合并幂指数：

$$
p(\theta\mid D)
\propto\theta^{k+\alpha-1}(1-\theta)^{n-k+\beta-1}.
$$

因此

$$
\boxed{\theta\mid D\sim\operatorname{Beta}(\alpha+k,\beta+n-k).}
$$

这个推导说明，后验公式来自指数相加，而不是额外记忆的计算规则。

### 2.5 两批模拟数据的计算

第一批：$k=70,n=100$。有

$$
p(\theta\mid D_1)\propto\theta^{71}(1-\theta)^{31},
\qquad\theta\mid D_1\sim\operatorname{Beta}(72,32).
$$

注意 Beta 的两个参数比密度的两个指数各大 1。后验均值为

$$
\mathbb E[\theta\mid D_1]=\frac{72}{104}=0.692308,
\qquad\mathbb E[P_v\mid D_1]=0.384615.
$$

此时最大似然估计为 $70/100=0.7$，后验众数 MAP 为 $71/102\approx0.696078$。三者不同，是因为均值、众数和只使用数据的估计不是同一个概念。

第二批：再观察到 65 个 $K$、35 个 $K'$ 事件。若物理条件和未知参数保持相同，以上一次后验作为新先验：

$$
\theta\mid D_1,D_2\sim\operatorname{Beta}(72+65,32+35)
=\operatorname{Beta}(137,67).
$$

得到

$$
\mathbb E[\theta\mid D]=\frac{137}{204}=0.671569,
\qquad\mathbb E[P_v\mid D]=0.343137.
$$

也可以合并 135/65 的计数后一次更新，得到同样结果。如果改变激光相位、强度或材料参数，通常要分别建模，不能直接合并成同一个 $\theta$。

### 2.6 不确定性与预测

Beta 后验的方差为

$$
\operatorname{Var}(\theta\mid D)
=\frac{ab}{(a+b)^2(a+b+1)},\qquad a=137,b=67.
$$

数值计算的后验标准差约为 $0.032801$。使用后验分布的 2.5% 和 97.5% 分位数，得到 95% 等尾可信区间：

$$
\theta\in[0.605768,0.734183],\qquad
P_v\in[0.211535,0.468366].
$$

它表示在本例先验与观测模型成立时，参数位于区间内的后验概率为 95%。没有建模的仪器误差不包含在这个区间中。

下一次事件来自 $K$ 谷的预测概率为

$$
p(X_{\rm new}=1\mid D)
=\int\theta p(\theta\mid D)\,d\theta
=\frac{137}{204}.
$$

本例支持 $K$ 谷占优，但不能仅凭这个统计模型判定造成谷不对称的微观机制。

### 2.7 检查先验选择的影响

对同一批 70/30 计数，换三个对称先验，得到：

| 先验 | 第一批后验 | theta 后验均值 |
| --- | --- | --- |
| Beta(1,1) | Beta(71,31) | 0.696078 |
| Beta(2,2) | Beta(72,32) | 0.692308 |
| Beta(10,10) | Beta(80,40) | 0.666667 |

三个先验的均值都为 0.5，但 Beta(10,10) 更集中在 0.5 附近，因此把结果更明显地向 0.5 拉近。比较说明本例估计会受先验影响，应报告所用先验，而不能把选择隐藏起来。这组数值由相同的后验更新规则计算得到。

### 2.8 如何实现与如何用于真实项目

```python
from scipy.stats import beta
a, b = 2, 2
for nK, nKp in [(70, 30), (65, 35)]:
    a += nK
    b += nKp
mean_theta = a / (a + b)
ci_theta = beta.ppf([0.025, 0.975], a, b)
mean_Pv = 2 * mean_theta - 1
ci_Pv = 2 * ci_theta - 1
```

真实应用需要先校准标签和效率，并用独立测量检验预测。如果两谷效率分别为 $\eta_K,\eta_{K'}$，则观测到的比例变成

$$
q(\theta)=\frac{\eta_K\theta}{\eta_K\theta+\eta_{K'}(1-\theta)}.
$$

此时二项似然应使用 $q(\theta)$，一般不再是简单 Beta 共轭更新。如果只有发射谱，应把项目动力学计算得到的布居和相干通过发射模型连接到观测信号，再选择相应噪声似然。

参考：当前项目的物理和方程分析文档；[SciPy Beta 分布文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html)。

---

## 3. 交叉熵：定义、推导与计算


### 3.1 定义

对真实离散分布 $p$ 和预测分布 $q$，交叉熵定义为

$$
H(p,q)=-\sum_xp(x)\log q(x).
$$

它是在真实分布下，模型负对数概率的期望。预测给实际发生的事件分配越小的概率，惩罚越大。本文用自然对数，单位为 nat。若 $p(x)>0$ 而 $q(x)=0$，交叉熵为无穷大。

熵 $H(p)$ 和 KL 散度分别为

$$
H(p)=-\sum_xp(x)\log p(x),\qquad
D_{\rm KL}(p\|q)=\sum_xp(x)\log\frac{p(x)}{q(x)}.
$$

展开 KL 散度可得

$$
H(p,q)=H(p)+D_{\rm KL}(p\|q).
$$

固定 $p$ 时，$H(p)$ 是常数，因此最小化交叉熵等价于最小化 KL 散度。若模型能够表示 $p$，最小值在 $q=p$ 取得，此时交叉熵等于 $H(p)$，一般不为零。

### 3.2 从似然推导二元交叉熵

对二分类标签 $y_i\in\{0,1\}$，预测正类概率为 $q_i$，单个样本的概率为

$$
p(y_i\mid x_i)=q_i^{y_i}(1-q_i)^{1-y_i}.
$$

独立样本的概率相乘，取负对数后变成相加，再除以样本数：

$$
L=-\frac1n\sum_{i=1}^n
\left[y_i\log q_i+(1-y_i)\log(1-q_i)\right].
$$

因此二元交叉熵可以由伯努利标签的平均负对数似然得到。它常用于逻辑回归和二分类神经网络。

### 3.3 具体数值例子

仍用上一题第一批模拟计数：70 个标签为 1、30 个为 0。总体概率未知，此处采用经验分布 $\hat p=(0.7,0.3)$。若同条件下模型统一预测正类概率 $q=0.6$，则

$$
L=-\frac{70\log0.6+30\log0.4}{100}
=-0.7\log0.6-0.3\log0.4
=0.632465.
$$

经验熵为

$$
H(\hat p)=-0.7\log0.7-0.3\log0.3=0.610864.
$$

对应 KL 散度为

$$
0.7\log\frac{0.7}{0.6}+0.3\log\frac{0.3}{0.4}
=0.021601.
$$

可以直接核对 $0.632465\approx0.610864+0.021601$。

若把预测改为 $q=0.9$，损失变为 $0.764528$。虽然预测的多数类别仍然正确，但对少数类给出的概率过低，因而受到更大惩罚。

### 3.4 求导并完成一次更新

记经验正类比例为 $p=0.7$，则

$$
L(q)=-p\log q-(1-p)\log(1-q),
\qquad
\frac{dL}{dq}=\frac{q-p}{q(1-q)}.
$$

驻点是 $q=p$，而

$$
\frac{d^2L}{dq^2}=\frac p{q^2}+\frac{1-p}{(1-q)^2}>0,
$$

所以本例最优预测为 $0.7$。

实际训练常把参数 $z$ 通过 sigmoid 转为概率：

$$
q=\frac1{1+e^{-z}},\qquad \frac{dq}{dz}=q(1-q).
$$

应用链式法则：

$$
\frac{dL}{dz}=\frac{dL}{dq}\frac{dq}{dz}=q-p.
$$

从 $q_0=0.6$ 出发，先计算 $z_0=\log(0.6/0.4)=0.405465$。取演示步长 $\eta=1$：

$$
z_1=z_0-\eta(q_0-p)=0.405465-(-0.1)=0.505465.
$$

再转换回概率，得到 $q_1=0.623743$，新损失为 $0.623657$，低于原来的 $0.632465$。此处更新的是 logit，不是直接令概率增加 0.1。

对于 $z_i=w^{\mathsf T}x_i+b$，同样得到

$$
\nabla_w L=\frac1n\sum_i(q_i-y_i)x_i,
\qquad \frac{\partial L}{\partial b}=\frac1n\sum_i(q_i-y_i).
$$

### 3.5 多分类与数值实现

对互斥类别，Softmax 给出 $q_j=e^{z_j}/\sum_\ell e^{z_\ell}$。若 $y$ 是 one-hot 标签，

$$
L=-\sum_jy_j\log q_j
=-\sum_jy_jz_j+\log\sum_\ell e^{z_\ell}.
$$

因此 $\partial L/\partial z_j=q_j-y_j$。例如 $z=(\log2,0,0)$，则 $q=(0.5,0.25,0.25)$。真实类别为第一类时，损失为 $-\log0.5=0.693147$，梯度为 $(-0.5,0.25,0.25)$。

二分类损失可以稳定实现为

```python
z = np.log(0.6 / 0.4)
loss = np.logaddexp(0.0, z) - 0.7 * z
z_new = z - (0.6 - 0.7)
q_new = np.exp(-np.logaddexp(0.0, -z_new))
```

这里使用 $L(z)=\log(1+e^z)-pz$ 的等价形式，避免直接求极端概率的对数。多分类实现则先减去最大 logit，再计算 log-softmax。

### 3.6 与贝叶斯题的联系

第一批数据的总负对数似然，去掉常数后就是 100 倍上述交叉熵。加入 Beta(2,2) 先验后，MAP 需要最小化

$$
J(\theta)=-71\log\theta-31\log(1-\theta)+\text{常数}.
$$

令导数为零，得到 $\theta_{\rm MAP}=71/102$，与上一题一致。交叉熵提供似然对应的优化目标；完整贝叶斯推断还能给出参数分布和可信区间。

参考：[Stanford CS231n 的分类与交叉熵说明](https://cs231n.github.io/linear-classify/)。
