# Chap3 Line Search Methods 线搜索方法
## 3.0 算法框架
从初始$x_0$点开始，逐步迭代，得到序列$\{x_1, \dots, x_k\}$  
引入简写：$\nabla f(x_k) \rightarrow \nabla f_k, \nabla^2f(x_k) \rightarrow \nabla^2 f_k$
### 下降方向
对应原书第2.2节中线搜索方向的前半部分，Page21~22  
$p^T \nabla f_k < 0$
### 线搜索方法框架
1. 选方向$p_k$：  
    要求为下降方向。
    普遍形式：$p_k = -B_k^{-1} \nabla f_k$  
    根据下降方向条件，$p^T \nabla f_k = -\nabla f_k^T B_k^{-1} \nabla f_k < 0$，故$B_k$要求为正定阵。   
    (1) 最速下降法：$p_k = -\nabla f_k $，即负梯度方向，$B_k = I$  
    (2) Newton method：$p_k = -(\nabla^2 f_k)^{-1} \nabla f_k$，牛顿方向，$B_k = \nabla^2 f_k$  
    (3) Quasi-Newton method  
    (4) 共轭梯度法
2. 定步长$\alpha_k$：  
    $\alpha_k = \argmin_{\alpha>0} f(x_k + \alpha p_k)$，无需完全解决，只需得到满足Wolfe条件的$\alpha_k$即可
3. 迭代：  
    $x_{k+1} = x_k + \alpha_k p_k$

## 3.1 步长
令$\phi(\alpha) = f(x_k + \alpha p_k), \alpha > 0$  
那么$\alpha = \argmin_{\alpha > 0} \phi (\alpha)$  
两个要求：
1. 确保$f$充分下降
2. 确保速度，$\alpha$不能过小

### Wolfe 条件
#### 上限：一维搜索停止条件(Armijo condition)
$\phi(\alpha) = f(x_k + \alpha p_k) \leq f(x_k) + c_1 \alpha p^T \nabla f_k, c_1 \in (0, 1)$  
  
令：  
$\begin{aligned}
l(\alpha) &= f(x_k) + c_1 \alpha p^T \nabla f_k \\
&=c_1 p^T \nabla f_k \alpha + f(x_k)
\end{aligned}$  
是关于$\alpha$的线性函数，且必过点$(0, f(x_k))$  
而$\phi(\alpha)$同样必过点$(0, f(x_k))$，且$\phi'(\alpha) = p_k^T \nabla f(x_k + \alpha p_k)$  

进一步，由于$c_1 \in (0,1), p^T \nabla f_k < 0$，  
所以$0 > l'(\alpha) = c_1 p^T \nabla f_k > p^T \nabla f_k = \phi'(0)$，$l(\alpha)$的下降比$\phi(\alpha)$要慢  
所以一定存在区间$(0, T)$，使得$\phi(\alpha) < l(\alpha)$，从而可以确保$f(x_k + \alpha p) < f(x_k)$

一般在实践中，取$c_1 = 10^{-4}$

上限可以确保充分下降，但$\alpha$如果取得非常小，能保证充分下降不假，但速度可能比蜗牛还慢。

#### 下限：曲率条件(curvature condition)  
$\phi'(\alpha) = p_k^T \nabla f(x_k + \alpha p_k) \geq c_2 p^T \nabla f_k, c_2 \in (c_1, 1)$

在Newton method和Quasi-Newton method中，$c_2 = 0.9$  
在非线性共轭梯度法中，$c_2 = 0.1$  

将上限和下限综合在一起，就形成了完整的Wolfe条件：  
$\begin{aligned}
f(x_k + \alpha_k p_k) &\leq f(x_k) + c_1 \alpha_k \nabla f_k^T p_k \\
\nabla f(x_k + \alpha_k p_k) ^T p_k &\geq c_2 \nabla f_k^T p_k\\
0 < c_1 &< c_2 < 1
\end{aligned}$

强Wolfe条件：进一步限制$\phi'(\alpha)$的绝对值。  
$\begin{aligned}
f(x_k + \alpha_k p_k) &\leq f(x_k) + c_1 \alpha_k \nabla f_k^T p_k \\
| \nabla f(x_k + \alpha_k p_k) ^T p_k | &\leq c_2 | \nabla f_k^T p_k |\\
0 < c_1 &< c_2 < 1
\end{aligned}$

#### 引理3.1：满足Wolfe条件和强Wolfe条件的步长一定存在

## 3.2 收敛性
$\begin{aligned}
    \cos \theta_k = \frac{-\nabla f_k^T p_k}{\|\nabla f_k\| \| p_k \|}
\end{aligned}$  
$\theta_k$为搜索方向$p_k$与负梯度方向之间的夹角。  

#### 定理3.2 Zoutendijk条件
对于一切形如$x_{k+1} = x_k + \alpha_k + p_k$的迭代，其中$p_k$为下降方向，$\alpha_k$满足Wolfe条件，若$f$有下界，在包含水平集$\mathcal{L} = \{x | f(x) \leq f(x_0)\}$的开区间$\mathcal{N}$上连续可微，其中$x_0$为迭代起点，且$\nabla f$满足Lipschitz连续：$\exist L > 0, S.t. \| \nabla f(x) - \nabla f(\widetilde{x}) \| \leq L \| x - \widetilde{x} \|, x, \widetilde{x} \in \mathcal{N}$，那么$\sum_{k \geq 0} \cos ^2 \theta_k \| \nabla f_k \|^2 < \infty$

说明：
1. 如果没有下界，那么优化问题就失去了意义
2. $\nabla f(x)$满足Lipschitz连续，确保局部收敛（Chap 6 & Chap 7）

由$\sum_{k \geq 0} \cos ^2 \theta_k \| \nabla f_k \|^2 < \infty$，可知

$\begin{aligned}
    \cos^2\theta_k \| \nabla f_k \|^2 \rightarrow 0
\end{aligned}$  

所以为了确保这个结果成立，
1. 要么$\theta_k \rightarrow \pi / 2$，使得$\cos \theta_k \rightarrow 0$
2. 要么$\| \nabla f_k \| \rightarrow 0$

而我们选择的几个搜索方向，与$-\nabla f_k$之间的夹角$\theta_k$都是小于$\pi /2$的，所以：
$\lim _{k \rightarrow \infty} \| \nabla f_k \| = 0$  
这是确保全局收敛的条件。确保线搜索的方向不与梯度的正交方向接近。
