# Chap 2 Fundamentals of Unconstrained Optimization
## 无约束优化问题
### 数学表达式：

$\min_x f(x), \\
x \in \mathbb R^n, n \geq 1, f: \mathbb R^n \rightarrow \mathbb R$  

### 举个例子——曲线拟合问题
假设包含参数$x$的关于$t$的函数：  
$\phi(t; x) = x_1 + x_2 e^{-(x_3-t)^2/x_4} + x_5\cos(x_6 t)$  
其中，参数$x_1, ..., x_6 \in \mathbf R$。假设目前已知如下点位于该函数曲线上：  
$(t_1, y_1), ..., (t_j, y_j), ..., (t_m, y_m)$  
  
为了拟合出这个函数，也就是要确定参数$x_1, ..., x_6$，我们可以定义每个点处的误差：

$r_j(x) = y_j - \phi (t_j; x)$，其中$x=(x_1, ..., x_6)^\mathrm T$  
当误差的绝对值之和达到最小时，我们就得到了拟合的结果。为了计算方便，我们可以为每个误差求平方后再求和，这样计算起来要比绝对值好处理得多，于是可定义如下的优化问题：
$\min_{x \in \mathbb R^6} f(x) = \sum_{i=1} ^ m r_i^2(x)$  
很明显这是个无约束优化问题，而且是一个非线性最小二乘问题。当已知的点数很少时，很容易求得这个优化问题的解，但如果点数较多，那就会比较麻烦了。

## 2.1 优化问题的解
### 全局解与局部解
最理想的情况无疑是全局解：  
如果对于$\forall x \in \mathbb R^n$，都有$f(x^*) \leq f(x)$，那么$x^*$就是优化问题的全局解。  
但这玩意相当难找，所以退而求其次，去找局部解：  
如果对于$\forall x \in N(x^*)$，都有$f(x^*)\leq f(x)$，那么$x^*$为$f(x)$的局部解。其中，$N(x^*)$表示$x^*$的邻域。  
如果再加强一步，加上限制条件$x \neq x^*$，并要求$f(x^*) < f(x)$，那么$x^*$就强化为严格局部解。

### 局部解的判别
首先引入几个工具：  
对于向量函数$f(x), x \in \mathbb R^n$，其梯度为  
$\nabla f(x) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}$

其海森阵为  
$\nabla^2f(x) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n}\\
\frac{\partial^2 f}{\partial x_2 x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}$

#### 泰勒公式：  
若$f: \mathbb R^n \rightarrow \mathbb R$连续可微，那么  
$f(x+p) = f(x) + \nabla f(x + tp)^\mathrm T p,\ t \in (0,1)$  

证明比较长，可以参考这个资料：
https://www.rose-hulman.edu/~bryan/lottamath/mtaylor.pdf

中值定理：  
$f(x+p) - f(x) = \int_0^1 \nabla f(x+tp)^T p \ dt$

如果$f$二阶连续可微，那么  
$\nabla f(x+p) = \nabla f(x) + \int_0^1 \nabla^2f(x+tp)p \ dt, \\
f(x+p) = f(x) + \nabla f(x)^Tp + \frac{1}{2}p^T \nabla^2f(x+tp)p, \\
t \in (0,1)
$  

#### 一阶必要条件
如果$x^*$为局部解，且$f$在$x^*$开邻域内连续可微，那么$\nabla f(x^*) = 0$  
如果$\nabla f(x^*) = 0$，则称$x^*$为驻点。由此可见，所有的局部解均为驻点。

#### 二阶必要条件
如果$x^*$为局部解，且$\nabla^2 f$存在并在$x^*$的开邻域内连续，那么$\nabla f(x^*) = 0$且$\nabla^2 f(x^*)$半正定。

正定：对于$\forall p \neq 0$，矩阵$B$满足$p^TBp > 0$，则为正定阵。

半正定：对于$\forall p$，矩阵$B$满足$p^TBp \geq 0$，则为半正定阵。

#### 二阶充分条件
如果$\nabla^2 f$在$x^*$开邻域内连续，且$\nabla f(x^*) = 0$，$\nabla^2f(x^*)$正定，那么$x^*$为严格局部解。

凸函数$f$的任意局部解均为其全局解，如果$f$是可微的，那么$f$的任意驻点均为其全局解。

## 2.2 算法概览
从初始值$x_0$出发，逐步迭代得到$\{x_1, \dots, x_k\}$。
### 线搜索方法
1. 找方向：$p_k$  
2. 定步长：$\alpha_k$
3. $x_{k+1} = x_k + \alpha_k p_k$

确定步长$\alpha$时嵌套的子优化问题：  
$\alpha_k = \argmin_{\alpha > 0} f(x_k + \alpha p_k)$  
但这个优化问题并不需要完全解决，否则会事倍功半。通常可以限制迭代的次数，达到一个可用的$\alpha_k$即可。

### 信赖域方法
在$x_k$点处，构建与目标函数$f$近似的模型函数$m_k$。往往$m_k$是一个二次函数，用于替代不易处理的原始函数$f$。  
由于$x$远离$x_k$时，$m_k$就不太能逼近$f$了，所以需要找出可以“信赖”的最小范围，这个范围就是信赖域。  
通常也需要通过求解一个子优化问题寻找步长$p$：  
$\min_{p} m_k(x_k + p)$，$x_k + p$位于信赖域中。  
如果这个优化问题不能收敛，那么说明信赖域选大了，需要缩小它。一般把信赖域定义成球：  
$|| p ||_2 \leq \Delta$，其中$\Delta > 0$即为信赖域半径。  
如上所述，$m_k$ 一般为二次函数：  
$m_k(x_k + p) = f_k + p^T \nabla f_k + \frac{1}{2}p^T B_k p$  
其中，
1. $f_k$为$f$在$x_k$点处的函数值（标量）
2. $\nabla f_k$为$f$在$x_k$点处的梯度（向量）
3. $B_k$为$f$在$x_k$处的海森阵或者其近似值（矩阵）

