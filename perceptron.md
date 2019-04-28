# 感知机
## 1.定义
由输入到输出空间的如下函数称为感知机:
$$f(x)=sign(w\cdot x+b)$$
其中线性方程$w\cdot x+b=0$对应特征空间$R^n$的一个超平面$S$,其中$w,b$分别为超平面的法向量与截距
## 2.学习算法
对于给定的的训练数据集：
$$T=\{( x_{1},y_{1}),( x_{2},y_{2}),...,( x_{N},y_{N}) \}$$
其中$x_{i}\in R^n,y_{i}\in \left \{-1,1 \right \},i=1,2,3,...,N$,求参数$w,b$,使其为以下损失函数极小化问题的解
$$\underset{w,b}{min} L(w,b)=-\sum_{x_{i}\in M}y_{i}(w\cdot x_{i}+b)$$
其中$M$为误分类点集合。
感知机学习算法是误分类驱动的，采用随机梯度下降法。首先选取一个超平面$w_{0},b_{0}$,然后采用梯度下降法不断极小化$L(w,b)$。极小化过程不是一次使$M$中所有误分类点梯度下降，而是一次随机选取一个误分类点使其梯度下降。
将损失函数分别对$w,b$求导得
$$\frac{\partial }{\partial w}L(w,b)=-\sum_{x_{i}\in M}y_{i}x_{i}$$
$$\frac{\partial }{\partial b}L(w,b)=-\sum_{x_{i}\in M}y_{i}$$
于是随机选取一个误分类点$(x_{i},y_{i})$,对$L(w,b)$进行更新：
$$w\leftarrow w+\eta y_{i}x_{i}$$
$$b\leftarrow b+\eta y_{i}$$
### 2.1 学习算法原始形式
输入：训练数据集 $T=\left \{ \left ( x_{1},y_{1} \right ),\left ( x_{2},y_{2} \right ),...,\left ( x_{N},y_{N} \right )  \right \}$,学习率 $\eta(0< \eta \leqslant 1)$;

输出：$L(w,b)$；感知机模型$f(x)=sign(w\cdot x+b)$

1. 选取初值$w_{0},b_{0}$
2. 在训练集中选取数据$(x_{i},y_{i})$
3. 如果$y_{i}(w\cdot x_{i}+b)\leqslant 0$
   $$w\leftarrow w+\eta y_{i}x_{i}$$
    $$b\leftarrow b+\eta y_{i}$$
4. 转至 2，直至训练集中没有误分类点

**当训练数据线性可分时，算法具有收敛性，即误分类次数有限；若数据线性不可分，则算法不收敛，迭代结果发生震荡。**
### 2.2 学习算法对偶形式
令 $a_{i}=n_{i}\eta$,则最终学习到的$(w,b)$可表示为
$$w=\sum_{i=1}^{N}a_{i}y_{i}x_{i}$$
$$b=\sum_{i=1}^{N}a_{i}y_{i}$$
$a_{i}\geqslant 0$,当$\eta=1$时，表示第$i$个实例点由于误分而被更新的次数。

输入：训练数据集 $T=\left \{ \left ( x_{1},y_{1} \right ),\left ( x_{2},y_{2} \right ),...,\left ( x_{N},y_{N} \right )  \right \}$,学习率 $\eta(0< \eta \leqslant 1)$;

输出：$(a,b)$；感知机模型$f(x)=sign(\sum_{j=1}^{N}a_{j}y_{j}x_{j}\cdot x+b)$,其中$a=(a_{1},a_{2},...,a_{N})^T$

1. 初始化 $a=0,b=0$
2. 在训练集中选取数据$(x_{i},y_{i})$
3. 如果$y_{i}(\sum_{j=1}^{N}a_{j}y_{j}x_{j}\cdot x_{i}+b)\leqslant 0$
   $$a_{i}\leftarrow a_{i}+\eta$$
   $$b\leftarrow b+\eta y_{i}$$
4. 转至 2，直至训练集中没有误分类点

对偶形式中数据仅以内积形式出现，为方便，可以预先将训练集中数据的内积形式计算出来并以矩阵方式存储，即 $Gram$矩阵
$$G=[x_{i}\cdot x_{j}]_{N\times N}$$