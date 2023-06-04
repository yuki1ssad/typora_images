# 笔记

## 20230210

### 1_CLIP the Gap: A Single Domain Generalization Approach for Object Detection_2023_暂无代码

> 作者：[Vidit Vidit](https://arxiv.org/search/cs?searchtype=author&query=Vidit%2C+V), [Martin Engilberge](https://arxiv.org/search/cs?searchtype=author&query=Engilberge%2C+M), [Mathieu Salzmann](https://arxiv.org/search/cs?searchtype=author&query=Salzmann%2C+M)

> 贡献：

单域泛化(Single Domain Generalization，SDG)致力于在单源域上训练模型，使其泛化到其他未见过的目标域。这在图像分类中已经得到了很好的研究，但关于SDG目标检测的研究很少。为将单域泛化应用到目标检测钟，同时学习鲁棒的物体定位和表示，本文提出利用预训练的视觉-语言模型CLIP通过文本提示引入语义领域概念。通过对检测器主干提取的**特征进行语义增强**，以及**基于文本的分类损失**来实现这一点。

 Domain adaptation：尝试学习源域（训练）与目标域（测试）之间的域不变特征，但这需要知道目标域的信息，实际中可能无法获得目标域的数据；

 Domain generalization：尝试学习可以泛化到目标域的表示

> **方法：**

![image-20230201190220157](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230201190220157.png)

**1、对特征做语义增强**

在SDG问题中，由于训练时只能访问单一领域内的图像，因此需要学习具有域迁移鲁棒性的物体表示。

$T和V$分别表示CLIP的文本编码器和图片编码器，将$V$拆分成用于特征提取的$V^a$和将特征映射到嵌入空间的投射器$V^b$，对于图片$I$和对应的提示$p$，CLIP的作用可以表示为最小化$V^b(V^a(I))$和$T(p)$之间距离。

本文为源域定义一般性文本提示$p^s$，如 An image taken during the day；为可能的目标域定义一个文本提示集$P^t=\{p_j^t\}_1^M$；

*M* = 15，an image taken on a *{*weather*} {*time of the day*}*

 *{*weather：*snow, fog, cloudy, rain, stormy*}       {*time of the day*：*day, night, evening*}

**目的：**找到作用于特征层面的**增强$\{A_j\}$**，使得该增强所引起的shift刚好对应于$p^s$和$p^t_j$之间的语义差距。

首先，对提示文本分别做encode得到embedding $q^s=T(p^s),  q_j^t=T(p_j^t)$

对图片进行随机裁剪，对于每张裁剪得到的图片$I_{crop}$，其image embedding为 $z=V(I_{crop})$;

创建目标image embedding： $z_j^*=z+\frac{q_j^t-q^s}{||q_j^t-q^s||_2}$;

期望找到$A_j \in R^{H*W*C}$使得$\overline{z_j}=V^b(V^a(I_{crop})+A_j)$尽量与$z_j^*$相似，（使用cosine相似度衡量）。

约束函数为$L_{opt}=\sum_{I_{crop}}\sum_jD(z_j^*,\overline{z_j})+||\overline{z_j}-z||_1$，其中$D(a,b)=1-\frac{a-b}{||a-b||_2}$表示余弦距离。$l_1$正则化用于防止embedding与原始图片差别太大，保留部分人你图像内容。

注：增强的优化只在离线阶段做一次，然后使用生成的增强来训练检测器。

**2、整体架构**

如上图（右）

对于RPN提出的候选区域$r$，计算经过$V_b$得到的$F_r$（$F_r\in D_{clip}$）与经过$T$得到的$Q$（$Q\in R^{K+1}*D_{clip}$，K个类别+背景类）之间的余弦相似性$sim(F_r,Q_k)$，作为基于softmax的交叉熵损失的logits。

$$
L_{clip-t}=\sum_rL_{CE}(\frac{e^{sim(F_r,Q_k)}}{\sum_{k=0}^{K}e^{sim(F_r,Q_k)}})
$$

训练损失：$L_{det} = L_{rpn}+ L_{reg} + L_{clip-t} $

### 2_DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection_ICLR 2023_有代码

**ICLR**：International Conference on Learning Representations

> 作者：[Hao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Feng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+F), [Shilong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Lei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L), [Hang Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+H), [Jun Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J), [Lionel M. Ni](https://arxiv.org/search/cs?searchtype=author&query=Ni%2C+L+M), [Heung-Yeung Shum](https://arxiv.org/search/cs?searchtype=author&query=Shum%2C+H)

> 代码：https://github.com/IDEA-Research/DINO

> 贡献：

本文提出端到端目标检测算法DINO (**D**ETR with **I**mproved de**N**oising anch**O**rboxes)实现SOTA性能，这是基于Transformer的端到端检测器**首次**在COCO排行榜上超越SOTA。主要使用了一下方式：

- **使用一种对比的方式进行去噪训练**：为了改进一对一的匹配效果，本文提出了一种对比去噪训练方法，通过同时添加正、负样本来进行对比去噪训练。对比去噪训练可以帮助模型避免同一目标的重复输出。
- **对于anchor初始化，使用一种混合的query选择方法**：从编码器的输出中选择初始锚定框作为位置查询(positional queries)；然而，本文让内容查询(content queries)像以前一样可以学习，从而鼓励解码器的第一层专注于空间先验。
- **对于box预测，使用两次forward**：用后层的梯度纠正相邻前层已更新的参数。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230207154111848.png" alt="image-20230207154111848" style="zoom: 67%;" />

**Contrastive DeNoising Training**

DN-DETR在稳定训练和加速收敛上非常有效，在DN query的帮助下，在有 GT框的附近锚框的基础上学着去做预测。然而它对”没有物体在附近的锚框“缺乏预测没有物体的能力。为了解决这个问题，本文提出了对比去噪 (CDN)方法来拒绝无用的锚框。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230207114231858.png" alt="image-20230207114231858" style="zoom: 50%;" align="left"/>小方块内的**positive queries**的噪声尺度小于λ1，并被期望重建其相应的GT框；

大、小方块之间的**negative queries**的噪声尺度大于λ1，小于λ2。他们被期望用于预测“没有目标”。

（λ2设置的比较小，因为靠近GT的hard negative samples能更好地提高模型性能。）

**Mixed Query Selection**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230207133232931.png" alt="image-20230207133232931" style="zoom: 67%;" />

如上图(a)，DETR和DN-DETR中解码器查询是没有用来自图片编码特征的**静态嵌入**；

图(b)，Deformable DETR有一个查询选择变量(叫做‘两阶段’)，它从编码器最后一层层中选择top-K个特征来增强解码器查询，位置和内容查询都是由被选中的特征的线性变换生成的；

**图(c)**本文，仅使用与所选top-K特征关联的**位置信息**初始化anchor boxes，但保持内容查询与以前一样采用静态的。

**Look Forward Twice**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230207151744280.png" alt="image-20230207151744280" style="zoom: 67%;" />

向前看一次：如图六(a)，Deformable DETR中的迭代框refinement阻止梯度反向传播以稳定训练，$i$ 层的参数仅根据盒子$b_i$的附加损失进行更新的;

本文推测来自后一层的改进$box$信息可能更有助于纠正其相邻前层的$box$预测。因此本文提出了另一种叫做向前看两次的方法来执行$box$更新，其中第 $i$ 层的参数是受第$ i$ 层和第$ i+1$ 层损失的影响，如图六(b)，对于每个预测的偏置$∆b_i$， 它将用于更新$box$两次，一次是$b_i'$，另一次是$b_{i + 1}^{p r e d}$ 。

向前看一次方法只优化$∆b_i$，因为梯度信息从$layer_i$ 到 $layer_{i-1}$是分离的；

向前看两次优化$b_{i-1}$和$∆b_i$。一个简单的方法来改进质量是用下一层$∆b_{i+1}$的输出监督 $i$ 层的最终盒子$b_i'$ ，因此本文用$b_i'$ 和$∆b_{i+1}$的和作为$i+1$层的预测盒子。

具体的：对于第$i$层给定输入$box$ $b_{i-1}$，最终的预测$box$ $b_i^{(pred)}$由以下式子获得

$$
\Delta b_i=Layer_{i}(b_{i-1}) \\
b_{i}'=Update(b_{i-1},\Delta b_i) \\
b_{i}=Detach(b_{i}') \\
b_i^{(pred)}=Update(b_{i-1}',\Delta b_i)  \tag{2}
$$

其中$b_i'$是$b_i$未分离的版本。

## 20230217

### 3_Sparse R-CNN:End-to-End Object Detection with Learnable Proposals_CVPR_2021_有代码

**CVPR**：IEEE Conference on Computer Vision and Pattern Recognition

> 作者：[Peize Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+P), [Rufeng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Yi Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+Y), [Tao Kong](https://arxiv.org/search/cs?searchtype=author&query=Kong%2C+T), [Chenfeng Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+C), [Wei Zhan](https://arxiv.org/search/cs?searchtype=author&query=Zhan%2C+W), [Masayoshi Tomizuka](https://arxiv.org/search/cs?searchtype=author&query=Tomizuka%2C+M), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Zehuan Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+Z), [Changhu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Ping Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+P)

> 代码：https://github.com/PeizeSun/SparseR-CNN

> 贡献：

借鉴DETR中使用二部图匹配的方法将目标检测作为集合预测，提出纯粹稀疏的R-CNN目标检测范式，在COCO数据集上，检测精度、训练时间以及训练收敛性能都与良好的检测器基线相当。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230212164901718.png" alt="image-20230212164901718" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230212160811505.png" alt="image-20230212160811505" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230212161547863.png" alt="image-20230212161547863" style="zoom: 75%;" align="left" />数据输入包括an image, a set of proposal boxes and proposal features 使用FPN作为Backbone，处理图像；

**Learnable porposal box:** (*N* *×*  *4*) 可以看成是物体潜在位置的统计概率；

**Learnable proposal feature:** (*N* *×* *d*) ，proposal box 用一个比较简洁的方式来描述物体，但缺少了很多信息，比如物体的形状与姿态，proposal feature就是用来表示更多的物体信息；

**Dynamic instance interactive head：**通过 proposal boxes 以及ROI方法获取每个物体的特征，将roi特征和Proposal Features进行实例级的可交互计算，从而突出对前景贡献最大的bin(7x7个)的输出值，从而最终影响物体的位置和分类预测，如果确实是背景，则相当于7x7个bin都没有高输出值

Head的数量与learnable box的数量相同，即 head — learnable proposal box — learnable proposal feature  一一对应

**级联refine：**为了进一步提高性能，作者还提出了 cascade rcnn 类似的 refine 回归思想，就是迭代运行 n 个 stage，每个 stage 都是一个 rcnn 模块，参数不共享，下一个 stage 接收上一个stage输出的 refine 后的 roi 作为输入。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230212164549935.png" alt="image-20230212164549935" style="zoom:80%;" align="left"/>   

  

fature reuse：在级联多个 rcnn head 时候，重用前层的 Proposal Features 特征，能带来不错的效果提升。





### 4_DN-DETR: Accelerate DETR Training by Introducing Query DeNoising_CVPR2022_有代码

> 作者：[Feng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+F), [Hao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Shilong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Jian Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+J), [Lionel M. Ni](https://arxiv.org/search/cs?searchtype=author&query=Ni%2C+L+M), [Lei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L)

> 代码：https://github.com/IDEA-Research/DN-DETR

> 贡献：

1、DN-DETR通过在**训练过程中**加入去噪（DeNoising）任务，在 DAB-DETR 的基础上进一步加速了模型的收敛速度。2、指出由于匈牙利匹配的离散性和模型训练的随机性，导致了 query 对 gt 的匹配变成了一个动态的、不稳定的过程，从而导致收敛慢。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230212155226675.png" alt="image-20230212155226675" style="zoom:80%;" />

> 框架：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230212143228630.png" alt="image-20230212143228630" style="zoom:80%;" />

 **DeNoising**

```tex
...our method additionally feeds GT bounding boxes with noises into the Transformer decoder and trains
the model to reconstruct the original boxes, which effectively reduces the bipartite graph matching difficulty and leads to faster convergence
```

DN要做的就是对 gt 加噪后输入decoder，让输出去重构原来的 gt。

Conditional DETR将每个decoder query解耦为content & position 两部分

**noised boxes:** 对 gt boxes 加噪，即对 query 的 position 部分加噪。DN-DETR 按照 DAB-DETR的设定，position 部分就是 4d anchor box，要做的就是对这4个分量都加上细微的“扰动”，可以概括为中心点位移 & 尺度缩放。

中心点位移：对中心点$x,y$添加一个随机的扰动$(\Delta x, \Delta y)$，控制$\Delta x < \frac {\lambda_1 w}{2}, \Delta y < \frac {\lambda_1 h}{2}, \lambda_1 \in (0,1)$使得加噪后的中心点还在原框内；

尺度缩放：缩放后的$w,h$从$[(1-\lambda_2)w, (1+\lambda_2)w]  ,  [(1-\lambda_2)h, (1+\lambda_2)h]$中分别随机采样得到。

**noised labes:** 对 gt labels 加噪，即对 query 的 content 部分加噪。由于 gt label 是一个数字，参考 query 的 position 部分把加噪的 label 编码为 embedding 向量。于是，在模型中设置一个 embedding matrix，由其来对加噪的 gt label 进行编码得到对应的 class embedding，另外，作者还在 class embedding 部分拼接(concat)了指示向量 indicator，用以区分 query 到底是做去噪任务还是匹配任务。对应地，做匈牙利匹配任务的那部分 query 的 content 部分也需要改造下，让它的值初始化为 'non-object'，这个值应当不小于类别数 num_classes，因为做去噪任务的 query 的 content 部分是由真实的 gt label 而来，其值域会是 $[0, num\_classes - 1]$，这个 non-object class 也通过 embedding matrix 去编码，从而得到对应的 embedding 向量。（如下图为DAB-DETR与DN-DETR的decoder query对比）

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230212140536449.png" alt="image-20230212140536449" style="zoom:80%;" />

**dn groups:** 将 one-to-many 范式引入到 DETR 训练中

为了更充分地利用 DN 任务去提升模型的学习效率，可以让模型对于每个 gt 在不同程度的噪声下都拥有“纠错”能力。作者设置了 dn groups，即多个去噪组，每个 gt 在每组都会由一个噪声 query(noised label & noised box) 负责去预测。在每组内，gt -> query 依然是 one-to-one 的关系；但综合所有组来看，gt -> query 就是 one-to-many 的关系了。

**attention mask：**防止信息泄露

mask准则：匹配任务的 queries 不能看到 DN任务的 queries && DN任务中，不同组的 queries 不能相互看到

对于后者，不同 dn group 的 queries 也不能相互看到。因为综合所有组来看，gt -> query 是 one-to-many 的，每个 gt 在每组都会有1个 query 拥有自己的信息。于是，对于每个 query 来说，在其它各组中都势必存在1个 query 拥有自己负责预测的那个 gt 的信息。

attention mask $A = [a_{ij}]_{W*W}, W=P*M+N$，其中$P$为 group 数、$M$为GT objects数、$N$为匹配任务中的query数。$a_{ij}=1$表示第$i$个 query 不能看见第$j$个query，否则能看见。

$$
a_{ij}=
\left\{\begin{matrix}
1,\quad if \quad j < P*M and \lfloor i/M \rfloor \neq \lfloor j/M \rfloor \\
1,\quad if \quad j < P*M and i \geq P*M; \\
0, \quad otherwise
\end{matrix}\right.
$$

## 20230224

### 5_Improved Visual-Semantic Alignment for Zero-Shot Object Detection_AAAI2020_有代码

**AAAI**：Association for the Advancement of Artificial Intelligence

> 作者：[Ankan Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+A), [Karan Sikka](https://arxiv.org/search/cs?searchtype=author&query=Sikka%2C+K), [Gaurav Sharma](https://arxiv.org/search/cs?searchtype=author&query=Sharma%2C+G), [Rama Chellappa](https://arxiv.org/search/cs?searchtype=author&query=Chellappa%2C+R), [Ajay Divakaran](https://arxiv.org/search/cs?searchtype=author&query=Divakaran%2C+A)

> 代码：https://github.com/salman-h-khan/PL-ZSD_Release

> 贡献：

针对 zero-shot 目前存在的正负样本失衡、视觉和语义概念对齐以及未知类与背景之间的模糊性问题，本文提出了一个端到端的深度学习框架，通过基于 Focal Loss 的极性损失 (**Polarity loss**) 来解决类不平衡问题并更好地对齐视觉和语义概念。其中极性损失通过最大化 positive prediction 和 negative predication 之间的 gap，不仅有助于对齐视觉—语义概念，也能解决背景和未知类模糊的问题。此外，对象的语义表示是有噪声的，因此使视觉域和语义域之间的对齐被复杂化。为此，本文使用相关概念的“语义词汇表 (*Semantic vocabulary*) ”来进行度量学习（**Vocabulary metric learning**），该词汇表改进了嘈杂的语义嵌入，并在视觉和语义域之间建立了更好的协同作用。

> **Polarity Loss**

在 zero-shot 学习中，将视觉特征与语义词向量对齐是非常重要的。这种对齐需要训练过程有以下特点：(1)将视觉特征推到接近 ground truth 嵌入向量，(2)将它们远离所有的负类向量；Focal Loss(FL) 只能做到 (1)。$multi-class\quad loss$  FL：$L=\sum_i-\alpha _t^i(1-p_t^i)^\gamma logp_t^i$

因此本文在 FL 的基础上增加了一项单调的惩罚函数，得到 Polarity Loss (PL)：

$$
L_{PL}=\sum_{i}f_p(p^i-p^l)FL(p^i,y^i)
$$

$$
f_p(p^i-p^l)=\frac{1}{1+exp(-\beta (p^i-p^l))}
$$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230218091425811.png" alt="image-20230218091425811" style="zoom: 67%;" align="left"/> 

其中 $i$为类别$C$中的某一类；$y = \{y^i\in \{0,1\}\}\in C$表示 ground-truth label，背景类由 **y=0** $\in R^C$表示；$p=\{p^i\in [0,1]\}\in R^C$表示prediction vector。$p^l$为对应于ground-truth label $y^l=1$的预测；$\beta$为超参数（20）

若$l\neq i$，则$p^i-p^l$表示positive prediction和negative prediction之间的 gap 。如下图，当$p^i-p^l$较大时，乘法函数值比较大。



**Vocabulary metric learning**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230222145520464.png" alt="image-20230222145520464" style="zoom:80%;" align="left"/> 

如左图，视觉特征 **f** $\in R^d$（对应图中$x$)。

- **传统方法**使用一个可学习的 FC 层$W_d \in R^{S*d}$（对应图中$W_1$），再使用一个激活函数$\sigma$，最终预测为$p_d=\sigma (W_df)$。但这种方法因为无法处理unseen 类，只适合于传统目标检测。

- **Learning with Word-vectors：**假定给一个不可学习的词向量$W_S\in R^{S*d}$（如Word2Net），其中$S$是可见类的数目，将该层冻结，预测分数则为$p_d=\sigma (W_sf)$，该投影将视觉特征与相应真实类的词向量对齐。通过更改词向量则可以检测任意unseen类物体。

  

- **Learning with vocabulary metric**：

词嵌入空间通常是使用来自未注释文本的数十亿个词来学习的，这会导致噪声词嵌入。 因此，仅使用 S 个词向量来理解语义空间是不稳定的，并且不足以对视觉语义关系进行建模。因此，本文呢使用了一个额外的 pre-defined vocabulary $D \in R^{v*d}$。则预测值可以表示为$p_d=\sigma ((W_sMD)f)$，其中$M\in R^{d*v}$是一个可学习的参数，用于连接 seen word vec 与  pre-defined vocabulary。

如下图实现方式即在目标检测的分类层加入此对齐方法：

![image-20230222145424958](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230222145424958.png)

### 6_Learning Open-World Object Proposals without Learning to Classify_IEEE Robotics and Automation Letters 2022_有代码

> 作者：[Dahun Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+D), [Tsung-Yi Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+T), [Anelia Angelova](https://arxiv.org/search/cs?searchtype=author&query=Angelova%2C+A), [In So Kweon](https://arxiv.org/search/cs?searchtype=author&query=Kweon%2C+I+S), [Weicheng Kuo](https://arxiv.org/search/cs?searchtype=author&query=Kuo%2C+W)

> 代码：https://github.com/mcahny/object_localization_network

> 贡献：

一般用于学习 object proposals 的方式为从带有定位标签以及带有相应类别信息的数据中学习，但这种方式不能处理开放世界中训练时未出现过的 novel objects。本文指出，问题在于现有 proposal方法中的二分类器倾向于对训练类别产生过拟合。因此，本文提出了一个无分类的物体定位网络（*Object Localization Network* **OLN**），它纯粹通过一个区域的**位置**和**形状**与任何 *ground-truth* 物体的重叠程度（如 *centerness* 和 *IoU* ）来估计每个区域的 *objectness*。这种简单的策略学习了可一般化的 objectness ，并在COCO的跨类别泛化，以及在 *RoboNet,* *Object365 *和 *EpicKitchens* 的跨数据集评估方面优于现有的 proposals 。

> **Object Localization Network (OLN)**

和 Faster R-CNN 相似，OLN 是一个两阶段的候选框提取器，由全卷积网络和 ROI 组成，不同在于FPN and ROI 阶段的分类器被 localization quality predictions 替代，

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230218155122722.png" alt="image-20230218155122722" style="zoom:80%;" align="left"/>

- **OLN-RPN**：以来自ResNet特征金字塔的特征作为输入，每个特征图都经过一个卷积层，然后是两个独立的层，一个用于边界盒回归，另一个用于定位质量预测。网络架构的设计遵循了标准的 RPN 。选择中心性作为定位质量目标，并用 L1 损失训练两个头。在提议阶段学习定位而不是分类是至关重要的，因为它可以避免通过分类过度拟合前景。

  为了训练定位质量估计分支，随机抽取 256 个 IoU 大于 0.3 的anchors与匹配的真实框，没有任何明确的背景抽样。对于框回归，将标准框增量目标（xyhw）替换为从该位置到 ground-truth 框（lrtb）四个边的距离
- **OLN-Box**：从OLN-RPN获取得分最高(eg: well-centered)的proposals，并执行RoIAlign。然后对每个区域特征进行线性化，并通过两个 fc 层，一个用于边界框回归，另一个用于定位质量预测。 使用与 Faster R-CNN 相同的网络架构，选择 IoU 作为定位质量目标，并用 L1 损失训练两个头。 在第二阶段学习定位质量是不可或缺的，因为它使得模型改进 proposal scoring ，同时避免过度拟合到前景。

  下表展示了 OLN 在跨数据集上的泛化能力：

  <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230218161048638.png" alt="image-20230218161048638" style="zoom:80%;" />

  下表验证了本文的假设，即分类的判别学习阻碍了 object proposal 的泛化，而纯粹的基于定位的对象性是泛化的关键。

  ![image-20230218161222029](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230218161222029.png)

## 20230303

### 7_Featurized Query R-CNN_Tech Report2022_有代码

> 作者：[Wenqiang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Tianheng Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+T), [Xinggang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Shaoyu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+S), [Qian Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Q), [Wenyu Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+W)

> 代码：https://github.com/hustvl/Featurized-QueryRCNN

> 贡献：

由于当前 query-based 检测框架存在以下两个问题：1、使用多层 decoder 结构来优化随机初始化的 object query 需要耗费大量的计算资源；2、由于训练之后 query 是固定的，其泛化能力因此受到限制。为解决上述问题，本文基于 Faster R-CNN 框架，提出使用**QGN**(query generate net) 产生 featurized object queries 的方法，在所有 R-CNN 系列检测模型中（包括当前 SOTA 的 Sparse R-CNN）达到了最好的 speed-accuracy trade-off。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230301223747566.png" alt="image-20230301223747566" style="zoom:80%;" />

如 Figure 1 所示，相对于 Faster R-CNN 本文去除了 NMS 从而提高了推导速度；相比于 Sparse R-CNN，使用图片信息获得的 featurized query 而不是可学习的 query ，相当于 query 本身就带有部分先验信息，因此能够使用少量的 decoder 就能维持相当的检测性能。

> 网络框架图：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230301225120069.png" alt="image-20230301225120069" style="zoom:80%;" />

> **QGN**:

基于从 FPN 获得的特征图（P3-P7），先经过 [3,3,256] 的卷积层，然后使用三个 1*1 的卷积层分别预测objectness prediction, query box regression 和 query feature。其中 query box regression 是 pixel 中心到 gt 边框四边的距离 $(l,t,r,b)$，类无关的objectness score 将被用来选择 top K 个 query feature 和 query box。

 **Query-based R-CNN Head**：

由于本文使用的级联动态交互 head 模块数较少，因此在Dyn. Conv 后面加了一层自注意力层，作者表明这个自注意力层在 动态交互模块较少的时候很 work。

 **Cascade Refinement**：

级联优化，相较于 Sparse R-CNN 用 6 个 decoder 层来优化 image-agnostic object querie，由于使用了 featurized query，本文只使用两级优化（一个标准 decoder（左） + 一个 query-based R-CNN head（右））就能达到较好的效果。而且，实验表明增加级联的层数，除了带来巨大的计算开销，并不会带来多少性能的提升（如 Figure 7）。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230301233417021.png" alt="image-20230301233417021" style="zoom:80%;" /><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230301234217235.png" alt="image-20230301234217235" style="zoom:80%;" />

总结：直觉上， Sparse R-CNN 使用随机初始化可学习的 query ，学起来肯定没有带 image content 的 featurized query 学的快。

### 8_PROB: Probabilistic Objectness for Open World Object Detection_2022_有代码

> 作者：[Orr Zohar](https://orrzohar.github.io/), [Jackson Wang](https://wangkua1.github.io/), [Serena Yeung](https://marvl.stanford.edu/people.html)

> 代码：https://github.com/orrzohar/PROB

> 贡献：

针对OWOD目标检测问题，由于缺乏对 unknow object 的监督，现有模型无法很好地区分 unknow 类和背景类，因此在 unknow 类的性能表现不佳。本文基于 Deformable-DETR ，提出 probabilistic 模块来衡量 query 是一个 object 的可能性，并且在嵌入空间最大化已匹配的 query 的 objectness likelihood，来帮助模型找到 object。然后结合 objectness head 与 classifification head 来提高模型对已知类以及未知类的检测性能。

模型框架：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230303160342656.png" alt="image-20230303160342656" style="zoom:80%;" />

> **Probabilistic Objectness**：

目前大多数OWOD方法都试图在训练的时候以伪标签的方式识别未知对象，本文另辟蹊径，将识别对象和识别对象类别分开来。

识别对象：一个 query 是 object 的可能性 $p(o|q)$， 由 objectness head 来学习；

识别对象类别：一个是 object 的 query 是类别 $l$的可能性$p(l|o,q)$，由 classifification head （在已知一个 query 是不是 object 的情况下）来学习；

则修改后的类别预测的推导方式可以表示为：$p(l|q) = p(o|q) * p(l|o,q) \quad = f_{cls}^{t}(q)*f_{obj}^{t}(q)$

本文使用多元高斯分布来拟合 $p(o|q)$的参数，通过在 mini-batch 上计算 query embedding 的均值和协方差的指数移动平均值来估计 embedding 的概率分布：

$$
f_{obj}^{t}(q)&=exp(-(q-u)^T\sum ^{-1}(q-u)) \\
&=exp(-d_{M}(q)^{2})
$$

其中$d_{M}$代表 query embedding 之间的 Mahalanobis distance 。

训练时轮流进行两个步骤：1、评估 $p(o|q)$的参数；2、通过 Mahalanobis distance 来最大化 匹配的 query 的 likelihood。

约束损失 objectness loss：$L_{o}=\sum_{i \in Z}d_{M}(q_i)^{2}$，$Z$代表 matched queries。

**Objectness for Incremental Learning**：

有别于之前OWOD方法随机选择样本来缓解灾难性遗忘问题，本文根据 objectness 来选择：即分别选择 25 个**高分/低分**的 instances 作为样例。其中 objectness 得分低的 instance 可以认为是难样本，用于指导模型学习新引入的 object；objectness 得分高的 instance 说明模型对该类学的比较好，用于防止灾难性遗忘。

> **数据集划分**：

本文在两种数据集划分设定下评估模型性能

* ORE 中的 superclass-mixed OWOD benchmark
* OW-DETR 中的 superclass-separated OWOD benchmark（由于每个任务中的类别按照超类划分因此更具挑战性）

各部分作用的消融实验：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230303160207054.png" alt="image-20230303160207054" style="zoom:80%;" />

## 20230310

### 9_OpenWGL: Open-World Graph Learning_2020 ICDM_有代码

**ICDM**：IEEE International Conference on Data Mining

> 作者：Man Wu , Shirui Pan , Xingquan Zhu

> 代码：https://github.com/GRAND-Lab/OpenWGL

> 贡献：

针对开放世界图学习的基本挑战：1、看不见的类没有带标签的样本，并且可能以不同于现有可见类的任意形式存在；2、图特征学习和预测都应该区分节点可能属于现有/可见类还是不可见类。为了应对这些挑战，本文提出了一种**不确定节点表示学习方法**，使用约束变分图自动编码网络，其中标签损失和类不确定性损失约束用于确保节点表示学习对看不见的类敏感。因此，**节点嵌入特征由分布表示**，而不是确定性特征向量。通过**使用采样过程生成多个版本的特征向量**，能够测试属于可见类的节点的确定性，并自动确定一个阈值，将不属于可见类的节点拒绝为不可见类节点。

> **框架：**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230307105141625.png" alt="image-20230307105141625" style="zoom:80%;" />

**Node Uncertainty Representation Learning**：

节点不确定性表示学习

大多数GCN模型生成确定性映射（deterministic mappings）来捕获节点的潜在特征。这些模型的一个主要局限性是它们不能表示由不完整或有限的可用数据（ incomplete or fifinite available data）所造成的不确定性。为了更好地学习每个节点的表示，本文采用变分图自动编码器网络（ Variational Graph Autoencoder Network）来获得每个节点的潜在分布，从而能够表示不确定性并提高鲁棒性。

为了对每个节点的潜在特征信息进行编码并获得有效的不确定性表示，本文采用变分图自编码网络（VGAE）根据提取的节点特征生成潜在分布，能够利用不确定性进行鲁棒表示学习。

* 图编码器模型：给定一个图 $G = (X, A)$，为了在统一的框架中表示节点内容 $X$ 和图结构 $A$，本文使用两层 GCN。

  * 第一层 GCN 生成低维特征矩阵：$Z^{(1)} = GCN(X,A)=ReLU(\widetilde{D}^{-1/2}\widetilde{A}\widetilde{D}^{1/2}XW^{(1)})$
  * 第二层 GCN ，假设输出 Z 是连续的，服从多元高斯分布，而不是生成确定性表示。因此本文遵循推理模型：

    * $q(Z|X,A) = \prod_{i=1}^n q(z_{i}|X,A)$
    * $q(z_{i}|X,A) = N(z_{i}|\mu,diag(\sigma_{i}^{2}))$

      其中 $\mu = GCN_{\mu}(X,A)=ReLU(\widetilde{D}^{-1/2}\widetilde{A}\widetilde{D}^{1/2}Z^{1}W^{(2)})$是平均向量 $\mu _{i}$的矩阵；$\sigma $是分布的标准方差矩阵， $log \sigma = GCN_{\sigma}(X,A)=ReLU(\widetilde{D}^{-1/2}\widetilde{A}\widetilde{D}^{1/2}Z^{1}W^{`(2)})$

      然后可以使用参数化技巧计算 Z：$Z = \mu + \sigma .\zeta, \zeta ~ N(0,I)$，0 是零向量，I 是单位矩阵。通过使用潜变量 Z，模型可以捕获数据中复杂的噪声模式。
* 图解码器模型：在得到潜在变量 Z 后，使用解码器模型重建图结构 A，以更好地了解两个节点之间的关系。这里，图形解码器模型由生成模型定义：

  * $p(A|Z) = \prod_{i=1}^n\prod_{j=1}^np(A_{i,j}|z_{i},z_{j})$
  * $p(A_{i,j}=1|z_{i},z_{j})=\sigma(z_{i}^T z_{j})$, 其中$\sigma $ 表示 sigmoid 函数。
* 优化：为了更好地学习类判别节点表示，本文通过以下两个损失优化变分图自动编码器模块

  * $L_{VGAE} = E_{q(Z|X,A)}[logp(A|Z)] - KL[q(Z|X,A) || p(Z)]$，其中第一项是输入邻接矩阵和重构邻接矩阵之间的重构损失，第二项是 $q 和 p$ 之间的 KL 散度，$p(Z) = N(0,I)$。

**Open-World Classififier Learning**：

为了将 seen class 节点正确分类并检测 unseen class 节点，本文引入了两个约束条件：标签损失（ label loss）和类不确定性损失（class uncertainty loss），来区分一个节点是属于现有的类还是属于不可见的类。总体目标函数：

$$
L_{OpenWGL} = \gamma_{1}L_{L} + \gamma_{2}L_{C} + L_{VGAE}
$$

本文将熵损失作为类不确定性损失，目标是最大化熵损失，使每个节点的归一化输出达到平衡。不是使用所有未标记的数据来最大化熵损失，首先将所有未标记的数据在softmax层之后输出概率值排序，然后**丢弃**最大的10%（具有较大概率值的节点很容易划分为可见类，因为它们的输出是有区别的）和最小的10%节点（具有较小概率的节点意味着节点的输出在每个可见类上是平衡的，可以很容易地检测为不可见类），最后利用剩余节点最大化其熵。
标签损失和类别不确定性损失的训练就像一个对抗性的过程。一方面，希望标签损失影响分类器，使每个节点的输出更具区分性，并通过最小化等式$L_{L}$将每个可见节点分类为正确的类。另一方面，希望类不确定性损失可以使每个节点的输出更加平衡，从而通过最大化熵损失$L_{C}$来帮助检测不可见的类。

**Open-World Classifification & Rejection**：

在测试阶段执行推理时，本文提出一个新的采样过程，通过生成多个版本的特征向量来测试节点属于已知类别的可能性，并自动确定一个阈值拒绝不属于已知类的节点，作为 unseen nodes 。

![image-20230307112010113](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230307112010113.png)

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230307165452064.png" alt="image-20230307165452064" style="zoom:67%;" />

### 10_Objects in Semantic Topology_ICLR 2022_无代码

> 作者：[Shuo Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+S), [Peize Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+P), [Yi Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+Y), [Xiaobo Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+X), [Ruiheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Zehuan Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+Z), [Changhu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Ping Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+P), [Min Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+M)

> 贡献：

针对 OWOD 的未知类识别以及增量学习问题，现有模型采用的方式为针对前述两个分别应用一个独立的模块。本文通过引入**语义拓扑（*Semantic Topology*）**的概念，同时解决上述两个问题。在学习 OW 的 OD 时，将同一类别内的所有 object instances （ unknow 类也是一样的操作）分配给在 semantic topology 相对应的预定义节点。这种约束建立了对象之间的**可区分的特征表征（discriminative feature representations）**和**一致性关系（consistent relationships）**，从而使检测器能够从已知类别中区分出未知对象，并在逐步学习新类别时使已知对象的学习特征不被扭曲。

> 框架：

**两个关键部分：**

* 可以识别未知的区域建议网络 (RPN) ，采用 ORE 中提出的 Unknown-Aware RPN；
* 具有判别性和一致性的特征空间，使用预定义的语义拓扑来约束检测器的特征空间拓扑。

具体来说，为特征空间中的每个类别创建一个**唯一**且**固定**的质心，称为 *semantic anchor*。 所有类的  *semantic anchor* 都是通过将它们的**类名**送到预先训练的语言模型来获得的。
关键思想是在检测器的整个生命周期中操纵检测器的 ***feature space***，使其与  ***semantic anchor*** 构成的语义拓扑保持一致。由于特征维度的差异，使用全连接层来对齐 RoI 特征和 semantic anchor 之间的维度。

* 在训练阶段，semantic projector 输出的  semantic features 被设计的“SA（semantic anchor）头”强制聚集在其相应的 semantic anchor 周围。
* 当增量学习时，SA Head 逐渐为新类注册新的 semantic anchor，并不断拉近新的类特征及其语义锚。
* 为了减轻由旧类特征失真引起的“灾难性遗忘”，SA Head 还在学习新类时最小化了一些存储的旧类示例与其语义锚之间的距离。
* 为了更好地利用结构良好的特征空间，本文附加了一个额外的分类层来对语义特征进行分类。
  Figure 3 显示了所提出的开放世界对象检测器的训练流程。在推理时，将两个分类头产生的类后验概率相乘，得到最终预测。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230307151530681.png" alt="image-20230307151530681" style="zoom:80%;" />

 **SEMANTIC TOPOLOGY**

假设 $l_{i} \in C^{t}_{kn}$是第 $t$ 时刻第 $i$ 个类别的 class name ，$M$ 为预训练好的语言模型。类 $l_{i}$的 semantic anchor $A_{i} = M(l_{i})$，其中$A_{i}  \in R^{n}$，维度 n 取决于预训练模型的输出维度。在$ t + 1$ 时，只要已知类集更新$C_{kn}^t → C_{kn}^{t+ 1} $，就重复执行语义锚注册。遵循相同的策略，未知类的  semantic anchor 也是由未知文本的 unknow 生成。

**OBJECTIVE FUNCTION**

RoI 特征 $f  \in R^{d}$ 是由对象检测器的中间层生成的特征向量，用于类别分类和边界框回归。本文操纵 RoI 特征 $f$  来构建检测器的特征流形拓扑。

将 $f_{i}$  表示为第 $i$个已知类的 RoI 特征，本文首先使用具有 $d × n$ 维权重的全连接层将 $f_{i}$ 的维度对齐为其对应的 semantic anchor $A_i $ 。对应的语义特征记为 $ \hat{f}_{i} \in R^{n} $ 。

* 通过围绕相应的 semantic anchor 聚类语义特征来约束检测器的特征流形拓扑，学习目标形式化为 $L_{sa}= ||\hat{f}_{i} - A_i ||$；
* 为了更好地利用构建的特征空间，本文使用一个额外的分类头来对语义特征 $ \hat{f}_{i} $ 进行分类，用的标签和 ROI 分类头一样；
* 总的训练目标是语义锚损失${L}_{s a}L $ 、语义特征分类损失${L}_{cl s_{s e}}$、RoI特征分类损失和边界框的组合回归损失:$L_{total} = L_{sa} + {L}_{cl s_{s e}} + {L}_{cl s_{roi}} + {L}_{reg}$

## 20230317

### 11_OTA: Optimal Transport Assignment for Object Detection_CVPR2021_有代码

> 作者：[Zheng Ge](https://arxiv.org/search/cs?searchtype=author&query=Ge%2C+Z), [Songtao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Zeming Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Osamu Yoshie](https://arxiv.org/search/cs?searchtype=author&query=Yoshie%2C+O), [Jian Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+J)

> 代码：https://github.com/Megvii-BaseDetection/OTA

> 贡献：

本文提出把标签分配当做**最优传输**问题，具体是把每个gt定义成一个 supplier，它可以提供一定数量的 label。把每个 anchor 定义成 demander，它需要一个 label。如果一个 anchor 从某个 gt 那得到了足够数量的 positive label，这个 anchor 就被当做这个 gt 的一个正样本。每个 gt 可以提供的 positive label 的数量可以理解为这个 gt 在训练过程中需要多少个正样本来更好的收敛。每对 anchor-gt 的传输 cost 定义为它们之间的分类和回归 loss 的加权和。此外，背景类也被定义为 supplier，它提供 negative label，anchor-background 之间的传输 cost 定义为它们之间的分类 loss。这样标签分配问题就被转化为了最优传输问题，最终是为了找到全局最优的分配方法而不再是为每个gt单独寻找最优anchor。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230313113516186.png" alt="image-20230313113516186" style="zoom:67%;" />

**Optimal Transport**

最优传输问题可以表述为：假设有 $m$ 个 $supplier$ 和 $n$ 个 $demander$，第 $i$  个 $supplier$ 有 $s_{i}$ 个物品，第 $j$ 个 $demander$ 需要 $d_{j}$个 物品，每个物品从第 $i$ 个 $supplier$ 运到第 $j$ 个 $demander$ 的运输运输成本为 $c_{ij}$ ，最优传输的目标是找到一个最优传输方案 $π^{∗}=\{π_{i,j}|i=1,2,...m,j=1,2,...n\}$ 能以最小的运输成本把所有的物品从 $supplier$ 运输到 $demander$。

**OT for Label Assignment**

对于目标检测问题，假设一张图片有 $m$ 个 gt 和 $n$ 个 anchor（所有FPN level加起来），每个 gt 当做一个 $supplier$，有 $k$ 个正标签 ($i.e.,s_{i}=k,i=1,2,...,m$)，每个 anchor 当做一个 $demander$，需要一个标签 ($i.e.,d_{j}=1,j=1,2,...,n$)。从 $gt_{i}$  传输一个正标签到 anchor $a_{j}$ 的运输成本 $c_{ij}^{fg}$ 定义为它们之间的分类损失和回归损失的加权和:

$$
c_{ij}^{fg}=L_{cls}(P_{j}^{cls}(θ), G_{i}^{cls})+αL_{reg}(P_{j}^{box}(θ), G_{i}^{box}) \tag{2}
$$

其中 $θ$ 是模型参数，$P_{j}^{cls}$  和 $P_{j}^{box}$分别 表示anchor $a_{j}$ 的**预测的**分类得分和bounding box。$G_{i}^{cls}$ 和 $G_{i}^{box}$ 分别表示 $gt_{i}$ 的ground truth类别和bounding box。$L_{cls}$ 和 $L_{reg}$ 分别表示交叉熵 loss 和 IoU loss，也可以分别替换成 Focal loss 和 GIoU/Smooth L1 loss，α 是权重系数。

此外，还有另一种提供负标签的 supplier，背景类。在标准的最优传输问题中，**supply 的数量和 demand 的数量是相等的**。因此背景类一共可以提供 $n−m×k$ 个负标签，从背景类传输一个负标签到 $a_{j}$ 的成本为:

$$
c_{j}^{bg} = L_{cls}(P_{j}^{cls}(θ), ∅) \tag{3}
$$

其中 $∅$ 表示背景类，把 $c^{bg}∈R^{1×n}$ 拼接到 $c^{fg}∈R^{m×n}$ 的最后一行即得到了完整的 cost matrix $c ∈ R^{(m + 1)×n}$。supply vector s 需要按下式更新：

$$
s_{i} =
\left\{\begin{matrix}
k, \quad if \quad i ≤ \quad m\\
n - m * k, \quad if i = m + 1 \\
\end{matrix}\right.
\tag{4}
$$

现在有了cost matrix $c ∈ R^{(m + 1)×n}$，supply vector $s ∈ R^{m+1}$，demand vector $d∈R^{n}$，则最优传输路径 $π^∗∈R^{(m+1)×n}$ 可通过现有的 Sinkhorn-Knopp Iteration 算法求得。**得到 $π^∗$ 后，对应的标签分配就是将每个 anchor 分配给传输给这个 anchor 最多标签的gt**。

> **Advanced Designs**

**Center Prior**
center prior 即只从 **gt 的中心有限区域**挑选正样本，而不是整个 bounding box 范围内选择。强迫模型关注**潜在 positive areas 即中心区域**有助于稳定训练，特别是在训练的早期阶段，模型的最终性能也会更好。作者发现 center prior 对 OTA 的训练也有帮助，因此引入了 center prior 策略。

具体做法是，对于每个 gt ，只挑选每个 FPN 层中**距离 bounding box 中心最近**的 $r^2$ 个anchor，对于 bounding box 内 $r^2$ 之外的anchor，cost matrix 中对应的 cost 会加上一个**额外的**常数项，这样就减少了训练阶段它们被分配为正样本的概率。

**Dynamic k Estimation**
每个 gt 需要的正样本数量应该是不同的并且基于很多因素，比如物体大小、尺度、遮挡情况等。由于很难将这些因素和所需 anchor 数量直接映射起来，本文提出了一种简单有效的方法，根据**预测框和对应 gt 的 IoU 值**来粗略估计每个 gt 合适的正样本数量。

具体来说，对于每个 gt ，选择 IoU 最大的 q 个预测，将这 **q 个 IoU 值的和**作为这个 gt 正样本数量的粗略估计值。这样做是基于直觉：某个 gt 的所需合适的 postive anchor 数量与和这个 gt 拟合的很好的 anchor 的数量正相关。

### 12_Assessing Domain Gap for Continual Domain Adaptation in Object Detection_CVIU_有代码

**CVIU**：Computer Vision and Image Understanding

> 作者：Anh-Dzung Doan, Bach Long Nguyen, Surabhi Gupta, Ian Reid, Markus Wagner, Tat-Jun Chin

> 代码：https://github.com/dadung/DGE-CDA

> 贡献：

本文研究了三种流行的域间隙评估度量方法，发现域间隙与检测精度之间存在相关性。为了减少不断调整检测器以适应域变化带来的高计算成本，本文提出**以域间隙**作为标准来**决定何时**调整检测器。只有在必要时，才有选择性地，使用与当前训练数据没有相同分布的新数据调整检测器。

> 模型架构：

本文呢采用 RetinaNet 作为OD，见图1a。具体来说，ResNet 主干有三个残差块 C3、C4 和 C5，其输出是特征金字塔网络（FPN）的 P3、P4 和 P5 的输入。然后，class 和 box 子网 B1、B2 和 B3 接收FPN的输出，预测对象类别和边界框。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230313160059913.png" alt="image-20230313160059913" style="zoom:80%;" align="left"/><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230313160150642.png" alt="image-20230313160150642" style="zoom:80%;" />

**Domain gap evaluation**：

源域图像为$D_{s} = \{x_{i}^{s} \} ^{N_{s}}_{i=1}$，目标域图像为$D_{t} = \{x_{i}^{t} \} ^{N_{t}}_{j=1}$。首先，将使用 RetinaNet 在源域图像上进行训练，见图1a。然后如图1b，C3、C4 和 C5 的输出用于评估源图像和目标图像之间的域间隙（在此步骤中不需要使用标签）。

> 三种距离：

*4.1. Maximum mean discrepancy (MMD)*

/*

ref：https://zhuanlan.zhihu.com/p/471732960

最大均值差异（MMD）是迁移学习，尤其是域适应中使用最广泛的一种损失函数，主要用来度量**两个不同但相关的随机变量**的**分布的距离**。基本思想式，如果两个随机变量的的任意阶矩都相同的话，那么这两个随机变量的分布一致。若两个随机变量的分布不相同，那么使得两个分布之间差距最大的那个矩被用来作为度量两个随机变量距离的标准。**基本定义式**：

$$
MMD[F,p,q]:=sup_{f \in F}(E_{p}[f(x)]-E_{q}[f(y)])
$$

此式的含义是寻找一个映射函数 $f$，这个映射函数能够将变量映射到高维空间，之后求两个分布的随机变量在映射后的期望的差，这个差值便是 Mean Discrepancy ，然后寻找这个 Mean Discrepancy 的上确界。这个最大值便是 MMD 。

*/

$MMD(D_{s} , D_{t}) = \frac {1}{3}\sum_{k=3}^{5}MMD_{k}(D_{s} , D_{t})$

其中$MMD_{k}(D_{s} , D_{t}) =||\frac{1}{N_{s}}\sum_{i=1}^{N_{s}}C_{k}(x_{i}^{s})-\frac{1}{N_{t}}\sum_{i=1}^{N_{t}}C_{k}(x_{j}^{t})||^{2}, \quad k = 3,4,5$

*4.2. Sliced Wasserstein distance (SWD)*

与其他度量（例如，詹森-香农散度(Jensen-Shannon divergence)、KL散度和总变化度( total variation distance)）相比，**WD考虑**了概率空间的**基本几何形状**。SWD是WD的一个变种，其目的是处理高维数据。

* 令$D_{S} \quad D_{t}$ 的经验分布为 $\mu_{s} = \sum _{i=1}^{N_{s}}p_{i}^{s} \delta _{x_{i}^{s}} \quad \mu_{t} = \sum _{i=1}^{N_{t}}p_{j}^{t} \delta _{x_{j}^{t}}$，其中 $\delta _{x_{i}^{s}} \quad \delta _{x_{j}^{t}}$分别为位置$x_{i}^{s} \quad x_{j}^{t}$的狄拉克函数。（**狄拉克δ函数**是一个广义函数，在物理学中常用其表示质点、点电荷等理想模型的密度分布，该函数在除了零以外的点取值都等于零，而其在整个定义域上的积分等于1。）$p_{i}^{s} \quad p_{j}^{t}$为样本的概率质量，一般为$p_{i}^{s} = \frac{1}{N_{s}} \quad p_{j}^{t} = \frac{1}{N_{t}}$；
* 定义集合 $B = \{\gamma \in (R^{+})^{N_{s}*N_{t}}|\gamma1_{N_{t}} = \mu_{s},\gamma1_{N_{s}}^{T} = \mu_{t}\}$，其中$1$为元素值都为1的向量；
* 对于$k = 3,4,5$，$WD_{k}(D_{s},D_{t}) = min_{\gamma \in B}<\gamma,D_{k}>_{F}$， $<·, ·>_{F}$ 是 Frobenius dot product，cost matrix：$D_{k}(i, j) = ||C_{k}(x_{i}^{s}) - C_{k}(x_{j}^{t})||^{2}$

$C_{k}(x)$的维度会导致$D_{k}$的计算量很大，因此SWD的主要思想是将$C_{k}(x)$投射到 1维：

* 定义集合$\{R_{k,m}\}_{m=1}^{M}$，$R_{k,m}$是第 $m$个 1 维线性投射样本，SWD：$SWD_{k}(D_{s},D_{t}) = \frac{1}{M} \sum_{m=1}^{M} min_{\gamma \in B}<\gamma,\widehat{D}_{k}>_{F}$，其中$\widehat{D}_{k}(i, j) = ||R_{k,m}^{T} C_{k}(x_{i}^{s}) - R_{k,m}^{T} C_{k}(x_{j}^{t})||^{2}$

最终 $SWD(D_{s} , D_{t}) = \frac {1}{3}\sum_{k=3}^{5}SWD_{k}(D_{s} , D_{t})$

*4.3. Distance of second-order statistics (DSS)*

![image-20230313174453153](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230313174453153.png)

![image-20230313174522449](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230313174522449.png)

![image-20230313174605426](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230313174605426.png)

> **Application of domain gap evaluation in continual domain adaptation**

在增量域适应过程中，**如果发现域间隙小于预定义的阈值，则丢弃新数据**。因为若新数据和训练数据之间的域差距很小，这两个数据集很可能共享相似的分布，利用这样的新数据调整网络并不会显著提高其性能，还会消耗不必要的资源。但是，**如果发现域间隙大于阈值，则将新的数据添加到训练数据库中，并用于调整当前的网络**。

> 结果：

使用本文提出的根据域间距离选择性调整模型的方法，在几乎不影响模型检测性能的情况下大大减少了计算资源的消耗。

![image-20230313162719064](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230313162719064.png)

## 20230324

### 13_Learning to Decompose Visual Features with Latent Textual Prompts_ICLR 2023_无代码

> 作者：Feng Wang1 , Manling Li2 , Xudong Lin3 , Hairong Lv1 , Alexander G. Schwing2 & Heng Ji2

​				1Tsinghua University 2University of Illinois at Urbana-Champaign 3Columbia University

> 贡献：

针对 CLIP 类模型存在的两个问题：1、在 zero-shot 范式下，通过检索文本类名进行推断时，准确性和鲁棒性会降低；2、在 linear probing 范式下，会打破 well-established 视觉语言对齐。本文提出 分解特征提示 DeFo（**De**composed **F**eature Pr**o**mpting），基于 CLIP 双模型架构，通过**可学习的嵌入作为文本输入**并添加一个**额外的线性层**来进行分类，DeFo 能够在文本提示的帮助下提取到**分解的视觉特征**  decomposed visual features 。此外，DeFo 支持可变的语言输入规模（不受限于类别数量）。

DeFo在使用 ResNet-50 backbone 的 ImageNet 上获得了 73.2% 的测试精度，比 zero-shot CLIP 高15.0%，比 SOTA 的 vision-language prompt tuning 高7.6%。

**hard-target retrieval**：

CLIP 类模型在进行 zero-shot 推理时，直接计算从 image encoder 获得的 vectorial image representation 与从 language encoder 获得的 文本提示表示 之间的距离。与图像的表示向量距离最小的文本提示符对应的目标类构成 zero-shot 推理结果。

使用 hard textual targets 进行推理存在以下两个问题：

* **expressive sensitivity:** text prompt 中的类别名称无法准确地总结图像中的语义信息，这导致推理结果非常受到类别名称选择的影响。（如 "plane" vs "airplane"，"car" vs "automobile"）
* **conceptual sensitivity:** 尽管数以亿计的预训练样本覆盖了大量可能出现在下游数据集中的概念，但 zero-shot 推理仍然难以识别稀有物体。

因此，本文提出 DeFo ，将 CLIP 类模型的 硬目标检索范式 转化为 双模型特征提示：

* DeFo 为 language encoder 提供了一组独立于  hard semantic targets 的可学习的嵌入序列；
* 通过调优一个附加的额外线性层来执行分类。

> 方法

DeFo 致力于使用 language encoder 构建一个**映射矩阵**，将视觉特征从 CLIP 潜在空间的 $d$-维 映射到 $n$-维的特征空间。

其中只有**线性分类层 **和 **textual queries** 中的参数是可训练的。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230315192850364.png" alt="image-20230315192850364" style="zoom:80%;" align="left"/>

如图，DeFo 的构成：

1、visual encoder$g_{V} : R^{w×h×3}→R^{d}$

输入：$w×h×3$ 的图像

输出：$f_{I} ∈ R^{d}$

2、language encoder$g_{L} : R^{m×d_{e}}→R^{d}$

输入：$X_{L}∈R{n×m×d_{e}}$ ,n 个带有 $m$ 个单词的  query sentences，每个单词被嵌入到 $d_{e}$ 维的向量中

输出：$f_{T}^{1},f_{T}^{2}, . . . ,f_{T}^{n} ∈R_{d}$

3、通过将经ℓ2标准化的$f_{I}$ 和每个经ℓ2标准化的 $f_{T}^{i}$ 做点乘，得到 $n-$ 维向量，第 $i$ 个元素即表示该图与第 $i$ 个 text query 的相似度。

4、通过一个线性层将$n-$ 维投射到 $k-$ 维，并对$k-$ 维向量进行 softmax 计算 probabilities。$p_{i} = \frac{exp(⟨f_{I},f_{T}^{i}⟩)/τ}{\sum_{j=1}^{k}exp(⟨f_{I},f_{T}^{i}⟩)/τ}$

### 14_Learning Object-Language Alignments for Open-Vocabulary Object Detection_ICLR 2023_有代码

> 作者：Chuang Lin,Peize Sun,Yi Jiang,Ping Luo,Lizhen Qu,Gholamreza Haffari,Zehuan Yuan,Jianfei Cai

> 代码：https://github.com/clin1223/VLDet

> 贡献：

本文提出直接从图像-文本对 ( image-text pair ) 中学习 fine-grained 对象-语言 ( object-language ) （又称 region-word ）对齐，将其看作集合匹配任务，使用匈牙利匹配算法，训练一个端到端的 Open-Vocabulary 目标检测器。在 Open-vocabulary LVIS and Open-vocabulary COCO 数据集上获得了 SOTA 性能。

**Open-vocabulary Object Detection：**

开放词汇表目标检测致力于构建这样一个目标检测器：通过在**带有 base-class 边框标注信息的数据集** $C^{base}$ 和 **包含大量词汇的 image-caption 对的数据集** $C^{open}$ 上训练，使模型在测试阶段有能力检测 novel classes $C^{novel}$ 。

注： $C^{base}$ + $C^{novel}$ 有可能会、有可能不会和 $C^{open}$ 存在交叉。

> 方法

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230316190316363.png" alt="image-20230316190316363" style="zoom:80%;" />

**通过 Bipartite Matching 学习 Object-Language 对齐：**

注：object <--> region  language <--> word

二分图匹配描述的是 **X** 个 workers 和 **Y** 个 jobs 之间的分配问题

* 本文中，来自图像 $I$ 的 regions $r_{i}$ 作为 jobs，来自描述 $C$ 的 words $w_{j}$ 作为 workers。
* 给定一张图像 $I$ ，通过 RPN 获得来自 image encoder 的候选区域特征 $R = [r_{1}, r_{2}, ... , r_{m}]$，$m$ 为  候选 regions 个数；
* 给定一个描述 $C$ ，从中选取所有**名词**，并用 language encoder 编码成 word embedding $W = [w_{1}, w_{2}, ... , w_{|W|}]$，$|W|$ 为描述 $C$ 中名词的数量； 通常 $m > |W|$
* regions 和 words 之间的对齐分数为 $S = WR^{T} \quad \quad \quad |W|*m$
* 匹配操作后，通过以下交叉熵损失对分类头进行优化：

$$
L_{region−word} = \sum_{i=1}^{|W|} - [log\sigma(s_{ik}) + \sum_{j \in W′} log(1 - \sigma(s_{ik}))] \tag{2}
$$

其中$\sigma$ 为 sigmoid 激活函数，$s_{ik}$ 为第 $i$ 个 word embedding 和 第 $k$ 个 region feature 之间的对齐分数；$W′$ 表示同一 batch 中 其它 Caption 中的名词。

进一步，本文还将图像-文本对视为特殊的区域-词对。通过将整个图像作为一个特殊区域，将文本编码器中的整个 caption 特征作为一个特殊单词，从而提取图像的 RoI 特征。对于一个图像，将其 caption 视为阳性样本，而同一 batch 中的其他 caption 作为阴性样本。用类似于公式 (2) $L_{image−text}$ 损失作为约束。

> 框架

VLDet 网络包含三个组成部分：一个视觉目标检测器、一个文本编码器、regions 和 words 的对齐

本文使用 CLIP 作为 text encoder，遵循 Faster-RCNN架构，第一阶段不做改变，通过 RPN 产生 2000 个 region proposal；第二阶段进行了如下两项调整来适应 open-vocabulary 设定：

* 将 class-specific 的定位头换成 class-agnostic 的；
* 用 language embedding 替换可训练的分类器权重，将检测器转换为开放词汇表设置。

> 消融

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230317112334504.png" alt="image-20230317112334504" style="zoom:80%;" />

* 通过对比使用 类别名称 vs 描述中的所有名词，由于描述中包含的名词比类别数**要多**很多，因此后者效果更好。
* 通过对比 1 v 1的  Hungarian 和 1v many 的 Sinkhorn（最优传输 OTA 中用到的算法），前者表现更好。 1 v 1的分配的假设通过为每个单词提供一个高质量的图像区域，从而减少了错误对齐。

  （考虑到在一幅图像中可能存在同一类别的多个实例，Sinkhorn 算法能够使同一单词 对应多个区域，但同时也可能引入更多**有噪声的**区域-单词对。）

> 总结

本文亮点，通过从粗粒度的 image-caption 对中学习细粒度的 region-word 对，由于 caption 中所包含的名词 object 肯定会多于 类别数量，因此能够较好地提高模型检测 novel 类的能力。

## 20230331

### 15_Energy-based Out-of-Distribution Detection for Graph Neural Networks_ICLR 2023_有代码

> 作者：Qitian Wu, Yiting Chen, Chenxiao Yang, Junchi Yan

​				Department of CSE & MoE Lab of Artificial Intelligence, Shanghai Jiao Tong University

> 代码：https://github.com/qitianwu/GraphOOD-GNNSafe

> 贡献

本文研究了图结构数据的 OOD 检测问题，并基于标准分类损失训练的能量函数确定了一个可证明有效的 OOD 鉴别器 $GNNS_{AFE}$。更进一步，可以通过无学习能量信念传播方案进一步聚合相邻节点的能量值，得到的新能量值就更容易区分出来自不同分布的节点。

> 方法

**1- ENERGY-BASED OOD DETECTION WITH DATA DEPENDENCE**

首先，对于输入图首先考虑一个图神经网络来得到节点的表征。具体的，如果采用图卷积网络（GCN），其节点表征的更新公式如下：

$$
Z^{(l)} = σ(D^{−1/2} \widetilde{A} D^{−1/2}Z(l−1)W^{(l)})

, Z^{(l−1)} = [z_{i}^{(l−1)}]_{i∈I}, Z^{(0)} = X, 
\tag{3}
$$

在上式中节点表征的计算依赖于图中相邻的节点，从而将样本间的依赖关系建模了出来。通过 $L*$ 层图卷积之后，将最后一层的输出结果 $h_{θ}(x_{i},G_{x_{i}} ) = z_{i}^{{(L)}}$ 作为 logits 用于对节点标签的预测，即模型给出的预测分布可以写为：

$$
p(y | x, G_{x}) = \frac{e^{h_{θ}(x,G_{x})_{[y]}}}{\sum_{c=1}^{C}e^{h_{θ}(x,G_{x})_{[c]}}}
 \tag{4}
$$

通过将其与定义能量函数与概率密度之间关系的EBM模型联系起来，本文可以得到由GNN模型所诱导的能量形式 $E(x,G_{x}, y; h_{θ}) = -{h_{θ}(x,G_{x})_{[y]}}$，边缘化掉 $y$ 后得到 $E(x,G_{x},h_{θ}) = -log \sum_{c=1}^{C}e^{{h_{θ}(x,G_{x})_{[c]}}}$

这一能量函数对每个输入节点都能返回一个能量值，它可以衡量分类器对图中节点的置信度，即作为判别是否是 OOD 样本的依据。

对于分布内数据的监督训练损失：

$$
L_{sup} = E_{x,G_{x},y} ∼ D_{in}(-log p(y|x,G_{x})) = \sum_{i \in I_{s}}(-h_{θ}(x_{i},G_{x_{i}})_{[y_{i}]} + log \sum_{c=1}^{C}e^{h_{θ}(x_{i},G_{x_{i}})_{[c]}})
\tag{6}
$$

**2- CONSENSUS BOOSTING WITH ENERGY-BASED BELIEF PROPAGATION**

为了进一步的利用图结构产生的样本依赖性，本文提出了基于能量的信任传播，具体实现为将每个节点的能量值沿着输入图进行信息传递:

$$
E^{k} = \alpha F^{k - 1} + (1 - \alpha) D^{-1}AE^{k - 1}, \quad
E^{k} = [E^{k}_{i}]_{i \in I}, \quad
E^{0} = [E({x_{i},G_{x_{i}};h_{\theta}})]_{i \in I}\
\tag{7}
$$

这样做的好处是，可以使得分类器产生的置信度沿着图结构加强。由于图中相邻的节点通常可以看作来自相似的数据分布，当本文聚合相邻节点的能量值后得到的新能量值就更容易区分出来自不同分布的节点。

* **在 K 步传播后，使用最后的结果来衡量一个点是否是 OOD 点：**

$$
G(x, G_{x}; h_{\theta}) = 
\left\{
\begin{aligned}
1,\quad &if \widetilde{E}(x, G_{x}; h_{\theta}) \leq \pi \\
0,\quad &if \widetilde{E}(x, G_{x}; h_{\theta}) > \pi
\end{aligned}
\right.

\\
其中，\widetilde{E}(x, G_{x}; h_{\theta}) = E_{i}^{K}

\tag{8}
$$

**3- ENERGY-REGULARIZED LEARNING WITH BOUNDING GUARANTEES**

作为模型的扩展，本文继续考虑现有 OOD 检测技术中的另一种设置，即加入额外的 OOD 训练数据。在这种情况下，可以通过正则化损失 $L_{reg}$ 来添加能量间隙的严格约束，$L_{reg}$ 限制分布内数据的能量。损失函数为：$L_{sup} + λL_{reg}$

其中 $L_{reg} = \frac{1}{|I_{s}|}\sum_{i \in I_{s}}(ReLU(\widetilde{E}(x_{i}, G_{x_{i}}; h_{\theta}))-t_{in})^{2} + \frac{1}{|I_{o}|}\sum_{j \in I_{o}}(ReLU(t_{out} - \widetilde{E}(x_{j}, G_{x_{j}}; h_{\theta})))^{2} $

上述损失函数将使分布内样本的能量值更低。分布外样本的能量值更高。

本文称纯监督损失训练的模型为 $GNNS_{AFE}$ ，用额外能量正则化训练的版本为 $GNNS_{AFE++}$ 。

### 16_Learning Multimodal Data Augmentation in Feature Space_ICLR 2023_有代码

> 作者：Zichang Liu, Zhiqiang Tang, Xingjian Shi, Aston Zhang, Mu Li, Anshumali Shrivastava, Andrew Gordon Wilson

> 代码：https://github.com/lzcemma/LeMDA/

> 贡献

从多种模式（如文本、音频和视觉数据）中联合学习的能力，是智能系统应该具备的特征。虽然神经网络在利用多模态数据方面已经取得了很好的进展，但目前数据增强的巨大成功仍然仅限于图像分类等单模态任务。本文引入了 $LeMDA$，*Learning Multimodal Data Augmentation*，一种易于使用的、自动学习在特征空间中联合增强多模态数据，不约束模态的身份或模态之间关系的多模态数据增强方法。

> 方法

在多模态深度学习中，利用数据增强的最直接的方法是将成熟的单模态增强策略分别应用到每个相应的模态中。然而，这种方法可能会有问题，因为孤立地转换一种方式可能会导致与其他方式的不和谐。本文指出在设计一个对多模态数据增强的一般方法时存在两个**关键挑战**：

* 首先，多模态深度学习从一组不同的模式中获取输入。增强转换对于视觉和语言等一些模式是明显的，但对于其他模式（如感官数据）则不是；
* 其次，多模态深度学习包括一组具有不同跨模态关系的不同任务。一些数据集存在冗余或完全相关的模式，而其他数据集则为互补的模式。

框架：

将 $task$ 网络分为两个部分：$F(x) = F_{after}(F_{before}(x))$，$F_{before}$ 表示 fusion 前的层，$F_{after}$ 表示 fusion 后的层。

* 对于输入的训练样本 $x$，先获得每个模态的隐层特征 $\{z_{i}\}^{N}_{i=1} = F_{before}(x)$，（共有N种模态）；
* 以 $\{z_{i}\}^{N}_{i=1}$作为增强网络 $G$ 的输入，产生额外的潜层向量 $G(\{z_{i}\}^{N}_{i=1})$；
* 将  $\{z_{i}\}^{N}_{i=1}$ 和 $G(\{z_{i}\}^{N}_{i=1})$ 送入 $F_{after}$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230321230434352.png" alt="image-20230321230434352" style="zoom: 67%;" align="left"/><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230321230522418.png" alt="image-20230321230522418" style="zoom: 67%;" />

如上图，左：task network 的训练过程； 右：augmentation network 的训练过程

**task network 的训练目标**是找到 $min \quad E_{x∼X} (L(\widehat{y}) + L(\widehat{y_{G}})) \quad 其中 \widehat{y} = F_{after}(F_{before}(x)) , \quad \widehat{y_{G}} = F_{after}(G(F_{before}(x)))$

**augmentation network 的训练目标**是找到 $max \quad E_{x∼X} (L(\widehat{y_{G}})) + min \quad E_{x∼X} (L_{consist}(\widehat{y}, \widehat{y_{G}})), \quad 其中 L_{consist}(\widehat{y}, \widehat{y_{G}}) 表示 \widehat{y}和\widehat{y_{G}}之间的散度度量，如 KL 散度$

* **Confidence masking：**对于分类问题，本文只将 consistency 应用于最高概率大于阈值 α 的样本。因为如果 task network 不能做出一个自信的预测，那么这个预测就不太可能为 gt 提供一个很好的参考。
* **Design decisions：**LeMDA 的优势在于其简单、通用以及强大的性能。关于应该如何定义一致性正则化约束，以及应该在多大程度上应用它，本文探讨了可作为 基于 KL 散度的一种替代方案——最小化增强特征向量到原始特征向量的 $L_{2}$ 距离，并通过消融实验验证了基于 KL 散度的一致性约束具有更好的效果。

> 关于 augmentation network 架构的设计

本文使用变分自编码器 VAE 作为增强网络，并设计了两种实现方式：

* **MLP-VAE:** VAE 的 encoder 和 decoder 都是 MLPs，$\{z_{i}\}^{N}_{i=1}$ concate 之后作为输入；
* **Attention-VAE:** VAE 的 encoder 和 decoder 由 self-attention 和 feedforward 网络构成，$\{z_{i}\}^{N}_{i=1}$ 被看成是 N 个 tokens。

关于两种 fusion 方式：

* late fusion architectures：在后期融合中，来自每个模态的输入都由不同的 backbone 独立处理。不同 backbone 提供的表示在后面的层中（通常就在分类器层之前）融合在一起。这种设计可以直接适用于任何新的模态和任何多模态任务。**这是本文 focus 的形式**。
* early fusion architectures：在早期的融合中，网络结合来自所有模态的原始输入或 token embedding。早期的融合架构可以被设计为利用低级特征之间的交互作用，使其成为具有较强跨模态相关性的多模态任务的良好选择。

注：**1、**本文通过消融实验表明，Attention-VAE 性能不如 MLP-AVE，可能是因为 N 的值较小（本文中N一般为2或3），因此 Attention 不太 make sense。**2、**MLP-AVE 不太适合早期融合架构，因为 concate 后维度会很高从而带来巨大的计算量。

## 20230407

### 17_Addressing the Challenges of Open-World Object Detection_under review 20230337上传_被接收后上传代码

> 作者：David Pershouse, Feras Dayoub, Dimity Miller, Niko Sünderhauf

​				昆士兰科技大学，阿德莱德大学，昆士兰科技大学，昆士兰科技大学

> 代码：暂未上传

> 贡献：

针对 open-world 目标检测，本文总结了三个基本阻碍：

* **Class-agnostic region proposals**：模型必须能够定位并提出包含物体的区域，同时忽略仅部分覆盖物体或不包含物体的区域；
* **Unknown-aware classification**：模型需要识别 unknown objects 出现的概率；首先，需要有能力检测 unknown objects ，然后，需要尽量防止模型检测 unknown 的能力带来的开集错误（即将 unknown 识别为某个 known）；
* **Incremental learning with incomplete labels**：在训练当前任务时，只能接触当前任务包含的类别，因此模型需要克服因训练数据的不完备带来的 catastrophic forgetting 问题。

`<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230403191235478.png" alt="image-20230403191235478" style="zoom:67%;" align="left" />`                                                                                                                                                                                                                                                                                                                                                           并针对以上三个挑战，分别提出相应措施。即

* 使用基于 centerness 的 objectness score 来生成类不可知的区域建议，避免需要学习未知对象的分类器
* 使用基于 IoU 的 objectness score 和高斯混合模型，进行 unknown aware 分类并减少开放集误差
* 使用仅进行微调的增量学习方案，在损失计算过程中处理训练数据中未标记的已知对象而并忽略它们。

> 方法：

本文框架基于 Faster-RCNN 进行改进。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230403144005093.png" alt="image-20230403144005093" style="zoom:80%;" />

**Class-agnostic region proposals**

为了解决 **class-agnostic region proposals**, OW-RCNN 将 Faster R-CNN 的 foreground/background classifification head

替换为 localization quality regression head $R_{ctr}$。

box 回归头 $R_{box}$ 预测 anchor 中心到 gt box 的 $(l,t,r,b)$;

box 定位质量回归头  $R_{ctr}$ 预测  $R_{box}$ 输出的 box 的 centerness；训练时，基于一个随机选择的中心点落在 gt box 里的 anchor。

其中 centerness 定义为： $centerness = \sqrt{\frac{min (l, r)}{max (l, r)} × \frac{min (t, b)}{max (t, b)}}$

**Unknown-aware classifification**

不像之前的大多数工作，向分类器添加一个输出来处理未知类。本文的解决方案是，将 OW-RCNN 中的分类头的作用视为确定某个区域是否包含一个**已知对象 known object**或**其他对象 other**（背景和未知对象的组合）。

进一步，为了将识别为 other 的区域区分为 unknown object 和 background，本文添加了一个 class-agnostic box 回归头 $F_{agnbox}$ 和 一个 localisation-based 的回归头 $F_{iou}$，其中 $F_{iou}$ 用于预测 $F_{agnbox}$ 输出的 box 与 object 的 IOU。

并从定位质量头的输出计算一个 objectness score $s_{obj} = \sqrt{σ(R_{ctr}) × σ(F_{iou})}$

使用 Algorithm 1 将目标得分与其他头的输出结合起来，得到 每类得分 和 每个已知类、未知类以及背景类的预测边界框。

$θ_{obj}(0.69)$, $θ_{conf}(0.05)$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230403152720162.png" alt="image-20230403152720162" style="zoom:80%;" align="left"/>    为了减少 open-set error，防止将未知对象错误分类为已知类。在 **inference** 阶段，采用高斯混合模型建模  classifier’s logits 的 likelihood，将低 likelihood 检测视为未知。如 Algorithm 2：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230403154744365.png" alt="image-20230403154744365" style="zoom:80%;" align="right"/>

通过使用  object score 从背景中分离出未知对象，使用高斯混合模型减少开集误差，OW-RCNN解决了开放世界检测问题带来的第二个挑战——Unknown-aware classifification。

**Incremental learning with incomplete labels**

现有的 OWOD 检测框架使用范例重放方法来实现新类的增量学习，在上一个任务中创建的模型 $M_{t−1}$首先在新的训练数据 $D_{t}$ 上进行训练，然后在平衡的范例 $R_{t}$ 集上进行微调。结果显示，模型对前一个任务类的性能有强烈的偏好，当前任务类的性能只能达到先前已知类一半的水平。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230403160246368.png" alt="image-20230403160246368" style="zoom:80%;" align="right"/>本文的框架在第一个任务中使用整个数据集 $D_{1}$ 进行训练，而对后续任务只使用当前任务可见类别的范例集 $R_{n}$（**300 exemplars per class**）。遵循 ORE 的类均衡样例采样方式如 Algorithm 3。尽管与在所有数据上训练的模型相比，仍然存在显著的性能差距，但本文发现，在实验中，与之前的 OWOD 框架使用的方法相比，本文使用的训练数据集方式平衡了当前和以前任务类之间的性能，减少了训练时间，并提高了性能。

由于训练任务 $T^{n}$ 时，缺乏先前已知类的标签，若不加约束，模型很有可能

将它们当成未知类或背景类。因此，本文在计算 $L_{cls}$ 时排除掉判定为先前已知类中   $confifident \quad prediction \quad s_{cls} > θ_{cls}(0.5) \quad and \quad high \quad objectness \quad s_{obj} > θ_{obj}(0.69) \quad and 不与gt \quad overlap的$样本。

> 训练：

损失：$L =L_{ctr} + L_{box}+ 3L_{cls} + L_{clsbox} + L_{agnbox} + L_{iou}$

​			其中$L_{cls}$使用交叉熵损失，其他都是用 $l_1$损失

每训练完一次 task，高斯混合模型和相应的 $θ_{like}$ 都重新产生。（*D*1 for the fifirst task, *R**t* for the remainder），阈值 $θ_{like}$ 取每个类的训练数据生成的可能性的最小值。

注：之前 OWOD 工作的样例回放，**目前已知的所有类别**每个类别样本数50，本文使用**当前任务可见类别**的每个类别样本数300。

### 18_OpenCon: Open-world Contrastive Learning_TMLR 2023_有代码

TMLR: Transactions on Machine Learning Research

> 作者：Yiyou Sun, Yixuan Li

​				*University of Wisconsin-Madison*

> 代码：https://github.com/deeplearning-wisc/opencon

> 贡献：

开放世界环境下学习有效表征 effective representations 的挑战在于：

1. 在未标记数据 $D_{u}$ 中，已知类数据和 novel 类数据是混在一起的；
2. novel 类数据缺乏监督

本文提出  open-world contrastive learning (**OpenCon**) ，利用训练中未标记的数据，为所有已知类以及新颖类分别学习紧凑的特征表示。

> 问题设定：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230404191815241.png" alt="image-20230404191815241" style="zoom:80%;" />

（弱）监督学习：都只涉及 labeled data $D_{l}$，而不考虑 unlabelled data $D_{n}$；

**open-world semi-supervised learning**（本文问题设定）：

数据设定：

训练数据包含两部分$D = D_{l} ∪ D_{u}$

labeled data：$D_{l} = \{x_{i}, y_{i}\}_{i=1}^n$, $y_{i} \in Y_{l}已知$

unlabeled data：$D_{u} = \{x_{i}\}_{i=1}^{m}$, $x_{i} \in X 可能是 known \quad class，也可能是 novel \quad class$

将 label set 表示为$Y_{all}$，则 $Y_{l} \subset Y_{all}$，且 $Y_{n} = Y_{all} 去除 Y_{l}$

目标： learn distinguishable representations for both known and novel classes simultaneously

> 方法：open-world contrastive learning ( **OpenCon**)

**Learning from Wild Unlabeled Data**

本文提出基于原型的学习策略从未标记的数据中学习信息。

方法的关键在于，为所有类维护一个类别原型  $µ_{c}$， $c ∈ Y_{all}$，原型向量$ M = [µ_{1} ... µ_{c} ...] $随机初始化，并随着模型的训练进行更新。

* Prototype-based OOD detection

利用原型学习进行 OOD 检测，即将 $D_{u}$ 中的 novel data 与 known 区分开。对于 $D_{u}$ 中的任意样本 $x_{i}$ 通过计算其 embedding $\phi (x_i)$与已知类（$Y_l$）原型向量的相似度来判断是否是 novel 类。若某一样本嵌入和所有的已知类原型相距都远，则归为 novel。

$$
Dn = \{x_i|\underset{j∈Y_l}{max} \quad \mu_j^T · \phi(x_i) < λ\}
$$

训练时使用的域值 λ，可以基于标记数据 $D_l$ 来确定：首先计算 $D_l$ 中所有样本 $x_i$ 的 scores $\underset{j∈Y_l}{max} \quad \mu_j^T · φ(x_i)$，然后使用 score 的 p%分位数作为域值 λ。

* Positive and negative set selection

学习 novel 类紧凑的表示，关键就在于 positive set $P_n(x)$ 的构建。由于缺乏标签，本文提出利用预测标签 $\hat{y} = \underset{j∈Y_{all}}{max} \mu_j^T · \phi(x_i)$ 来选择 positive set。

对于从 $D_n$ 中选取的 mini-batch $B_n$，对其中的每个样本进行随机的两种增强得到 multi-viewed batch $\widetilde{B}_n$，并通过编码器获得相应的 embedding $A_n$，有 $|A_n| = 2|B_n|$。对于  $\widetilde{B}_n$ 中的样本 $x$，$P_n(x)$ 与 $N_n(x)$ 的选择方式如下：

$$
P_n(x) = \{z^{'} |z^{'}∈ {A_n/z}, \hat{y^{'}} = \hat{y}\}
\\
N_n(x) = A_n/z
$$

$z$ 是样本 $x$ 的$L_2$归一化 embedding， $\hat{y^{'}}$ 是 $z^{'}$ 的预测标签。

positiv set：the augmentation-based sample + any of the remaining samples with the same predicted label

损失：$L_{n} = \sum_{x \in \widetilde{B}_n} L_{\phi}(x; τ_n, P_n(x), N_n(x))$

* Prototype update

通过移动平均的方式更新类别原型

$$
\mu_{c} := Normalize(\gamma \mu_{c} + (1-\gamma)z), 
for \quad c = \left\{
\begin{aligned}
y(ground truth label),\quad\quad & if  \quad z \in D_{l}\\
\underset{j∈Y_n}{argmax} \quad \mu_j^T · z\quad\quad & if  \quad z \in D_{n}\\
\end{aligned}
\right.
$$

**Open-world Contrastive Loss**

$$
L_{OpenCon} = λ_{n}L_{n} + λ_{l}L_{l} + λ_{u}L_{u}
$$

## 20230414

### 19_GOOD: EXPLORING GEOMETRIC CUES FOR DETECTING OBJECTS IN AN OPEN WORLD_2023 ICLR_有代码

> 作者：Haiwen Huang 1,2,  Andreas Geiger 2,3,  Dan Zhang 1,4

1Bosch Lab, University of T¨ubingen
2Autonomous Vision Group, University of T¨ubingen
3T¨ubingen AI Center 4Bosch Center for Artificial Intelligence

> 代码：https://github.com/autonomousvision/good链接有问题，已发邮件询问（是license 还有点问题，弄好了就会公布）

> 贡献：

本文面向的问题是 open-world class-agnostic object detection，在 OLN （20230224-6）的基础上做了改进。

由于 RGB-based model 主要依赖于**外表的相似性**（如纹理）来检测 novel 类，会对训练中的已知类数据 over fitting。为此，本文提出模型 GOOD（Geometry-guided Open-world Object Detector），通过引入**几何线索** （geometric cues）例如 **深度（depth）以及法线（normals）**，使模型能够关注到物体的形状（深度）和相对位置变化（法线），从而在训练过程中为 novel objects 打伪标签来提高模型检测 novel 类的能力。另外，该方法，能够使用较少的训练类别数就能达到较好的性能。其中几何线索是通过现成的预训练模型 Omnidata models 提取的。

> 方法：

首先，基于预测好的几何信息发现的 nove objects 来训练一个 object proposal 网络；

其中 top-k 个 nove objects 的 proposals 会被作为 pseudo boxes 加入 open-world object detector的训练。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230410160653500.png" alt="image-20230410160653500" style="zoom:80%;" />

> 实验：

做了跨类别以及跨数据集泛化：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230410162428375.png" alt="image-20230410162428375" style="zoom:55%;" align="left"/><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230410162504477.png" alt="image-20230410162504477" style="zoom:55%;" align="right"/>

> 消融：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230410162802291.png" alt="image-20230410162802291" style="zoom:80%;" />

如图5(a)，RGB-based object proposal 网络表现比depth-based 以及 normal-based都好，这表明深度和法线映射不能简单地替代RGB图像来进行目标检测，本文是将它们结合一起使用。

如图5(b)，数据增强对有助于抵消伪标签中的噪声。然而，使用自动增强在基类 gt 注释上训练OLN，只有在召回基类对象方面有改进（从58.4%到61.7% $AR_{Base}@100$），但在新对象检测中则变差。这表明，通过随机调整、裁剪和翻转来实现数据增强不能改善跨类别的泛化。

如图5(c)，训练中更多的基类能让模型学习物体更一般的性质，以便更好地检测新对象。另外，在基类数量只有一半的$AR_{N}@100$，例如，39对80，GOOD可以实现与OLN相似的性能。这表明，GOOD在学习一般的 objectness 方面更有效，并且更不容易对基类进行过拟合。

### 20_HOW TO EXPLOIT HYPERSPHERICAL EMBEDDINGS FOR OUT-OF-DISTRIBUTION DETECTION? _2023 ICLR _有代码

> 作者：

Yifei Ming1, Yiyou Sun1, Ousmane Dia2, Yixuan Li1
		Department of Computer Sciences, University of Wisconsin-Madison1 Meta2

> 代码：https://github.com/deeplearning-wisc/cider

> 贡献：

关于 OOD 检测，之前的模型使用现成的对比损失能够满足对 ID（分布内）类别进行区分，但是对 OOD 样本表现不佳。

因此本文利用**超球面嵌入**，提出一个对于 OOD 任务的表征学习框架 **CIDER**（**C**ompactness and **D**isp**E**rsion **R**egularized），通过两个损失共同提升 ID-OOD 的区分性：

* 分散损失（ a dispersion loss）：增大不同类原型之间的角距（angular distance）
* 紧凑损失（ a compactness loss）：使样本靠近相应的类原型

注：角距，在数学(特别是几何学和三角学)和自然科学(包括天文学、地质学等等)，从不同于两个点物体的位置（即第三点）观察这两个物体，由观测者指向这两个物体的直线之间所夹角度的大小。角距离(或分离)与角度本身是同义的，但意义却是对两个天体(对恒星，是当从地球观测)之间线距离的建议(通常是很大或未知的)。（百度）

> 方法：

模型由两部分组成：

* 一个编码器 $f$，将增广的输入 $\tilde{x}$ 映射到一个高维的特征嵌入 $f(\tilde{x})$，即 $X → R^{e}$
* 一个投射层 $h$，将高维的特征嵌入 $f(\tilde{x})$ 投射到低维特征嵌入 $\tilde{z} := h(f(\tilde{x}))$

训练损失基于归一化的特征嵌入层$z := \tilde{z}/∥\tilde{z}∥_{2}$做。

![image-20230411202530012](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230411202530012.png)

**MODEL HYPERSPHERICAL EMBEDDINGS**

超球面嵌入可以自然的使用 von Mises-Fisher (vMF) 来建模，对于单位向量 $z ∈ R^d$ 是类别 $c$ 的概率密度为：

$$
p_{d}(z; \mu_{c}, κ) = Z_{d}(κ) exp (k \mu_{c}^{T} z)
$$

其中$\mu_{c}$ 是具有单位范数的类原型

$k>0$ 表明分布离平均方向 $\mu_{c}$ 的紧凑性，κ越大，分布就越集中在μ附近，反之则越分散（越接近球面上的均匀分布）。因此，κ也被形象地称为“凝聚度（concentration）”参数。当κ=0 的时候，vMF分布是球面上的均匀分布。

$Z_{d}(κ)$ 为归一化因子

因此，嵌入向量 $z$ 被分配给类别 $c$ 的概率为：

$$
P(y = c|z; \{κ, \mu_{j}\}^{C}_{j=1}) &=& \frac {Z_{d}(κ) exp (k \mu_{c}^{T} z)}{\sum_{j=1}^{C}Z_{d}(κ) exp (k \mu_{j}^{T} z)} \\
&=&  \frac { exp (\mu_{c}^{T} z /τ)}{\sum_{j=1}^{C} exp (\mu_{j}^{T} z /τ)}
$$

其中$k = 1/τ$

**OPTIMIZE HYPERSPHERICAL EMBEDDINGS**

优化目标：

1. 每个样本以高概率被分配给正确的类
2. 不同的类彼此相距很远

对于目标1，使用最大化似然函数（MLE）的方式，即最小化负对数损失：$L_{comp} = − \frac{1}{N} \sum_{i=1}^{N}log\frac { exp ( z^{T}_{i} \mu_{c(i)} /τ)}{\sum_{j=1}^{C} exp (z^{T}_{i} \mu_{j} /τ)}$，它鼓励样本与它对应的类原型靠的更近。

对于目标2，本文提出 dispersion loss ，最大化不同类原型之间的距离：$L_{dis} = \frac{1}{C} \sum_{i=1}^{C} log\frac{1}{C-1} \sum_{j=1}^{C} 1\{j \neq i\} e^{\mu_{i}^{T} \mu_{j} / \tau}$

总损失：$L_{CIDER} = L_{dis} + λ_{c}L_{comp}$

消融实验：

两项损失的消融可以看出，类间分散是提高 OOD 检测性能的关键。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230411211433841.png" alt="image-20230411211433841" style="zoom:80%;" />

可视化：

![image-20230411220130387](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230411220130387.png)

**CHARACTERIZING AND UNDERSTANDING EMBEDDING QUALITY**，作者提出使用 **类间离散度**和**类内紧致性**、**可分离性** 来描述和理解嵌入质量：

$$
↑ Dispersion(\mu) = L_{dis} = \frac{1}{C} \sum_{i=1}^{C} \frac{1}{C-1} \sum_{j=1}^{C}  \mu_{i}^{T} \mu_{j} 1\{j \neq i\}
$$

$$
↓ Compactness(D_{tr}^{T}, \mu) = \frac{1}{C} \sum_{j=1}^{C} \frac{1}{n} \sum_{i=1}^{n} z_{i}^{T}\mu_{j}1\{y_{i} = j\}
$$

$$
↑ Separability = \frac{1}{D_{test}^{ood}}\sum_{x \in D_{test}^{ood}} max_{j \in [C]}z_{x}^{T} \mu_{j} - \frac{1}{D_{test}^{in}}\sum_{x' \in D_{test}^{in}} max_{j \in [C]}z_{x'}^{T} \mu_{j}
$$

![image-20230411214139403](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230411214139403.png)

## 20230421

### 21_Learning to Name Classes for Vision and Language Models_CVPR 2023_有代码

> 作者：

​				Sarah Parisot， Yongxin Yang， Steven McDonagh

​				Huawei Noah’s Ark Lab

> 代码：https://gitee.com/mindspore/models/tree/master/research/cv/

> 贡献：

大规模视觉-语言模型虽然有较好的 zero-shot 能力，但仍然面临两个重要挑战：1、对于 hand-crafted 的类别名称敏感；2、难以扩展到新的、小规模数据集。因此本文提出，利用可获得的数据，为每一个类别，**从 visual content 中学习一个优化的 word embedding**。通过在**冻结的大模型**上学习新的单词嵌入，不仅能够保留新类的零射击能力，还能轻松地使模型适应新的数据集，并调整潜在的 non-descriptive 或 ambiguous class names 错误。本文提出的方法能方便的应用到图像分类以及目标检测的任务中去。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230417095239943.png" alt="image-20230417095239943" style="zoom:67%;" />

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230414100859604.png" alt="image-20230414100859604" style="zoom: 67%;" align="left"/> 模型输入：image-text 对：$x = \{I, T\}$，$T$ 中包含 N 个类别，$T= [t_1,t_2,...,t_N]$，第 $i$ 个类别的文本$t_i =[prompt \quad prefix]+[CLASS]+[prompt \quad suffix]$，其中前缀和后缀在所有类别间共享。

 **Learning optimal class names**

 本文认为预训练好的识别模型的性能和用于表示图像类别的类名（class names）直接相关。选择不能很好反映对象的视觉外观的类名对模型的识别能力有很大影响。

 因此本文提出，通过从图片中学习 class specific 的 word embedding，来缓解模型对于 hand crafted class names 的敏感性。

 首先，预定义一个预训练好的 word embedding 矩阵 $E \in R^{V*F}，V是vocabulary-size，F是特征维度$；

 用可学习的 $E^{l} \in R^{N*F}，（N是数据集中类别数量）$ 来扩充 $E$；





> **Interpretability**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230417101856539.png" alt="image-20230417101856539" style="zoom:80%;" align="left"/>   

  

  

两个有意思的趋势：

 1、模型倾向于将单词嵌入从英式英语表示调整为美式英语表示，（CLIP是基于美式英语训练的），这体现了本文模型跨语言的能力。

 2、细粒度的类别表示倾向于学习成更接近于超类别的词类型（ e.g. vegetable, musical，instrument），这表明本文模型还利用了跨类的相似性来学习罕见类的表示。











### 22_Prompt Pre-Training with Over Twenty-Thousand Classes for Open-Vocabulary Visual Recognition_xx_有代码

> 作者：

Shuhuai Ren, Aston Zhang, Yi Zhu, Shuai Zhang, Shuai Zheng, Mu Li, Alex Smola, Xu Sun

Amazon Web Services， National Key Laboratory for Multimedia Information Processing, School of CS, Peking University

> 代码：https://github.com/amazon-science/prompt-pretraining

> 贡献：

将视觉识别任务（如图像分类、目标检测和语义分割）定义为**语言引导的视觉识别**或视觉和语言问题已经成为一种新的规范。文本提示符（text prompt）作为类名的上下文，在语言引导的视觉识别模型中起着关键的作用。一个好的提示应该能全面地表达视觉类别的语义，以更好地引出视觉语言模型（VLM）在预训练阶段所学到的知识。目前有两种流行的提示类型：硬提示（**hard prompts**，例如，a
photo of a [CLASSNAME]）和软提示（**soft prompts**）。软提示在下游任务上往往比硬提示更有效和稳定，因为软提示可以在给定一些输入数据时进行微调。然而，传统的软提示调优方法通常使用有限数量的类标签**对任务特定的数据集**进行微调软提示（如 CoCoOP、MaPLe），这使得很难泛化到新的类和任务。

本文致力于学习一个涵盖广泛的视觉概念，且是 task-agnostic 的通用的提示。本文提出 PrOMpt Pre-training (POMP) 模型，通过在有两万多种类别的 ImageNet-21K 数据集上进行训练，将通用的视觉识别信息压缩进 soft prompts，一旦预训练完成，所获得的  **universal prompt** 可以：1、很容易地应用于下游数据集，以提高 zero-shot 设置下的模型性能；2、同时兼容区 region-level 和 pixel-level 的视觉模式，使其可用于各种视觉任务，如目标检测和语义分割。

如果使用 CoOp 的方式来学习如此大量的 prompt 需要 300多 GB 的 GPU 内存，在 POMP 中，本文通过 **local contrast 的类采样策略将 GPU 内存的消耗控制在了 16 GB 以内**；此外，使用 **local correction strategy** 提高预训练的 prompt 的泛化能力，并减少类抽样引起的偏差。

> 方法：

对于区分每一个类，需要分配近15 MB 的 GPU 内存来保持整个冻结编码器的状态（Transformer-base、12层），并将梯度通过最后一层传播到第一层。 prompt tuning 的计算成本和缓存成本与类的数量N成正比。对于 ImageNet-21K 则需要 15 MB * 21K （大于 300 GB）。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230417150258002.png" alt="image-20230417150258002" style="zoom:80%;" align="left"/> **Local Contrast**

 为了缓解显存占用过大的问题，本文提出将对比学习的范围**从全局缩小到局部**，只从全类集的一个子集中识别输入图像的 gt 类。在每个训练步骤中对类子集进行采样，让模型在不断变化的类别集中进行区分，并逐步重建所有类别之间的关系。

 对于一张图片，使用均匀分布采样方式采样 K 类（K 远小于总类别数 N，K = 1 个 gt 类 + K-1 个负类，$p = 1/(N − 1)$），这样显存存消耗就降到了原来的 K/N。

 **Local Correction**

为了减少类抽样的偏差和提高模型性能，本文添加一个局部校正项 $m_i$ 到采样的负类 logits 中 $x^{T}w_{i}^{(\Theta)}/\tau(i \neq y )$。因此，POMP的最终预测概率为：$\widetilde{P}(y|x;\Theta) = \frac{exp(x^{T}w_{y}^{(\Theta)}/\tau)}{exp(x^{T}w_{y}^{(\Theta)}/\tau) + \sum_{i∼N}exp(x^{T}w_{i}^{(\Theta)}/\tau + m_{i})}$ 其中 $m_{i} = − log \frac {K-1}{N-1}$.

 $m_{i}$ 是个正数，因此能够得到一个更严格的决策边界：

 $C_{+} : x^{T}w_{y}^{(\Theta)}/\tau \geq x^{T}w_{i}^{(\Theta)}/\tau + m_{i}, i \neq y$



> 实验：

在分类、分割以及检测任务上分别做了跨类、以及跨数据集的 open vocabulary 的实验，性能均比之前的 SOTA 好。

> 消融：

![image-20230417163743103](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230417163743103.png)

> 对 POMP 所学习到的特征空间的探索

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230417163910959.png" alt="image-20230417163910959" style="zoom:80%;" align="left"/> 图中 POMP 的圆圈位于左下角，颜色最轻，表明在跨数据集设置下，损失相对较小，性能最好。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230417164354429.png" alt="image-20230417164354429" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230417164422224.png" alt="image-20230417164422224" style="zoom:67%;" />

 总之，本文的预训练提示不仅确保了图像和 gt 的对齐，而且还将类特征分散在表示空间中，从而提高了模型的泛化性和鲁棒性。

## 20230428

### 23_Discovering Objects that Can Move_CVPR 2022_有代码

> 作者：

Zhipeng Bao1, Pavel Tokmakov2, Allan Jabri3, Yu-Xiong Wang4, Adrien Gaidon2, Martial Hebert1

1CMU 2Toyota Research Institute 3 UC Berkeley 4 UIUC

> 代码：https://github.com/zpbao/Discovery_Obj_Move/

> 贡献：

本文研究对象发现的问题——将没有手动标签的对象从背景中分离出来。现有的方法利用外观线索，如颜色、纹理和位置，将像素分组为类似对象的区域。然而，由于仅依靠外观，这些方法不能从杂乱的场景中从背景中分离出来。因为对象的定义本质上是模糊的和依赖于上下文的，为了解决这种模糊性，本文选择关注动态对象——可以独立移动的实体。然后，本文将最近的基于自动编码器的无监督物体发现框架从简单合成图像扩展到复杂的现实世界场景。为此，本文简化了它们的架构，并**使用来自一般运动分割算法的弱学习信号**来增强生成的模型。本文的实验表明，尽管只捕获了一小部分移动对象，但这个信号足以推广到分割动态对象的移动和静态实例。

> 方法：

预备知识：**SlotAttention**

给定一张图片 $I \in R^{H*W*3}$，首先送入 encoder CNN 获得隐层表示 $H = f_{enc}(I) \in \mathbb{R}^{H'*W'*D_{inp}}$，然后经过注意力模块将 $H$ 映射到一系列（K个）特征向量中，称为 slots $S \in \mathbb{R}^{K*D_{slot}}$。其中每个 slot $S_{i} \in S$广播到2维 grid ，然后分别使用 decoder CNN 解码：$O_{i} = f_{dec}(S_{i}) \in \mathbb{R}^{H*W*4}$，其中第 4 维表示 alpha mask $A_{i}$，将前三个通道表示为 $I_{i}’$，完整的重构图片则通过 $I' = \sum_{i}A_{i}*I_{i}’$获得。

其中的注意力模块是上述方法的关键部分，类似于 Transformer，使用迭代注意力方式将 $H$ 映射到 $S$。

注意力权重：$W=\frac{1}{\sqrt{D}} k(H) \cdot q(S) \in \mathbb{R}^{N \times K}$，$N = H' × W'$

更新值：$U = W^T v(H) ∈ \mathbb{R}^{K \times D}$，$W^T$ 已归一化

在每一step l ，更新 slot：$S_l = update(S_{l−1}, U_l)$

与经典 Transformer 的不同点在于 slot 是随机初始化的。

类似于 DETR，slot 相当于 object query，解码后的 slot 会对应到图片中的特定区域，如有 object 的地方。

 **A framework for object discovery in videos**

对于视频中的目标发现，首先将一系列视频帧送入 encoder CNN 获得相应的嵌入表示，然后使用 ConvGRU 获得视频编码 $H_{t}’$，接下来要将视频编码  $H_{t}’$ 映射到 slot，如果采用原始的 SlotAttention 方法，那么对于 $T$ 个视频帧进行 $L$ 次迭代的话，$T \times L$次的 attention 操作存在**计算量很大**而且可能加剧**梯度消失**的问题，因此本文只执行一次注意力操作来计算 slot 状态：$S^t = W^{t^T}v(H'^{t})$，注意力矩阵$W^{t^T}$使用之前帧的slot state $S^{t-1}$得到，对于第一帧，使用一个可学习的 $S^{0}$。

接下来就是对 slot states $S^t$ 进行重构。但是，如果为每个插槽计算一个完整的图像重建**特别耗内存**，特别是对于大分辨率的帧。因此，本文将 slot 解码和 slot 重组的顺序调换，即先重组再解码。在实验中发现，虽然先重组再解码的方法在目标发现能力方面不如原来的方式，但是原来的方式在从简单/合成图片泛化到更实际复杂的图片上的表现并不好。接下来，介绍加入运动先验的方法，它能提供很强的学习信号并和先重组再解码的方式很好地配合。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230424105738101.png" alt="image-20230424105738101" style="zoom:80%;" />

 **Incorporating independent motion priors**

假设对于每个视频都有instance-level 的运动分割掩码 $M = {M^1, M^2, ..., M^T }$, $M^t = {m_1, m_2, ..., m_{C^t} }$，$C^t$为 frame t 中分割出的移动对象的数量。

本文提出使用运动分割掩码指导 slot 注意力映射。首先找出预测和运动掩码之间的最优二分图匹配：$\hat{\sigma} = argmin_{\sigma} \sum_{i=1}^K L_{seg}(m_{i}^{t},W_{:,\sigma(i)}^{t})$;

那么 motion supervision 目标：$\mathcal{L}_{\text {motion }}=\sum_{i=1}^K \mathbb{1}_{\left\{m_i^t \neq \emptyset\right\}} \mathcal{L}_{\text {seg }}\left(m_i^t, W_{:, \hat{\sigma}(i)}^t\right)$

其中 $L_{seg}(m,W) = \sum_{j=1}^{N}-m_{j}log(W_{j})-(1-m_{j})log(1-W_{j})$

**Loss function and optimization**

总损失：$L=L_{recon} + \lambda_{M}L_{motion} + \lambda_{T}L_{temp}$

其中，$\mathcal{L}_{t e m p}(S)=\sum_{t=1}^{T-1}\left\|\mathbb{I}-\operatorname{softmax}\left(S^t \cdot\left(S^{t+1}\right)^{\mathbf{T}}\right)\right\|$是 temporal consistency regularization term。

### 24_Object Discovery from Motion-Guided Tokens_CVPR 2023_有代码

> 作者：

Zhipeng Bao1, Pavel Tokmakov2, Yu-Xiong Wang3, Adrien Gaidon2, Martial Hebert1

1CMU 2Toyota Research Institute 3UIUC

> 代码：https://github.com/zpbao/MoTok/

> 贡献：

在这项工作中，本文增加了自动编码器表示学习框架的两个关键组件：**运动引导**和**中间层特征token化**。本文引入了一种新的Transformer 解码器，表明通过motion-guided vector quantization他们的优点可以结合。本文的方法使可解释的特定于对象的中间层特征的出现成为可能，展示了运动引导（无标记）和量化（可解释性、内存效率）的好处。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230424163238524.png" alt="image-20230424163238524" style="zoom:80%;" />

**Slot Decoders**

Slot Decoders 的目标是将 slot 表示$(S^t，W^t)$映射到重建空间的二维特征映射$F^t$ ，本文提出四种解码方式：

1. **Linear Decoder**  根据注意掩模$W^t$直接将 slot 特征 $S^t$ 映射到相应的位置

   $$
   F_{linear}^{t}(x) = \frac{\sum_{i=1}^{K}S_{i}^{t}(x)W_{i,x}^{t}}{\sum_{i=1}^{K}W_{i,x}^{t}}
   $$
2. **CNN Decoder** 再线性的基础上增加两个卷积层

   $$
   F_{CNN}^{t} = CNN(\frac{\sum_{i=1}^{K}S_{i}^{t}W_{i,:}^{t}}{\sum_{i=1}^{K}W_{i,:}^{t}})
   $$
3. **Transformer Decoder**

   $$
   F_{transformer}^{t} = Transformer(P,S^{t},S^{t})
   $$

   $P$ 是 2维 position embedding

   与前两个线性解码器相比，Transformer 进一步考虑了 slot 特征与输入查询之间的全局连接，从而形成更强大的特征图。然而，一个明显的限制是，Transformer 对输入的位置查询应用自注意力，这是1)冗余的，因为位置嵌入本身是可学习的，2)计算效率不高，因此限制了整个模型的可扩展性。为了解决这一限制，本文进一步提出了感知器解码器。
4. **Perceiver Decoder**

   <div>
       <center>
       <img src="typora-user-images\image-20230424165623454.png"
            alt="Algorithm 1"
            style="zoom:80%"
        />
       <br>
       Algorithm 1
       </center>
   </div>

**Reconstruction Space**

在获得 2 维特征图 $F^{t}$，接下来就是使用 CNN-based 解码器将其解码到重构空间。关于重构空间有四种选择（如 Figure 2 最右侧显示）。 RGB 空间包含的信息最多，但也最难解决 object/background ambiguity 问题；flow 和 depth 空间更具结构性但不如 RGB 空间得信息量大，另外， flow 空间无法捕捉到不懂得物体，depth 空间不好区分互相挨着的物体。因此，本文提出 VQ-space，能够端到端训练而且兼具结构性以及信息量。

**Vector-quantized reconstruction space**

与其他三个重建空间不同的是，这里本文不是直接预测重建，而是监督特征图 $F^t$ 来匹配 VQ-VAE 的潜在嵌入空间。

首先，定义一个潜在嵌入空间 $E = \{e_{i} \in \mathbb{R}^{d_{vq}}|i=1,2,...,M\}$，给定一张图片 $I$,首先得到它的编码 $z_{e}^{t} = Encoder_{VQ}(I^{t})$；然后离散的潜在变量 $z$ 由最近邻算法得到：$z_{q}^{t}(x) = e_{k},\quad where\quad k = argmin_{j}\|z_{e}^{t}-e_{j}\|_{2}$（x为任意的二维坐标）；最后，重构的图片 $\hat{I}^{t} = Decoder_{VQ}(z_{q}^{t})$

VQ-VAE 的优化目标：$L_{VQVAE} = logP(I^{t}|z_{q}^{t}) + \|sg[z_{e}^{t}] - z_{q}^{t}\|_{2} + \|sg[z_{q}^{t}] - z_{e}^{t}\|_{2}$, (*sg*[*·*] is the stop-gradient operation)

使用量化特征图 $z_{q}^{t}$ 作为 slot feature map $F^{t}$ 的目标信号，因此最终的优化目标为：$L_{VQ} = L_{VQVAE} + \|sg[F^{t}] - z_{q}^{t}\|_{2} + \|sg[z_{q}^{t}] - F^{t}\|_{2}$

**Motion-guided token representation**

上述公式中 $ \|sg[F^{t}] - z_{q}^{t}\|_{2}$ 使 slot 学习中的运动信号通过 slot 解码的输出共同优化 token space。

**Optimization**

为了使 VQ-space 更具结构性，增加对比约束：$L_{contrastive} = \|\mathbb{I} - softmax(E \cdot E^T\|$

最终的优化损失：$L = \lambda L_{motion} + L_{recon} + \lambda_{c} \mathbb{1}_{VQ}L_{contrastive}$, 当使用 VQ-space 重构时，$L_{recon} = L_{VQ}$

## 20230505

### 25_Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection_CVPR 2023_有代码

> 作者：

Luting Wang1  Yi Liu1  Penghui Du1  Zihan Ding1  Yue Liao1*  Qiaosong Qi2  Biaolong Chen2  Si Liu1

1Institute of Artificial Intelligence, Beihang University 2Alibaba Group

> 代码：https://github.com/LutingWang/OADP

> 贡献：

之前采用知识蒸馏从预先训练好的视觉和语言模型（PVLMs）中提取知识，并将其转移到检测器中的方法，由于非自适应裁剪方案和单级特征蒸馏过程，在知识提取过程中存在信息破坏和知识传递效率低下的问题。因此，本文提出 Object-Aware Distillation Pyramid (OADP) 框架，其由 Object-Aware Knowledge Extraction (OAKE) 模块 与 Distillation Pyramid (DP) 机制组成。从 PVLMs 中提取知识时， **OAKE** 自适应的调整 object proposals，并使用 object-aware mask attention 获得精确、完整的 object knowledge。**DP** 通过引入 global、block 蒸馏获取更全面的知识迁移，以弥补 object 蒸馏中缺少的相关信息。

**OVD Benchmarks**

 根据训练数据划分：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230418101347115.png" alt="image-20230418101347115" style="zoom:80%;" align="left"/>  Vanilla OVD (V-OVD)，仅使用特定类别的目标检测数据信息

 Caption-based OVD (C-OVD)，

 Generalized OVD (G-OVD)

 Weakly Supervised OVD (WS-OVD)

> 方法：

 **Object-Aware Distillation Pyramid (OADP)**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230418173029679.png" alt="image-20230418173029679" style="zoom:80%;" />

本文基于 Faster-RCNN 架构，首先使用 CLIP 从图片中提取知识，然后迁移到检测器中。因此提出  Object-Aware Knowledge Extraction (**OAKE**) 模块，通过在 CLIP visual encoder $V$ 中添加一个 [OBJ] token 来获取 region proposals 中的信息。为了获得更有效的知识迁移，本文提出蒸馏金字塔结构 （Distillation Pyramid(**DP**)），包括 object distillation module $M^O$,  distillation module $M^B$, 以及 global distillation module $M^G$，相应的蒸馏损失为 $L^O, L^B, L^G$。

总的训练损失为：

$$
L^{all} = L + w^O · L^O + w^B · L^B + w^G · L^G
$$

$L$ 为 RCNN 损失。

使用 $M^{O}$ 校准 $P_{C}(p, c)$ 的过程如下，类似于 R-CNN，$M^{O}$ 提取 proposal embeddings $E^O = \{e^{O}_{p}\}_{p∈P} ⊂ R^{d}$ 并计算 logits：

$$
l^{O}(p, c) =\frac{e^{O}_{p} · t_{c}}{||e^{O}_{p}|| · ||t_{c}||}  \tag{5}
$$

$$
P_{C}^{O}(p,c) = \frac{exp(l^{O}(p,c))}{\sum_{c' \in C} exp(l^{O}(p,c'))} \tag{6}
$$

$t_c$ is the category embedding of c.

校准概率 $P_{C}^{cal}(p, c)$：

$$
P_{C}^{cal}(p, c) = 
\left\{
\begin{aligned}

& (P_{C}(p, c))^{\lambda} \cdot (P_{C}^{O}(p, c))^{1-\lambda}, & c \in C^{B} \\
& (P_{C}(p, c))^{1-\lambda} \cdot (P_{C}^{O}(p, c))^{\lambda}, & c \in C^{N} \\
& 1 - \sum_{c' \in C}P_{C}(p, c'),  & c = bg

\end{aligned}
\right.
\tag{7}
$$

$\lambda = 2/3$

**Object Distillation**

由于对信息的全面性和尽量少噪点的权衡，现有方法仅能获得次优的 $\widetilde{\epsilon}^{O}$，例如，当非方形建议区域直接传递给CLIP时，$V$ 中的中心裁剪操作将裁剪出对象的信息部分，导致对对象的结构知识不完整。另一方面，如果建议区域是方形的或被扩大了，建议区域将包含更多的环境上下文，这可能会破坏建议区域的嵌入。为了获得更精确的  $\widetilde{\epsilon}^{O}$ ，本文提出 **OAKE** 模块。

对于 proposal $p \in P$，其变换后的 $p'$ 是以为 $s = \sqrt[2]{r × p_h × p_w}$ 边长的方形，其中 $p'$ 的中心和 $p$ 的一样（在 $p'$ 超出图片边界时会被改变）。

上图中 $I_{O} = \{I_{p'}\}_{p' \in P'}$，$I_{p'}$ 首先映射到 tokens $X \in R^{N_{x}*d_{x}}$，其中 $X _{1:N_{x}-1}$为图像的patch tokens，$X _{N_{x}}$ 是 [CLS] token，本文增加了 [OBJ] token 并用 [CLS] token 初始化  [OBJ] token，得到 $X' = [X;x_{[OBJ]}] \in R^{(N_{x} + 1)*d_{x}}$，另外通过掩码机制约束 [OBJ] token 与其他 token 的交互：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230418214711915.png" alt="image-20230418214711915" style="zoom:67%;" />，这样 [OBJ] token 只会关注与原始 proposal 重叠的 patch token。

（为什么不直接把 [CLS] 改造成 [OBJ]，而是新增加一列呢？）

**Global and Block Distillation**

只使用 object distillation ，模型缺乏对不同 proposals 之间的关系的全面理解，因此本文提出使用 Global Distillation 来学习全局信息。由于 CLIP 倾向于忽略图像中的非显著信息（如背景或对象的突出属性）。这些信息对于检测可能很有价值，因此，本文提出 Block Distillation 来补充 Global Distillation 中缺失的知识。

**Pseudo Label Generation**

为了探索本文模型 OADP 在 G-OVD 设置下的性能，本文使用伪标签策略

$l^{PL}(p, c) =\frac{\widetilde{e}^{O}_{p} · t_{c}}{||\widetilde{e}^{O}_{p}|| · ||t_{c}||}$, $P_{C}^{PL}(p,c) = \frac{exp(l^{PL}(p,c))}{\sum_{c' \in C} exp(l^{PL}(p,c'))}$, 这里的 $C$ 包含所有基类+新类。由于 $P_{C}^{PL}(p,c)$不能反应 proposal 的定位质量，因此，将 confidence score 设置为 $S_{C}(p, c) = P^{PL}_C (p, c)^{γ} · o_{p}^{(1−γ)}$，$o_{p}^{(1−γ)}$ 为来自 RPN 的对象分数。$S_{C}(p, c)$ 反映了 proposal 精确定位类别c的一个实例的概率。最后，对新的类别应用  class-wise NMS来得到伪标签。

### 26_Prompt-Guided Transformers for End-to-End Open-Vocabulary Object Detection_xx_暂无代码

训练：在训练阶段，只有基类数据可用，类提示的数量根据当前batch中可见的基类的数量变化

测试：在测试阶段，类提示包括所有基础类和新类

> 作者：

Hwanjun Song, Jihwan Bang

AWS AI Labs, NAVER Cloud

> 代码：

> 贡献：

本文提出 Prompt-OVD ，利用 CLIP 编码的类别嵌入作为 prompt，指导 Transformer 检测器检测基类以及新类目标。其中使用 **RoI-based masked attention** 和 **RoI pruning techniques** 更好地利用 CLIP 分类的 zero-shot 能力，从而提升模型的检测能力。

> 方法：

首先回顾 DETR 的检测流程：

1. 用 L 层 Transformer Encoder 编码作为 patch 输入的图像（$I_{0} = [patch_{1}, ..., patch_{n}] \in \mathbb{R}^{n \times d}$）: $I_{L} = Encoder(I_{0}) \in \mathbb{R}^{n \times d}$
2. Transformer Decoder 以 query 和 $I_{L}$ 作为输入：$O_{L} = Decoder(Q_{0}, I{L}) \in \mathbb{R}^{n \times d}$
3. 得到的输出送入三层 FFNs 获得边框和类别的预测：$\hat{B}_{det} = FFN_{3-layer}(O_{L}) \in \mathbb{R}^{m \times 4} \quad \hat{P}_{det} = Linear(O_{L}) \in \mathbb{R}^{m \times k}$, k 是类别数
4. 使用匈牙利匹配

**Prompt-OVD**：首先改装 DETR 的 Decoder 部分，使用 CLIP embeddings 作为类别提示，然后使用 RoI-based masked attention 和RoI pruning 进一步增强对于 CLIP 的利用。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230426150139013.png" alt="image-20230426150139013" style="zoom:80%;" />

第一步：将 Decoder 改装成 OV的

类似于  OV-DETR，使用 CLIP 编码的图片$e_{img}$或文本$e_{txt}$嵌入作为提示，以下将两者统称为类别提示 class prompt，这些提示通过自注意力操作实时地提示 object queries 基于提示去检测。

做法：

将 CLIP embedding 加入到 Decoder 的每层自注意力中：$Q_{l}^{’} = SelfAttn_{l}([\mathbb{F} _{proj}(e_{clip}^{mod});Q_{l}]) \in \mathbb{R}^{m \times d}, mod \in \{txt, img\}$，然后通过同层的交叉注意从 $I_{L}$ 中聚合关键信息：$O_{l} = CrossAttn_{l}(Q_{l}^{'},I_{L}) \in \mathbb{R}^{m \times d}$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230426135424441.png" alt="image-20230426135424441" style="zoom:80%;" align="left"/>如图，本文和 OV-DETR 的匹配方式差异：

 本文方法的输出是 class-agnostic，因此只需要 m 个queries，后者则需要类别数 * m个queries（总 queries 的数量与数据集中类别总数呈正相关关系）。

第二步：利用 CLIP 的 zero-shot 能力

假设 $\hat{B}_{det} = \{b_{1}, ...,b_{m}\}$，使用 zero-shot 分类，最直接的方式是使用 CLIP 分别编码这些边框对应的图片区域：

$$
\forall_{1 \leq i \leq m} e_{i}^{img} = CLIP_{img}(Crop(x;b_{i})) \in \mathbb{R}^{d'} \tag{7}
$$

然后计算其与类别文本嵌入的相似度：

$$
\hat{P}_{clip} = [\hat{p}_{1}, ..., \hat{p}_{m}] \in \mathbb{R}^{m \times k} \\
\hat{p}_{i} = Softmac(\frac{1}{\tau}[cos(e_{i}^{img}, e_{1}^{txt}), ..., cos(e_{i}^{img}, e_{k}^{txt})])
$$

为了缓解基类和新类的性能差异较大问题，综合考虑 Decoder 端输出的概率预测和 CLIP 的分类概率：

$$
\mathrm{P}_{\text {final }}[\cdot, j]=\left\{\begin{array}{l}\alpha \mathrm{P}_{\mathrm{det}}[\cdot, j]+(1-\alpha) \mathrm{P}_{\mathrm{clip}}[\cdot, j] \text { if } j \in C_B \\ \beta \mathrm{P}_{\mathrm{det}}[\cdot, j]+(1-\beta) \mathrm{P}_{\mathrm{clip}}[\cdot, j] \text { if } j \in C_N\end{array}\right.
$$

**RoI-based Masked Attention**

由于使用公式(7) 的方式编码边框特征很低效，因此使用 RoI-based Masked Attention 提高效率。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230426151007466.png" alt="image-20230426151007466" style="zoom:80%;" align="left"/>  

如图，在 CLIP 文本编码器 ViT 的最后一层加入 RoI-based Mask，这就消除了对所有roi进行迭代推理的需要。：

 $Softmax(\frac{(E_{Q}W_{Q})(E_{K}W_{K})^{T} + M_{b_{i}}}{\sqrt{d}})$

 公式(7) 就可以变为：$[e_{1}^{img}, ... , e_{m}^{img}] = CLIP_{img}(x,\hat{B}_{det}) \in \mathbb{R}^{m \times d'} $



 **RoI Pruning**

由于 CLIP 的对比学习并不知道图像局部区域和文本标记之间的对齐，从而导致对背景区域不准确的预测。为了解决这个问题，本文提出了一个简单的RoI剪枝，从边框预测中识别非背景RoI。具体做法为只将预测概率高于某一阈值的边框用于 CLIP zero-shot 预测。

$$
\hat{B}_{prue} = \{b_{i} \in \hat{B}_{det}:p_{i} \in \hat{P}_{det} \and max(p_i) \geq \epsilon\}
$$

## 20230512

### 27_Multi-view Adversarial Discriminator: Mine the Non-causal Factors for Object Detection in Unseen Domains_CVPR 2023_有代码

> 作者：

Mingjun Xu, Lingyun Qin, Weijie Chen, Shiliang Pu, Lei Zhang

School of Microelectronics and Communication Engineering, Chongqing University, China Hikvision Research Institute, Hangzhou, China

> 代码：[GitHub - K2OKOH/MAD](https://github.com/K2OKOH/MAD)

> 贡献：

为了缓解域偏移对模型性能的影响，之前的工作大多通过域对抗学习（DAL）的方法，从源域中学习 domain-invariant （common）特征。受启发于因果机制（ *causal mechanisms*），由于 DAL 单一视角的性质，之前的方法可能会将那些暗藏的微小的非因果因素也包含在 common feature 中。为此，本文提出基于 Multi-view Adversarial Discriminator(MAD) 的域泛化模型，通过在源域上进行多视角对抗学习来移除 common feature 中的非因果因素。

domain shift：测试集和训练集分布不一样（输入分布不一样或输出分布不一样或输入和输出的关系不一样）

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230506093722173.png" alt="image-20230506093722173" style="zoom:80%;" align="right"/>**因果机制（ *causal mechanisms*）**:

基于因果机制的方法认为基于统计依赖性的预测是不可靠的，因为统计相关性既包含虚假的非因果相关性，也包含因果相关性。例如，吸烟、黄牙和肺癌密切相关，但只有吸烟是肺癌的病因。

如右图2，common feature 中还包含着一些非因果因素如灯光颜色、照明、背景等，本文的目的就是探索并去除这些隐藏的非因果信息。

MAD 由两部分组成：

1. 伪相关性生成器（Spurious Correlations Generator (SCG)）：通过随机增强来增加源域的多样性，以使得其中的非因果因素变得更突出；
2. 多视角域分类器（Multi-View Domain Classifier (MVDC)）：通过从 image 和 instance level 识别非因果因素，从而使域对抗学习更充分，指导特征提取器忽略非因果因素。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230506094643268.png" alt="image-20230506094643268" style="zoom:60%;" align="right"/>  整体结构如右图 Figure 4 所示：

 问题形式化：

 源域：$D_{s} = \{X_{s}, Y_{s}\}$

 提取特征：$S = f(X_{s})$

 特征 $S$ 包含  causal 和 non-causal factors：${s_{\text{cau}}, s_{\text{non}}}$ 存在以下关系，$\left\{\begin{array}{l}s_{\text {cau }} \subset s_{\text {com }} \\ s_{\text {non }} \supset s_{\text {pri }}\end{array}\right.$

 其中假设 $s_{\text{non}}$ 服从高斯分布

**Spurious Correlations Generator (SCG)**

之前的文献指出，图像的极高和极低频部分都包含了更多的域私有特性。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230506101616743.png" alt="image-20230506101616743" style="zoom:60%;" align="right"/>SCG 的做法如 Figure 5，

1. 先用 Discrete Cosine Transform 的到输入图片 $x$ 的频谱 $\mathscr{F}(x)$；
2. 然后通过带通滤波器对 causal 和 non-causal 进行分离: $\mathcal{M}(r) = e^{- \frac{u^{2} + v^{2}}{2R_{H^{2}}}} - e^{- \frac{u^{2} + v^{2}}{2R_{L^{2}}}}$；
3. 对 causal 部分保持不变，随机打乱 non-causal 部分：$R_{G}(S) = S \cdot (1 + N(0,1))$;
4. 使用 Inverse Discrete Cosine Transform 得到带有  non-causal factors 的增强后的图片 : $\hat{x}=\mathscr{F}^{\prime}\left(R_G(\mathcal{M}(r) \cdot \mathscr{F}(x))+(1-\mathcal{M}(r)) \cdot \mathscr{F}(x)\right)$

 **Multi-View Domain Classififier(MVDC)**

DAL是提取不同域共同特征的标准方法，它最小化不同域之间特征的对抗距离。H表示所有可能的域分类器的假设集，单个h依赖于最具区别性的域私有特征，因此它忽略了特征中不显著的域特定成分，并错误地将这些非因果成分作为共同特征。因此，MVDC 通过用编码器 $e_i$ 将特征映射到多个潜在空间，然后用独立的域分类器 $h_i$ 在每个空间中区分特征。这些域分类器鼓励特征提取器 F 忽略隐式的非因果因素，学习域不变的但具有因果性的特征。

$$
\begin{aligned}
\min _{\mathcal{F}} d_{\mathcal{A}}\left(D_{s 1}, D_{s 2}\right) & =\underbrace{\max _{\mathcal{F}} \min _{h \in \mathcal{H}} \operatorname{err}(h(s))}_{\text {Standard DAL }} \\
& \Rightarrow \underbrace{\max _{\mathcal{F}} \sum_{i=1}^M \min _{h_i \in \mathcal{H}, e_i} \operatorname{err}\left(h_i\left(e_i(s)\right)\right)}_{\text {Ours }}
\end{aligned}
$$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230506104208749.png" alt="image-20230506104208749" style="zoom:67%;" align="right"/>图6显示了MVDC的一个分支的结构，它代表了观察特征的多个视图之一。MVDC的完整结构分别包含M个图像级特征的分支和M个实例级特征的分支。MVDC的每个分支都包含一个自动编码器和一个结构上的分类器。

在图像层面，本文关注图像的全局非因果因素，如照明、颜色和背景纹理。这些全局的非因果因素在整个图像中是相似的，所以本文使用卷积层来构造编码器和解码器。在每个分支中，本文使用不同扩张率的膨胀卷积来提取不同的域的非因果因素。

在实例层面，本文使用全连接层来考虑更多的语义非因果因素，比如每个实例的摄像机角度。

 **Loss Function**

1. 重构损失 $L_{RC} = \frac{1}{M} \sum_{m=1}^{M} MSE(s, g_{m}(e_{m}(s)))$
2. 对抗域分类器损失 $L_{DC} = - \frac{1}{M} \sum_{m=1}^{M} \sum_{k=1}^{K} y_{k} \cdot log(p(D_{m}(e_{m}(s_{k}))))$
3. view-different loss 确保编码器将特性映射到不同的潜在空间 $L_{MV} = - \frac{\sum_{i}^{M} \sum_{j,i \neq j}^{M} \| e_{i}(s) - e_{j}(s) \|^{2}}{M^{2} - M}$

$L_{MVDC}^{img, ins} = L_{RC} + L_{DC} + L_{MV}$

consistency loss 确保  image-level 和 instance-level 分支的结果一致性$L_{cst} = \sum_{i,j}^{M} \sum_{n}^{N} \| \frac{1}{|I|} \sum_{u,v} p_{i}^{(u,v)} - p_{j,n}\|$

（I 是每张特征图的像素数，N是每张图片的实例数）

总损失： $L_{MAD} = L_{det} + \lambda (L_{MVDC}^{img} + L_{MVDC}^{ins} + L_{cst})$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230506111829004.png" alt="image-20230506111829004" style="zoom: 80%;" align="left"/>  

 **关于视图数量的消融：**

 视图的数量M是MAD中的关键超参数。更多的视图会导致更好的性能，但太多的自动编码器会增加模型的复杂性，从而减少边际效应。如图，本文发现，性能在M = 5之前有所提高，然后收敛到M = 8。因此，为了平衡性能和成本之，本文将M = 3设置。





### 28_Training Networks in Null Space of Feature Covariance for Continual Learning_CVPR 2021_有代码

> 作者：

Shipeng Wang, Xiaorong Li, Jian Sun, Zongben Xu

> 代码：https://github.com/ShipengWang/Adam-NSCL

> 贡献：

Plasticity-Stability Dilemma：在增量学习的背景下，网络按顺序在一系列任务上进行训练存在可塑性与可扩展性权衡的问题。**可塑性**(plasticity)指模型当前任务中学习新知识的能力；**稳定性**(stability)指模型保持其在先前任务上的性能。然而，同时实现可塑性和稳定性是很具有挑战性的，所以如何让网络能够在**学习新知识的同时避免对先前任务的灾难性遗忘**(catastrophic forgetting)就成了增量学习的难点。

本文作者首先提出了两个理论条件，分别对应增量学习网络的稳定性和可塑性。基于它们，作者设计了一种称为Adam-NSCL的新的网络训练算法，它迫使网络参数更新位于每个网络层的先前任务输入特征的零空间中，如图 1 所示。

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="D:\shu\paper\0_note\new\typora-user-images\image-20230510162545726.png" 
         alt="图1"
         style="zoom:80%"/>
    <br>		<!--换行-->
    图1：将候选参数更新投影到网络训练过程中之前所有任务的近似零空间中	<!--标题-->
    </center>
</div>

输入特征的分层零空间可以建模为特征的非中心协方差的零空间，该协方差可以在学习每个任务后递增计算。由于保证零空间的存在过于严格，故本文用输入特征非中心协方差的最小奇异值对应的奇异向量跨越的子空间来近似每一层的零空间。本文将该策略嵌入到 Adam 优化算法中，将Adam生成的候选参数更新逐层投影到近似的零空间中。

**输入特征分层零空间 （The layer-wise null space of input feature)** 指的是特征图为零的位置

**特征的非中心协方差** 我们可以正常计算得到特征的协方差，并对其进行去中心化。本文用最小奇异值对应的奇异向量来近似

> 方法：

本文基于正则化思想，采取约束模型的参数更新的方式来保证模型的特征学习。结构上，所有任务共享同一个主干网络，但每个任务都有自己的分类器。 模型在相应的任务上训练后，分类器将被固定。（multi-head classifier）

**conditions for continual learning**

* **Condition 1 (stability).**

对于 $l=1,⋅⋅⋅,L$ ，当 $f$ 在任务$T_t$ 上被训练时，$Δw^l_{t,s}$ 在每个训练步骤 $s$ 应该位于无中心特征协方差矩阵$\overline{X}^l_{t−1}$ 的零空间中，即：$\overline{\chi}^l_{t−1}∆w^l_{t,s} = 0. \quad(3)$，其中，$∆w^l_{t,s}$ 表示 $s$ 阶段网络参数的变化量。

* **Condition 2 (scalability).**

假设网络 $f$ 正在接受任务 $T_t$ 训练，且 $g_{t,s}=\{g^1_{t,s},…,g^L_{t,s}\}$ 表示在 $s$ 阶段训练 $f$ 经梯度下降算法生成的参数更新。$<∆w_{t,s}, g_{t,s} > > 0$ 应该成立，其中 $<·， ·> $代表内积。

condition 1 的内在逻辑是让参数的更新在特征的非中心协方差的零空间中，这样的好处是**降低参数更新对于原特征的影响，从而保护稳定性**（奇异值小的空间对原矩阵的影响较小）；condition 2 的 $<.,.>$ 表明梯度更新方向与参数变化量相似，即**梯度是朝着最小化损失的方向前进的，从而保护了伸缩性**。

**Covariance Null Space**

当网络 $f$ 在任务 $T_{t-1}$ 上训练完成后，将当前任务数据 $X_{t-1}$ 送入网络获得每一层的特征输入  $X_{t-1}^{l}$ ，然后计算当前任务数据的去中心化协方差矩阵 ${\mathcal{X}}_{t-1}^l=\frac{1}{n_{t-1}} ({\mathcal{X}}_{t-1}^l)^{T} {\mathcal{X}}_{t-1}^l $ ，接着更新去中心化特征协方差矩阵：

$$
\overline{\mathcal{X}}_{t-1}^l=\frac{\bar{n}_{t-2}}{\bar{n}_{t-1}} \overline{\mathcal{X}}_{t-2}^l+\frac{n_{t-1}}{\bar{n}_{t-1}} \mathcal{X}_{t-1}^l
$$

其中：$\bar{n}_{t-1} = \bar{n}_{t-2} + {n}_{t-1}$

**Approximate Null Space**

$\overline{\mathcal{X}}_{t-1}^l$ 的零空间不容易直接求得，所以通过奇异值分解（SVD）来近似计算它的零空间。对 $\overline{\mathcal{X}}_{t-1}^l$ 进行奇异值分解：

$$
U^l, \Lambda^l,\left(U^l\right)^{\top}=\operatorname{SVD}\left(\overline{\mathcal{X}}_{t-1}^l\right)
$$

$U^l = [U^{l}_{1}，U^{l}_{2}]$，$\Lambda^l=\left[\begin{array}{cc}\Lambda_1^l & 0 \\ 0 & \Lambda_2^l\end{array}\right]$。

如果 $\Lambda_2^l$ 中恰好是所有元素为0的奇异值，那么 $U_{1}^{l} \Lambda_1^l (U_{1}^{l})^{T} = \overline{\mathcal{X}}_{t-1}^l -> \overline{\mathcal{X}}_{t-1}^l U_{2}^{l} = U_{1}^{l} \Lambda_1^l (U_{1}^{l})^{T} U_{2}^{l} = 0$ 成立，因为 $U^{l}$ 是酉矩阵。这就表明 $U_{2}^{l}$ 的值域空间是 $\overline{\mathcal{X}}_{t-1}^l$ 的零空间。那么就可以将梯度下降算法返回的参数更新投影到 $\overline{\mathcal{X}}_{t-1}^l$ 的零空间内得到满足条件1的第 $s$ 步训练更新：

$$
$∆w^l_{t,s} = U_{2}^{l} (U_{2}^{l})^{T} g^l_{t,s}
$$

但并不能保证一定存在**0**奇异值，受主成分分析的启发，本文选择满足 $\lambda \in \{ \lambda \leq a \lambda_{min}^{l} \}$ (a > 0) 的奇异值放进 $\Lambda_2^l$ ，$\lambda_{min}^{l}$ 是最小的奇异值。这样就可以用 $U_{1}^{l} \Lambda_1^l (U_{1}^{l})^{T}$ 来近似 $\overline{\mathcal{X}}_{t-1}^l$ ，即  $\overline{\mathcal{X}}_{t-1}^l U_{2}^{l} \approx 0$ .

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="D:\shu\paper\0_note\new\typora-user-images\image-20230510164255042.png" 
         alt="图2"
         style="zoom:80%"/>
    <br>		<!--换行-->
    图2：网络流程图	<!--标题-->
    </center>
</div>
Adam-NSCL与以往的增量学习策略不同在于，它不需要额外存储之前任务学习过的数据，不需要靠惩罚项使网络参数变化减小，也不需要随着任务增加扩增网络的内存。它通过梯度下降法在零空间上的投影强制网络在不降低之前任务上的性能的前提下学习新任务，为增量学习提供了新的思路。

## 20230519

### 29_Detecting Everything in the Open World: Towards Universal Object Detection_CVPR 2023_有代码

> 作者：Zhenyu Wang1,2 Yali Li1,2 Xi Chen3 Ser-Nam Lim4 Antonio Torralba5 Hengshuang Zhao3 Shengjin Wang1,2

1 Beijing National Research Center for Information Science and Technology (BNRist)

2 Department of Electronic Engineering, Tsinghua University

3 The University of Hong Kong

4 Meta AI

5 Massachusetts Institute of Technology

> 代码：https://github.com/zhenyuw16/UniDetector

> 贡献：

背景：由于带标签的视觉数据有限，且在开放世界中存在 novel 类，因此传统目标检测器的泛化能力非常有限。

本文提出 UniDetector ，致力于打造一个能在开放世界中检测大量类别对象的同统一目标检测器。具体做法如下：

1. 通过使用多各**包含不同标签空间的数据集进行预训练**，对齐图像和文本的特征空间，这保证了能够有足够的信息来表示大量的表征；
2. 利用视觉及语言模态的信息，使得模型能够通过**保持已知类和未知类的平衡**从而很好地泛化到开放世界；
3. 本文提出**解耦的训练方式**以及**概率校准**能够进一步提高模型的泛化能力。

结果：

- 训练时有500个类参与训练，能够检测7K个类别
- 在没有看到任何对应图像的情况下，它比传统有监督学习的baseline方法的精度高了4%以上。在13个不同场景的公共检测数据集上，UniDetector仅用3%的训练数据量就达到了最先进的性能

> 方法：

如Figure 2，UniDetector框架：

* 第一阶段：基于 image-text 预训练，对齐文本和图像的特征空间；
* 第二阶段：用解耦的方式训练包含不同标签空间的训练数据集；
* 第三阶段：利用概率校准来平衡对已知类和未知类的预测。

<img src="D:\shu\paper\0_note\new\typora-user-images\image-20230514220939801.png" alt="image-20230514220939801" style="zoom:80%;" />

**Heterogeneous Label Space Training**

使用多种标签空间的组合数据训练模型的三种方式：

a.  seperate label spaces，各模型在单个数据集上单独训练，然后分别对测试数据进行评估，最终结果由所有模型的结果结合得到；

b.  unified label space，将所有标签映射到同一标签空间；

c.  partitioned label space，所有数据集共享 backbone，但有各自的分类层。在推理时，利用测试label的class embedding 即可避免标签冲突。

<img src="D:\shu\paper\0_note\new\typora-user-images\image-20230514222546399.png" alt="image-20230514222546399" style="zoom:80%;" />

**Decoupling proposal generation and RoI classifification.**

两阶段目标检测器：视觉骨干编码器 + RPN + ROI 分类模块，因为RPN具有较好的能力泛化到未知类，但是ROI 特定于类的分类模块会阻碍模型的泛化性能，因此，作者提出使用解耦的方式进行训练：

1. 区域候选生成阶段，使用 traditional ImageNet pre-trained parameters 初始化参数并以 class-agnostic way 训练；
2. ROI分类阶段，使用 image-text pre-trained parameters 初始化参数并以 Fast R-CNN 的方式训练；

    两种预训练参数包含互补的特征，为检测器提供了更全面的信息。

**Class-agnostic localization network**

为了在开放的世界中提出一般性的建议框，使用 CLN 替代RPN：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230516212702689.png" alt="image-20230516212702689" style="zoom:67%;" align="left"/>   

CLN能通过RoI进一步提高 box 的质量，其中 classification 为二元类未知分类；

 对于第 $i$ 个 proposal,

 来自 RPN 的定位分数为：$s_{i}^{r_{1}}$，来自 RoI 的定位分数为：$s_{i}^{r_{2}}$，分类分数为：$s_{i}^{c}$，那么来自 CLN 的最终confidence为：$\eta_{i} = (s_{i}^{c})^{\alpha} \cdot (s_{i}^{r_{1}}s_{i}^{r_{2}})^{1-\alpha}$



**Open-world Inference**

由于训练过程中只出现基本类别，所以在使用 test vocabulary $L_{test}$推理过程中，基类往往比新类的置信度更大，从而占据推理的主导地位；
基类的过度置信将容易导致检测器忽略数量多的新类，从而损害检测器在 open world 中的性能。

为了避免偏差问题，进行概率校准。目的：降低基类的概率，增加新类的概率，从而平衡最终的概率。

$$
p_{ij} = \frac{1}{1+exp(-z_{ij}^{T}e_{j}/\tau)}/ \pi_{j}^{\gamma} , j \in L_{test} \tag{1}
$$

公式1 表示对于第 $i$ 个候选框特定于类别的预测，考虑到  class-agnostic task 的泛化能力，作者将 $p_{ij}$ 乘上来自 CLN 网络的对象得分作为最终的检测分数：

$$
s_{ij} = p_{ij}^{\beta} \eta_{i}^{1-\beta}
$$

### 30_Overlooked Factors in Concept-based Explanations: Dataset Choice, Concept Learnability, and Human Capability_CVPR 2023_有代码

> 作者：Vikram V. Ramaswamy, Sunnie S. Y. Kim, Ruth Fong, Olga Russakovsky Princeton University

> 代码：https://github.com/princetonvisualai/OverlookedFactors

> 贡献：

背景：**基于概念的可解释性方法**旨在使用一组预定义的语义概念来解释深度神经网络模型的组成部分和预测，通过在一个新的“探测”数据集（“probe” dataset）上评估一个训练好的模型，并将模型的输出与该数据集中标记的概念相关联。现有的工作主要聚焦于寻求开发新的方法，但这些方法存在一些局限性尚未被很好地理解与阐明，例如被忽视的重要因素——“探测”数据集的选择、解释中所使用的概念。

本文作者识别并分析了基于概念的可解释性方法中被忽视的因素，主要从以下三个角度：

1. **“探测”数据集的选择**；本文发现不同的“探测”数据集对生成的解释影响很大，并提出建议：应尽量选择与模型训练的数据集分布相似的“探测”数据集。

   “探测”数据集：一组标记有语义概念的（图像）数据集
2. “探测”数据集中**概念的可学习性**；本文发现“探测”数据集中的很多概念比要解释的目标类别更难学习，这引起了对解释的正确性的担忧。因此本文建议，仅使用比目标类更好学习的概念。
3. 人类倾向于用于解释的**概念数量的上限**；现有的方法中使用成百上千的概念来解释，但是关于人类理解的研究表明用于解释的概念的数量上限为32，多出的概念对于解释不能带来更好的解释作用。因此，本文建议将解释中的概念数量限制在32以内。

> 问题设定：

在这项工作中，本文深入研究了图像分类模型的基于概念的可解释性方法，这些方法使用预定义的语义概念来解释模型组件和（/或）预测。将“探测”数据集作用于一个训练好的模型，这些方法可以用所提供的概念进行解释。

![image-20230516110516969](D:\shu\paper\0_note\new\typora-user-images\image-20230516110516969.png)

如 Figure1：

- 模型对于场景类别的预测结果被解释为一系列视觉概念的线性组合结果，根据这些概念的出现与否（场景中出现某场景，则该场景处分数对应于1，否则为0）进行加权计算得到；
- 线性组合的系数通过将训练好的模型在“探测”数据集上评估，并将预测结果与相应的视觉概念相关性结合得到。

> 实验方案：

第一行：4种代表性基于概念的可解释性方法

第一列：“探测”数据集

|              | NetDissect | TCAV | Concept Bottleneck | IBD |
| ------------ | ---------- | ---- | ------------------ | --- |
| ADE20k       | √         | √   |                    | √  |
| Pascal       | √         | √   |                    | √  |
| CUB-200-2011 |            |      | √                 |     |

> **Dataset choice: Probe dataset has a profound impact on the explanations**

Baseline：衡量模型预测和概念之间的相关性，并生成以概念线性组合而成的 class-level 的解释；

NetDissect：识别模型中被特定概念激活的神经元。并生成 neuron-level 的解释；

TCAV：根据模型特征空间中对应于概念的向量，以 concept activation vectors 的形式生成解释。

结果：

<div>
    <img
         src="D:\shu\paper\0_note\new\typora-user-images\image-20230516133253234.png"
         >
</div>
<img src="D:\shu\paper\0_note\new\typora-user-images\image-20230516133933577.png" alt="image-20230516133933577" style="zoom:80%;" align="left"/><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230516134043594.png" alt="image-20230516134043594" style="zoom:67%;" align="left"/> 



























> **Concept learnability: Concepts used are less learnable than target classes**

从下图可以看出，随机选择的10各场景类别中，所有场景类别的解释概念都至少包含一个比目标类更难学习的概念（红色字体标识）。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230516134326137.png" alt="image-20230516134326137" style="zoom:67%;" />

> **Key findings from the human studies**

* When presented with more concepts, participants spend more time but are worse at recognizing concepts.
* Concept-based explanations offer little to no advantage in model output prediction over example-based explanations.
* The majority of participants prefer explanations with 8, 16, or 32 concepts.

（补充：

所谓的**saliency map**（显著图），就是在给定的一张图上面对每一个像素点进行扰动，对结果影响越大的像素点越显著。smoothgrad做的就是这样一份工作，旨在提供分类器做决策的证据。

不过saliency map的局限性体现在了人类认知对解释器的确认偏见（Confirmation Bias），就是说不能是这个图迎合了你的认知（比如说显示了鸟的轮廓）就能说明这个是分类器把它分为某类的依据。

除此之外，saliency map的问题还体现在对于其反映的决策“证据”在人类的理解上有出入。举个例子，下面这个图是用smoothGrad改良后用来解释图片识别模型的ATM取款机的saliency map。图中可以看出，显著的点意味着这些点的扰动对于识别决策有着较大的影响。比较有趣的一点是，我们可以发现图中很多和ATM机无关的图像的点也十分的显著。比如说人的帽子，以及旁边的小推车。那么问题就来了：小推车的像素点怎么也被分类器纳入了作为识别ATM机的证据了呢？这个模型对于ATM的“概念”到底了解多少？

<img src="https://pic1.zhimg.com/70/v2-552aefebdf4db561eae76acd9abdab9e_1440w.image?source=172ae18b" alt="基于概念的模型可解释性" style="zoom: 50%;" />

）

## 20230526

### 31_RegionCLIP: Region-based Language-Image Pretraining_CVPR 2022_有代码

> 作者：Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, Jianfeng Gao

> 代码： https://github.com/microsoft/RegionCLIP

> 贡献：

背景：用（image-text pairs）图像-文本对数据的对比语言图像预训练模型（Contrastive language-image pretraining，CLIP）在图像分类方面的 zero-shot 和迁移学习的取得了非常好的结果。但是，直接应用 CLIP 这样的模型来进行图像区域推理(如目标检测)效果将会不好。主要原因：CLIP 只是进行图像整体与文本描述匹配，而不能捕获图像区域（image regions）与其对应文本之间的细粒度对齐(fine-grained alignment)，不能准确定位图片上的区域。

因此，**RegionCLIP** 的目的便是实现从 image-text pairs 的匹配到 region-text pairs 的匹配。构建一个模型进行图像区域的推理研究(如目标检测)，目的是学习一个包含丰富的对象概念的**区域视觉-语义**空间，以便它可以用于开放词汇的目标检测。实质上就是训练一个视觉**编码器 V**，使它可以编码图像区域，并将它们与语言编码器 L 编码的区域描述相匹配。

**关键思想**是在预训练期间显式对齐图像区域和文本标记。

存在**两个关键挑战**：1、图像区域和文本标记之间的细粒度对齐在图像-文本对中并不可靠；2、图像的文本描述通常不完整，即许多图像区域没有相应的文本描述。

因此本文使用 COCO Cap 及 CC3M 数据，借助 CLIP 从 image-text pairs 中获取 region-text pairs 伪标签数据，然后通过对比学习及蒸馏的方式使 region-text 对齐，训练 RegionCLIP 中的视觉编码器。

> 方法：

从图像描述的文本语料中解析的对象概念（获得大量表示对象的概念），将这些概念填充到预定义的模板中来形成图像区域描述（像 CLIP 那样的 prompt templates 操作）。根据输入图像及其生成的候选区域（如RPN来生成），使用一个预先训练好的 CLIP 模型来进行图像区域与其区域描述的匹配，形成 region-text pairs 数据，即为**图像区域**创建相应的**“伪”标签**。此外，结合“伪”图像区域-文本描述对（“pseudo” region-text pairs）和真实的图像-文本对（ground-truth image-text pairs），通过对比学习和知识蒸馏对我们的视觉语言模型（VLP）进行预训练，从而学习 region representations。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230522202621511.png" alt="image-20230522202621511" style="zoom:80%;" />

说明：***Vt*** 表示 CLIP 的 Visual encoder，***L*** 表示 CLIP 的 Language encoder，***V*** 表示本文 (RegionCLIP) 要预训练的 Visual encoder。

先利用 CLIP 预训练好的模型 ***Vt*** 和 ***L*** 进行region-text pretraining，将 ***Vt*** 作为 teacher model，***V*** 作为 student model，利用知识蒸馏将 ***Vt*** 根据学到的知识指导 ***V***，***L*** 在 image-text pretraining 与 region-text pretraining 始终保持一致。

**第一步：Image-text pretraining (CLIP)** 获得 ***Vt*** 和 ***L***

**第二步：Region-text pretraining**

***1、首先要获得 region-text 伪标签***

***1.1 Visual and Semantic Region Representation***

文中指出将目标检测任务分解为**定位**与**识别**。

Visual region representation：

定位：利用现成的目标定位器 (如RPN) 生成一些候选区域 r (Proposed regions)，(训练 RPN 时只使用了 bounding boxes 标签，不带分类标签)

识别：将由 RPN 形成的候选区域 r 经过 visual encoder ***Vt*** 进行编码（这里通过了特征池化 feature pooling (如RoIAlign) 的方法对RPN形成的候选区域的特征进行池化），形成每个候选区域对应的视觉特征 $v$。

Semantic region representation：

利用现成的语言解析器从文本语料库（text corpus，指的就是对图片的描述文本）提取出有关目标/对象的词库（a pool of object concepts，提出图像中对象的名称，图片有猫，就将图片对应文本描述中cat这个词提取出来）。然后将这个object concepts pool 转化为 prompt templates，（比如：cat-->a photo of a cat），**相当于从对图像的描述转化为了对图像区域的描述**。然后用 CLIP 预训练好的 ***L*** 进行编码形成对应的语义特征 $l$

***1.2 Alignment of region-text pairs***

计算相似度：$S(v,l) = \frac{v^{T} \cdot l}{\|v\| \cdot \|l\|}$

对于 $v_{i}$, 选择使得 $S$ 最大的那个  $l_{m}$ 作为其对应的文本描述特征，因此获得 region-text 对伪标签。

***2、RegionCLIPd 的预训练***

使用的数据，生成的 region-text pairs + 现有的 image-text pairs

用类似于 CLIP 的方式进行对比训练：

$$
L_{cntrst} = \frac{1}{N} \sum_{i} - log(p(v_{i}, l_{m})) \\
其中 p(v_{i}, l_{m}) = \frac{exp(S(v_{i},l_{m}) / \tau)}{exp(S(v_{i},l_{m}) / \tau) + \sum_{k \in N_{r_{i}}}exp(S(v_{i},l_{m}) / \tau)}
$$

由于该 contrastive loss 是针对 region-level 的，而在训练过程中使用了 image-text pairs 数据，因此，还设计了 image-level contrastive loss，对于 image 就直接当作由一个框覆盖了所有的区域即可，类似地还有一个 $L_{cntrst-img}$

蒸馏：

$$
L_{dist} = \frac{1}{N} \sum_{i} L_{KL}(q_{i}^{t}, q_{i}) \\
其中 q_{i}^{t} = softmax(S(v_{i}^{t},l_{1}) / \tau, ..., S(v_{i}^{t},l_{C}) / \tau, ) 来自teacher\_model \\
q_{i} 以相同的计算方式，来自 student\_model
$$

所以，总预训练损失为：$L + l_{cntrst} + L_{cntrst-img} + L_{dist}$

**第三步：将预训练好的 RegionCLIP 用到下游任务**

### 32_CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching_CVPR 2023_有代码

> 作者：Xiaoshi Wu, Feng Zhu, Rui Zhao, Hongsheng Li

> 代码：[tgxs002/CORA: A DETR-style framework for open-vocabulary detection (OVD). CVPR 2023 (github.com)](https://github.com/tgxs002/CORA)

> 贡献：

以往基于大规模视觉-语言预训练模型做 OVD 的工作中，本文识别了两个需要解决的核心障碍：

1. 将基于完整一张图片预训练的 VL 模型应用于局部区域识别的任务时，存在分布不匹配的问题；（完整到局部）；
2. 未知类对象的定位困难；

为此，本文提出一个 DETR-style 框架，通过 Region prompting 和 Anchor pre-matching 来将 CLIP 应用于 OVD。其中：

**Region prompting**：通过提示  CLIP-based region classifier 的 区域特征来缓解整体到局部的分布gap；

**Anchor pre-matching**：通过 class-aware 的匹配机制学习  generalizable object localization。

> 方法：

给定一个图像作为输入，首先从预训练的 CLIP 图像编码器中使用 ResNet 获取空间特征图，该 backbone 由区域分类和物体定位分支共享。与传统检测器不同，在本文的框架中，定位和分类是解耦、并按顺序进行的，从而更好地适应 OVD 问题。训练一个DETR-style 的对象定位器，细化一组 object queries 及其相关的锚框从而定位对象，然后由来自 CLIP 的区域分类器对对象进行分类。

<img src="C:\Users\wangxuefei\AppData\Roaming\Typora\typora-user-images\image-20230524101246340.png" alt="image-20230524101246340" style="zoom:67%;" />

**Region Prompting：**

<img src="C:\Users\wangxuefei\AppData\Roaming\Typora\typora-user-images\image-20230524145603791.png" alt="image-20230524145603791" style="zoom:80%;" align="left"/>  

给定一张图片和一系列感兴趣区域，首先用 CLIP img encoder 的前三个 blocks 编码整张图片获得完整的 feature map；

 （传统用 CLIP 做 region 分类的方式为，先将感兴趣区域对应的图片裁剪下来，然后分别送入 CLIP img encoder 编码。那么这样的做的缺点在于，如果有很多 RoI有重叠区域的话，就很不高效，相当于重复编码；另外，这样也会丢失上下文信息）

 在将特征送入 CLIP img encoder 的最后一个 block 之前，使用 RoIAlign 池化根据 anchor boxes（或是预测的boxes）获得 region features；

 由于存在 whole-region 的分布 gap，因此本文提出 region prompting ，通过可学习的 prompts $p \in \mathbb{R}^{S \times S \times C}$ （大小和维度与 region feature保持一致）来对 region feature 做增广，即按元素加加到 region feature 上：$v_{prompt} = P(f_{region} \oplus p)$



**Region Prompts** 的训练：

 保持模型的其他部分参数冻结，使用 base class 的注释信息来训练。通过 gt box 获得相应的 region feature，类名信息送入 CLIP text encoder 编码获得的 class name embedding 作为分类权重，并用标准的交叉熵损失作为分类损失。

 **Anchor Pre-Matching：**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230524152530915.png" alt="image-20230524152530915" style="zoom:75%;" align="left"/> Anchor Pre-Matching（锚点预匹配）是一种预处理技术，旨在将一组锚点（anchors）与其对应的目标进行匹配，从而生成训练样本。

 每个 gt 框都预先匹配到一组具有相同标签的 queries，某个 query 的标签通过对对应的 anchor region feature 进行分类得到：$\hat{c_{i}} = argmax_{c \in C^{B}} \quad cosine(v_{i}, l_{c})$

 与 Conditional Matching 不同，本文的锚点预匹配机制根据图像内容自适应地为不同的类分配锚点框，能够保持常量的 query 数，而与类别数量大小解耦。通过锚点预匹配，所有的类都可以在一次传递中一起解码，从而消除了对每个类重复解码的需要。

为了提高带有锚点预匹配的开放词汇检测器的泛化能力和训练收敛性，本文还引入了两种有效的训练技术，即 “Drop Class” 和 “CLIP-Alsigned Labeling”。
**Drop Class**：在训练过程中随机丢弃类别可以进一步提高模型的泛化能力。由于目标是训练一个检测器，从用户指定的类别列表中检测对象，在固定的类别列表上训练会导致偏差。通过在训练期间随机删除基本类别来减轻这种偏见。

 **CLIP-Alsigned Labeling**：通过锚定预匹配机制，只有当至少一个具有相同预匹配标签的 query 存在时，该 gt 才会被训练。否则就会被忽略，从而影响收敛。一部分原因是 anchor 不准确。但是，即使 gt 有对应精确的anchor，由于区域分类器的识别精度限制，仍然可能被忽略（gt 标签与用于预匹配的CLIP区域分类器没对齐）。因此，本文用区域分类器重新标记训练数据集中的边框。使用这种技术，可以匹配更多的 gt 框。



## 20230602

### 33_Varying demands for cognitive control reveals shared neural processes supporting semantic and episodic memory retrieval_NATURE COMMUNICATIONS 2021（认知）

> 作者:

​				Deniz Vatansever 1,2✉, Jonathan Smallwood 2 & Elizabeth Jefferies2

​				1 Institute of Science and Technology for Brain-inspired Intelligence, Fudan University, Shanghai, PR China. 2Department of Psychology, University of York, York, UK

> 贡献：

​		复旦大学和约克大学的合作团队研究发现了与控制记忆提取相关的脑区，这些脑区辅助大脑完成认知控制功能。

​		大脑的长期记忆分为外显记忆和内隐记忆，其中**外显记忆**包括语义记忆和情景记忆两部分。**语义记忆**指对一般事实的概念性记忆，是一种客观性的知识，与个人经验无关；而**情景记忆**记录生活中所发生的特定事件，与个人经历息息相关，两种记忆共同帮助我们了解并回应周围的世界。数十年的临床和实验研究表明，**语义记忆和情景记忆存储于两部分相互独立的大脑区域**。

（补充：心理学家把那些可以主动提取出来的记忆称为外显记忆，在大脑中存在但却无法主动将其提取出来的记忆称为内隐记忆。）

​		本文设计了两组相互独立的实验。

​		第一组实验使用功能性磁共振成像技术，发现大脑在提取语义记忆和情景记忆时，一部分**共同脑区**的激活程度增强。研究结果显示：**左侧额下回**（left inferior frontal gyrus, LIFG）、**前岛叶皮层**（anterior insular cortex， aINS）都参与了弱关联的**语义**和弱编码的**情景**记忆的提取。

​		第二组大规模的个体差异性实验进一步发现了一个**共用的脑区环路**，其中共同的激活区域（左侧额下回和前岛叶皮层）与作为默认网络的核心区域腹内侧前额叶皮层（ventromedial prefrontal cortex，vmPFC）的**功能性连接减弱**与**两种记忆提取的更好表现有关**。

​		==这两组实验结果揭示了辅助语义记忆和情景记忆提取的共有神经环路，使得大脑能够灵活地提取功能不同的长时记忆。==

尽管在语义记忆和情景记忆的认知和神经实例上长期存在这些的区别，但新出现的证据现在对它们之间分离的程度提出了质疑。具体来说，常见的认知过程被认为是在支持语义记忆和情景记忆的检索网络中观察到的大量重叠的基础。这两个记忆领域可能共有的一个核心过程是**认知控制**。认知控制通常被定义为一个**目标导向的执行系统**，它能根据不断变化的环境对反应做出灵活调整。

本文提到，之前关于两种记忆不同的神经检索机制可能是由于单独的研究某一领域造成的偏差。在有限的实验设置下，访问记忆所需的自动重新激活与受控检索过程的程度，可能是两个长期记忆系统的神经提取网络之间区别的一个重要因素。

本研究的主要目的是比较和对比参与语义记忆和情景记忆提取的神经过程。通过在两种记忆类型中引入**记忆强度操纵**（即强试验和弱试验），我们能够评估对记忆的控制提取是依赖于共享的还是不同的神经过程。

 3-alternative forced choice (**3-AFC**)（三点强迫选择法）：给定三个选项，选择其中一个。

### 34_Continual Detection Transformer for Incremental Object Detection_CVPR 2023_有代码

> 作者：

​				Yaoyao Liu1   Bernt Schiele1   Andrea Vedaldi2   Christian Rupprecht2
​						1Max Planck Institute for Informatics   2Visual Geometry Group, University of Oxford  

> 贡献：

背景：在增量设定的任务中，通常借助知识蒸馏（KD）、样例回放（ER）技术来缓解灾难性遗忘问题。然而直接将 KD 和 ER 用在基于 Transformer 的目标检测算法中却不能获得很好的效果。

因此，为了解决这个问题，本文提出 ContinuaL DEtection TRansformer (==CL-DETR==)，使得在基于 Transformer 的持续学习目标检测设定下能够有效的利用 KD 以及 ER 技术。首先，通过 **Detector Knowledge Distillation (DKD) loss** 专注于旧模型中信息最丰富且可靠的预测，忽略冗余的背景预测，并确保可用 GT 信息的兼容性；关于 ER ，本文呢提出一种**校准策略**，通过保存训练数据的标签分布，更好地匹配训练集和测试集的数据分布。

另外，本文提出一种新的数据划分设定，如下图，左（传统）vs右（本文）：(2In this way, **some images end up containing no annotated objects**.)

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230601103826629.png" alt="image-20230601103826629" style="zoom:80%;" align="left"/><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230601103927163.png" alt="image-20230601103927163" style="zoom:80%;" />

> 方法：

**Detector knowledge distillation**

标准的蒸馏损失：

对于现阶段的一张图片输入$x$：$\hat{y}^{oid} = \Phi^{oid} (x)$

最小化损失：$L_{KD} (\hat{y}, \hat{y}^{oid})= \sum_{j \in N} [\sum_{c \in C} - \hat{p}_{j}(c) log \hat{p}_{j}^{old}(c)] + L_{box}(\hat{b}_{j}, \hat{b}^{oid}_{j})$

由于预测框数量 N = 100，其中大部分是无对象的背景，因此该损失会被背景信息主导；另外，由于旧模型见过所有之前阶段训练图片，现阶段的训练图片中可能包含旧的类别对象，因此 $L_{DETR} + L_{KD}$ 这两项损失的约束存在矛盾（？）。因此本文提出新旧知识应该以结构化的方式融合，如下图：

* 将旧模型认为最有可能包含物体，但与当前 gt 框 IOU 小于 0.7 的预测框作为为标签：$\hat{y}^{psedu}$

* 将为标签和当前 gt 结合，并使用背景填充至数量 N：$y^{distill} = (y_{i})_{i:c(p_{i})\neq \phi} \oplus \hat{y}^{psedu} \oplus y^{bg}$

* 然后使用DETR的训练方式，用匈牙利进行一对一匹配，所以，本文提出的 detector knowledge distillation (DKD) loss ：$L_{DKD}(\hat{y}, y^{distill}) = L_{DETR}(\hat{y}, y^{distill})$

  

  <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230531193240713.png" alt="vv" style="zoom: 80%;" />

 **Distribution-preserving calibration**

ER 方法在 IOD 中存储少量样本并在未来阶段重放，在保存旧的类别知识方面是有效的，但存在新旧类别之间严重不平衡的问题。在分类任务中。通常选择创建一个类平衡的子集来微调模型的部分参数（如分类器），但这个方法在目标检测中不适用。首先，由于检测数据的类分布很不平衡，与其将其变成均匀分布不如去你和其自身的数据分布；其次，因为一张图像中可能包含多个不同类的对象，因此去选出平衡的子集并非易事。本文决定选择回放样例的时候，使回放样例的数据分布拟合训练数据的分布：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230531205503129.png" alt="image-20230531205503129" style="zoom:80%;" align="left"/>     

​    

如算法 2 所示，在第 i 个训练阶段，产生一个分布尽可能靠近该阶段训练数据 $D_{i}$ 的回放样例集 $\epsilon_{i}$

 通过最小化数据集 $D_{i}$ 和  $\epsilon_{i}$ 中类别的边际分布来挑选样本。

$e^{*} ⬅ \sum_{c \in C_{i}} p_{D_{i}}(c) log p_{\epsilon_{i} \cup\{e\}}(c)$





**Learning using balanced data**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230531212715368.png" alt="image-20230531212715368" style="zoom:67%;" align="left"/>     

​    

​    

 训练分为两步：

 第一步：基于所有当前可用的数据$D_{i}和\epsilon_{1:i-1}$使用 DKD loss 训练模型

 第二步：基于新的样例回放集 $\epsilon_{i}$，使用 DETR 损失微调模型















# 实验



## 评价指标

* 精确率 $𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 = \frac{𝑇𝑃}{𝑇𝑃 + 𝐹P}$
* 召回率 $𝑅𝑒𝑐𝑎𝑙𝑙 = \frac{𝑇𝑃}{𝑇𝑃+𝐹N}$
* **AP**(Average Precision)，AP 值是 Precision-Recall 曲线下方的面积，表示的是不同召回率下精确率的平均值。
* **mAP**(Mean Average Precision)是多个类别 AP 的平均值。
* **WI**（Wilderness Impact）表示模型在开放环境下与在封闭条件下测试时的性能变化，其定义如下：

$$
WI = \frac{Precision_{c}}{Precision_{o}} - 1= \{\frac{\frac{TP_{c}}{TP_{c} + FP_{c}}}{\frac{TP_{c}}{TP_{c} + FP_{c} + FP_{o}}}\} - 1 = \frac{FP_{o}}{TP_{c} + FP_{c}}
$$

​		其中𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛𝑐表示在封闭集下的精确率，𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛𝑜表示在开放集下的正确率

* **A-OSE** 表示的是被错误地分类为任何已知类的未知对象的数量，即$𝐹𝑃_{o}$

## 20230310

![ORE](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221230205937979.png)

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230316101831878.png" alt="image-20230316101831878" style="zoom:80%;" />

![image-20230328160525928](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230328160525928.png)

### Featurized-QueryRCNN+nms（0.6） （HungarianMatcher）闭集

| t1      |       |
| ------- | ----- |
| mAP(↑) | 58.33 |

## 20230317

### Featurized-QueryRCNN+nms + CLIP-TEXT + HungarianMatcher 开集

cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.5

#### clip-hungari-1-0.5

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 42.1470 |
| WI(↓)       | 0.0394  |
| A-OSE(↓)    | 957     |
| U-Recall(↑) | 26.1063 |

#### clip-hungari-1-0.7

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 43.1189 |
| WI(↓)       | 0.0399  |
| A-OSE(↓)    | 988     |
| U-Recall(↑) | 26.2392 |

#### clip-hungari-0.5-0.7

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 42.3855 |
| WI(↓)       | 0.0412  |
| A-OSE(↓)    | 1015    |
| U-Recall(↑) | 26.3421 |

以上三个实验小结：只通过 THRESHOLD_UNKNOWN  来给 unknown 类打标签，可能会因为模型打的 unknown 伪标签太多而大大影响对已知类的检测性能。

因此考虑加入限制：TOP_K_UNKNOWN

* cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.7	cfg.OWOD.TOP_K_UNKNOWN = 5

#### clip-hungari-1-0.7-5

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 55.4414 |
| WI(↓)       | 0.0745  |
| A-OSE(↓)    | 6828    |
| U-Recall(↑) | 25.2272 |

#### clip-hungari-1-0.5-5

将 cfg.OWOD.THRESHOLD_UNKNOWN 从 0.7 降到 0.5，mAP 和 U-Recall 都下降，说明从模型训练出来的 feature 与 text-feature 相似性阈值应该设置的更高一点比较好。

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 55.2986 |
| WI(↓)       | 0.0678  |
| A-OSE(↓)    | 5719    |
| U-Recall(↑) | 25.0900 |

#### clip-hungari-1-0.7-3

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 55.3188 |
| WI(↓)       | 0.06958 |
| A-OSE(↓)    | 6311    |
| U-Recall(↑) | 25.0471 |

## 23230324

#### clip-hungari-1-0.7-1

* **cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.7	cfg.OWOD.TOP_K_UNKNOWN = 1**

| t1           |                   |
| ------------ | ----------------- |
| mAP(↑)      | **56.4216** |
| WI(↓)       | 0.0674            |
| A-OSE(↓)    | 7750              |
| U-Recall(↑) | 25.2144           |

分析：使用 HungarianMatcher ，cfg.OWOD.TOP_K_UNKNOWN 值较大时，对 mAP 的影响较大，因此 cfg.OWOD.TOP_K_UNKNOWN = 1，比较合适。说明模型打的 unknown 伪标签不宜太多。

### Featurized-QueryRCNN + nms + Sinkhorn 闭集

| t1      |         |
| ------- | ------- |
| mAP(↑) | 61.6203 |

## 23230331

### Featurized-QueryRCNN + nms + CLIP-TEXT  + Sinkhorn 开集

#### clip-ota-1-0.7-1

* cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.7	cfg.OWOD.TOP_K_UNKNOWN = 1

| t1           |                   |
| ------------ | ----------------- |
| mAP(↑)      | **60.1783** |
| WI(↓)       | 0.0765            |
| A-OSE(↓)    | 11291             |
| U-Recall(↑) | 15.5317           |

#### clip-ota-1-0.7-3

* cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.7	cfg.OWOD.TOP_K_UNKNOWN = 3

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 59.9798 |
| WI(↓)       | 0.0708  |
| A-OSE(↓)    | 7542    |
| U-Recall(↑) | 18.9922 |

#### clip-ota-1-0.7-5

* cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.7	cfg.OWOD.TOP_K_UNKNOWN = 5

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 59.5417 |
| WI(↓)       | 0.0655  |
| A-OSE(↓)    | 6261    |
| U-Recall(↑) | 20.2058 |

分析：

* 使用 ota 替换 hungari ，mAP确实会提高，但是 U-Recall 指标效果会变差；
* ota 的 U-Recall 随着 TOP_K_UNKNOWN 设置的数值增加有变好趋势，与此同时 mAP 会略微下降。

## 20230407

由于 ota 在本实验开放设置下，最终损失从闭集设置的收敛到 1（左右）变成收敛到 8 （左右），虽然 mAP 没有大幅下降，但是损失优点奇怪，怀疑会不会是没有收敛好？

尝试将加入的损失项 loss_clip 设置为 0 ，将 ``losses['loss_clip'] = 1 - sim_matched_per_box / (2 * bs)`` 换成 ``losses['loss_clip'] = sim_matched_per_box - sim_matched_per_box``，如果直接 ``losses['loss_clip'] = torch.Tensor(0)``会报错：有参数没有参与反传。

实验结果（nohup-ota-1-0.7-5-wo-clip-loss.out）显示模型变得没有 unknow 检测的能力 ？？

难道，一定要将从模型获得的相应已知类的特征靠近 clip 编码的文本特征靠近？

```
# nohup-ota-1-0.7-5-wo-clip-loss.out
[04/02 16:56:19 d2.evaluation.pascal_voc_evaluation]: Evaluating voc_coco_2007_test using 2012 metric. Note that results do not use the official Matlab API.
[04/02 16:56:19 d2.evaluation.pascal_voc_evaluation]: aeroplane has 4986 predictions.
[04/02 16:56:21 d2.evaluation.pascal_voc_evaluation]: bicycle has 5584 predictions.
[04/02 16:56:21 d2.evaluation.pascal_voc_evaluation]: bird has 9560 predictions.
[04/02 16:56:22 d2.evaluation.pascal_voc_evaluation]: boat has 10553 predictions.
[04/02 16:56:22 d2.evaluation.pascal_voc_evaluation]: bottle has 19999 predictions.
[04/02 16:56:24 d2.evaluation.pascal_voc_evaluation]: bus has 4891 predictions.
[04/02 16:56:24 d2.evaluation.pascal_voc_evaluation]: car has 21841 predictions.
[04/02 16:56:25 d2.evaluation.pascal_voc_evaluation]: cat has 4487 predictions.
[04/02 16:56:25 d2.evaluation.pascal_voc_evaluation]: chair has 30639 predictions.
[04/02 16:56:27 d2.evaluation.pascal_voc_evaluation]: cow has 3482 predictions.
[04/02 16:56:27 d2.evaluation.pascal_voc_evaluation]: diningtable has 10008 predictions.
[04/02 16:56:28 d2.evaluation.pascal_voc_evaluation]: dog has 6129 predictions.
[04/02 16:56:28 d2.evaluation.pascal_voc_evaluation]: horse has 5218 predictions.
[04/02 16:56:29 d2.evaluation.pascal_voc_evaluation]: motorbike has 4782 predictions.
[04/02 16:56:29 d2.evaluation.pascal_voc_evaluation]: person has 56833 predictions.
[04/02 16:56:32 d2.evaluation.pascal_voc_evaluation]: pottedplant has 13248 predictions.
[04/02 16:56:33 d2.evaluation.pascal_voc_evaluation]: sheep has 3405 predictions.
[04/02 16:56:33 d2.evaluation.pascal_voc_evaluation]: sofa has 9928 predictions.
[04/02 16:56:34 d2.evaluation.pascal_voc_evaluation]: train has 7525 predictions.
[04/02 16:56:34 d2.evaluation.pascal_voc_evaluation]: tvmonitor has 9661 predictions.
[04/02 16:56:35 d2.evaluation.pascal_voc_evaluation]: truck has 1 predictions.
......
[04/02 16:56:53 d2.evaluation.pascal_voc_evaluation]: bowl has 1 predictions.
[04/02 16:56:53 d2.evaluation.pascal_voc_evaluation]: unknown has 1 predictions.
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Wilderness Impact: {0.1: {50: 0.012509197939661515}, 0.2: {50: 0.01982353615217142}, 0.3: {50: 0.038687973086627414}, 0.4: {50: 0.06672551666839754}, 0.5: {50: 0.08094857172347947}, 0.6: {50: 0.0768052269911006}, 0.7: {50: 0.07285074546819693}, 0.8: {50: 0.07717934782608696}, 0.9: {50: 0.08152190622598002}}
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: avg_precision: {0.1: {50: 0}, 0.2: {50: 0}, 0.3: {50: 0}, 0.4: {50: 0}, 0.5: {50: 0}, 0.6: {50: 0}, 0.7: {50: 0}, 0.8: {50: 0}, 0.9: {50: 0}}
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Absolute OSE (total_num_unk_det_as_known): {50: 19435.0}
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: total_num_unk 23320
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'unknown']
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: AP50: ['87.8', '59.8', '67.3', '52.1', '26.5', '73.5', '58.7', '84.2', '23.2', '74.2', '19.8', '79.5', '81.2', '70.0', '51.2', '36.0', '73.2', '53.1', '81.1', '61.9', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'nan', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'nan', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Precisions50: ['6.4', '7.9', '5.7', '3.3', '4.9', '6.8', '10.4', '10.6', '4.7', '8.1', '5.7', '10.0', '7.3', '8.9', '21.2', '4.7', '8.0', '4.2', '4.5', '5.3', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Recall50: ['94.9', '67.4', '78.5', '71.7', '45.1', '85.3', '71.4', '92.9', '41.7', '89.6', '40.8', '93.1', '92.6', '77.9', '67.4', '66.1', '85.2', '77.5', '91.5', '80.5', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'nan', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'nan', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Current class AP50: 60.72647522411283
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Current class Precisions50: 7.418903316950939
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Current class Recall50: 75.56362805219221
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Known AP50: 60.72647522411283
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Known Precisions50: 7.418903316950939
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Known Recall50: 75.56362805219221
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Unknown AP50: 0.0
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Unknown Precisions50: 0
[04/02 16:56:54 d2.evaluation.pascal_voc_evaluation]: Unknown Recall50: 0
[04/02 16:56:54 d2.engine.defaults]: Evaluation results for voc_coco_2007_test in csv format:
[04/02 16:56:54 d2.evaluation.testing]: copypaste: Task: bbox
[04/02 16:56:54 d2.evaluation.testing]: copypaste: AP for all classes,AP50 for all classes
[04/02 16:56:54 d2.evaluation.testing]: copypaste: nan,nan
/data/wxf/algorithm_code/SparseR-CNN/detectron2/evaluation/pascal_voc_evaluation.py:467: RuntimeWarning: invalid value encountered in true_divide
  rec = tp / float(npos)
```

作为参考，尝试 HungarianMatcher 时，将损失项 loss_clip 设置为 0 ：

实验结果，模型没有检测 unknown 的能力，可见这项损失的必要性。

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 58.2843 |
| WI(↓)       | 0.0659  |
| A-OSE(↓)    | 35873   |
| U-Recall(↑) | 0       |

### Featurized-QueryRCNN + nms + glove-emb + HungarianMatcher 开集

#### glove-hungari-1-0.7-1

* cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.7	cfg.OWOD.TOP_K_UNKNOWN = 1

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 57.9229 |
| WI(↓)       | 0.0650  |
| A-OSE(↓)    | 34325   |
| U-Recall(↑) | 0       |

疑点：

* Absolute OSE (total_num_unk_det_as_known): {50: 34325.0}

  total_num_unk 23320，A-OSE 的值为什么会大于 total_num_unk ？
* U-Recall 又是 0，是不是 cfg.OWOD.THRESHOLD_UNKNOWN = 0.7 设置的过大了，试试 0.5 ？

#### glove-hungari-1-0.5-1

* cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.5	cfg.OWOD.TOP_K_UNKNOWN = 1

| t1           |                   |
| ------------ | ----------------- |
| mAP(↑)      | **57.4434** |
| WI(↓)       | 0.0768            |
| A-OSE(↓)    | 10683             |
| U-Recall(↑) | 23.7178           |

#### glove-hungari-1-0.5-3

* cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.5	cfg.OWOD.TOP_K_UNKNOWN = 3 (不ok)

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 56.3522 |
| WI(↓)       | 0.0829  |
| A-OSE(↓)    | 10686   |
| U-Recall(↑) | 23.8936 |

glove-hungari-1-0.6-1

* cfg.OWOD.CLIP_LOSS_WEIGHT = 1.	cfg.OWOD.THRESHOLD_UNKNOWN = 0.6	cfg.OWOD.TOP_K_UNKNOWN = 1 (不ok)

| t1           |         |
| ------------ | ------- |
| mAP(↑)      | 58.4596 |
| WI(↓)       | 0.0647  |
| A-OSE(↓)    | 33758   |
| U-Recall(↑) | 2.4528  |

### T1-T4

losses['loss_clip'] = 1 - sim_matched_per_box / (2 * bs)  # 2 GPU，防止 loss 为负数

#### clip-hungari-1-0.7-1

| ch-1-0.7-1   | t1      | t2      | t3      | t4      |
| ------------ | ------- | ------- | ------- | ------- |
| Prev mAP(↑) |         | 47.0555 | 36.6402 | 30.5188 |
| Cur mAP(↑)  | 57.3283 | 31.4212 | 21.0477 | 15.8600 |
| Both mAP(↑) |         | 39.2383 | 31.4427 | 26.8541 |
| WI(↓)       | 0.0634  | 0.0254  | 0.0162  |         |
| A-OSE(↓)    | 6825    | 3864    | 3348    |         |
| U-Recall(↑) | 24.1852 | 19.6385 | 21.8948 |         |

#### glove-hungari-1-0.5-1

| gh-1-0.5-1   | t1      | t2      | t3      | t4      |
| ------------ | ------- | ------- | ------- | ------- |
| Prev mAP(↑) |         | 47.4479 | 37.2257 | 30.6633 |
| Cur mAP(↑)  | 57.5139 | 31.1836 | 20.8204 | 15.9746 |
| Both mAP(↑) |         | 39.3157 | 31.7573 | 26.9911 |
| WI(↓)       | 0.0686  | 0.0441  | 0.0223  |         |
| A-OSE(↓)    | 8127    | 16433   | 9749    |         |
| U-Recall(↑) | 23.7521 | 9.8103  | 10.2161 |         |

分析：

* clip-hungari-1-0.7-1   设置下，结果与 20230324 那周的结果有点不一样，因为本次的损失设置变了，改变之后clip-loss占的比重会更小一些；
* 两种文本的实验设置下，U-Recall 指标均在 t2 出现谷值；随着任务的推进，mAP 下降较多。

## 20230414

### 找找合适的超参数

#### glove-hungari-1-0.4~0.5-1

| t1           | 0.4     | **0.45**    | 0.46              | 0.47    | 0.48     | 0.49    | 0.5     |
| ------------ | ------- | ----------------- | ----------------- | ------- | -------- | ------- | ------- |
| mAP(↑)      | 57.1737 | **57.4499** | 57.4320           | 57.2505 | 57.17603 | 57.3459 | 57.5139 |
| WI(↓)       | 0.0633  | 0.0635            | 0.0643            | 0.0641  | 0.0647   | 0.0642  | 0.0686  |
| A-OSE(↓)    | 7180    | 7454              | 7278              | 7025    | 7353     | 7045    | 8127    |
| U-Recall(↑) | 24.4382 | 24.5240           | **24.7855** | 24.0437 | 24.7641  | 24.5197 | 23.7521 |

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/gh.png" style="zoom: 80%;" align="left"/>

glove-hungari-1-0.4-1，不ok，已知与未知均不如1-0.45-1

 glove-hungari-1-0.46-1，相比于0.45，mAP略低，U-Recall略 高

 glove-hungari-1-0.47-1，不ok，已知与未知均不如1-0.45-1

 glove-hungari-1-0.48-1，不ok，已知与未知均不如1-0.46-1

 glove-hungari-1-0.49-1，不ok，已知与未知均不如1-0.46-1

#### clip-hungari-1-0.6~0.8-1

| t1           | 0.6     | 0.65    | 0.66    | **0.67**    | 0.68    | 0.69    | 0.7     | 0.71    | 0.73    | 0.75    | 0.8     |
| ------------ | ------- | ------- | ------- | ----------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| mAP(↑)      | 56.9559 | 56.5979 | 57.2657 | **57.5300** | 57.4844 | 57.1821 | 57.3283 | 57.3468 | 57.2142 | 56.9670 | 57.0046 |
| WI(↓)       | 0.0657  | 0.0625  | 0.0640  | 0.0630            | 0.0620  | 0.0647  | 0.0634  | 0.0623  | 0.0628  | 0.0644  | 0.0630  |
| A-OSE(↓)    | 6950    | 6630    | 6911    | 6927              | 6702    | 7053    | 6825    | 7038    | 7031    | 6912    | 7401    |
| U-Recall(↑) | 24.7684 | 24.2710 | 24.6054 | 24.3224           | 24.4125 | 24.0308 | 24.1852 | 24.0351 | 24.5454 | 24.9013 | 24.0823 |

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/ch.png" alt="ch" style="zoom:80%;" align="left"/>

## 20230421

### t1-t4

#### glove-hungari-1-0.46-1

| gh-1-0.46-1  | t1                | t2      | t3      | t4      |
| ------------ | ----------------- | ------- | ------- | ------- |
| Prev mAP(↑) |                   | 47.9531 | 37.3139 | 30.7533 |
| Cur mAP(↑)  | 57.4320           | 32.1108 | 20.9634 | 15.9172 |
| Both mAP(↑) |                   | 40.0320 | 31.8637 | 27.0442 |
| WI(↓)       | 0.0643            | 0.0418  | 0.0224  |         |
| A-OSE(↓)    | 7278              | 9496    | 8407    |         |
| U-Recall(↑) | **24.7855** | 16.7044 | 15.1521 |         |

#### clip-hungari-1-0.67-1

| ch-1-0.67-1  | t1                | t2      | t3      | t4      |
| ------------ | ----------------- | ------- | ------- | ------- |
| Prev mAP(↑) |                   | 46.7834 | 36.1308 | 30.4104 |
| Cur mAP(↑)  | **57.5300** | 31.6454 | 21.1934 | 15.6453 |
| Both mAP(↑) |                   | 39.2144 | 31.1516 | 26.7191 |
| WI(↓)       | 0.0630            | 0.0251  | 0.0153  |         |
| A-OSE(↓)    | 6927              | 3816    | 3233    |         |
| U-Recall(↑) | 24.3224           | 20.7299 | 21.1850 |         |

↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓一次意外的超参错误实验结果↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

t1：clip-hungari-1-0.67-1

t2-t3：glove-hungari-1-0.46-1

| ch-1-0.67-1  | t1                | t2      | t3      | t4      |
| ------------ | ----------------- | ------- | ------- | ------- |
| Prev mAP(↑) |                   | 47.0971 | 36.4644 | 30.9857 |
| Cur mAP(↑)  | **57.5300** | 31.5812 | 20.6692 | 16.6973 |
| Both mAP(↑) |                   | 39.3392 | 31.1993 | 27.4136 |
| WI(↓)       | 0.0630            | 0.0453  | 0.0220  |         |
| A-OSE(↓)    | 6927              | 13587   | 8336    |         |
| U-Recall(↑) | 24.3224           | 14.4262 | 15.1414 |         |

作为对比：

t1：glove-hungari-1-0.46-1

t2-t3：clip-hungari-1-0.67-1

| ch-1-0.67-1  | t1 | t2      | t3      | t4      |
| ------------ | -- | ------- | ------- | ------- |
| Prev mAP(↑) |    | 47.1674 | 36.3941 | 30.4265 |
| Cur mAP(↑)  |    | 31.4041 | 21.2362 | 16.2398 |
| Both mAP(↑) |    | 39.2858 | 31.3415 | 26.8798 |
| WI(↓)       |    | 0.0265  | 0.0165  |         |
| A-OSE(↓)    |    | 3868    | 3555    |         |
| U-Recall(↑) |    | 19.2986 | 21.8410 |         |

分析：中途换文本提示，模型会立马将特征空间从前种文本特种空间转向后种特征空间（表现为训练t2的时候，文本约束损失会重新由0.9左右降到0.5左右）。

↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑一次意外的超参错误实验结果↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

下一步实验计划：

1. 用文本特征初始化类别原型作为prompt-proto
2. 考虑用物理因素为unknown打标签
3. 考虑将QGN中的约束换成纯边框约束，试试能否提升定位准确度

## 20230505

1、用文本特征初始化类别原型作为prompt-proto（希望通过使相同类的特征靠近+不同类的特征空间分散，提高已知类的检测精度）：

1. 对于预测为当前某类别的边框特征，通过聚合损失$loss_close = L_{1}(pred\_feature, prompt\_proto)$使特征靠近相应的提示原型，在迭代 500 iter后，每 20 iter 通过 $prompt\_proto= prompt\_proto * 0.9 + pred\_feature * 0.1$更新prompt-proto；
2. 对于不同类的prompt-proto，在上述更新 iter 后，计算损失 $loss_{dis}$(损失公式参考文章20)
3. 在训练t2_train的时候，将和前面20个类的prompt-proto相似度最高的打上相应的伪标签（对应类别），在fine-tune之前，对于pre-known的精度提升很小（0.06）。

存在的问题：没啥效果

1. $loss_{dis}$不收敛（基本不降低），没起到作用，可能是因为 prompt-proto 就算更新后，和原始的文本特征还是很相似（相似度大概在0.8-09，说明更新幅度不大），因此$loss_{dis}$起不到作用。

|              | t1      | t2      | t3 | t4 |
| ------------ | ------- | ------- | -- | -- |
| Prev mAP(↑) |         | 48.4426 |    |    |
| Cur mAP(↑)  | 58.2679 | 30.7083 |    |    |
| Both mAP(↑) |         | 39.5755 |    |    |

下一步，考虑将prompt-proto在推理阶段利用起来 ————>也没什么用。。。

outputs_class ：模型原本的cls预测

logits_prompt：pred_freature和prompt_proto相似性得分

| 0.5 * outputs_class + 0.5 * F.softmax(logits_prompt)           | t1           | t2           | t3 | t4 |
| -------------------------------------------------------------- | ------------ | ------------ | -- | -- |
| Prev mAP(↑)                                                   |              | 48.4381      |    |    |
| Cur mAP(↑)                                                    | 58.2714      | 30.7055      |    |    |
| Both mAP(↑)                                                   |              | 39.5718      |    |    |
| **0.7 * outputs_class + 0.3 * F.softmax(logits_prompt)** | **t1** | **t2** |    |    |
| Prev mAP(↑)                                                   |              | 48.4344      |    |    |
| Cur mAP(↑)                                                    | 58.2694      | 30.7240      |    |    |
| Both mAP(↑)                                                   |              | 39.5792      |    |    |

下一步计划：

看一下文章19的代码，是如何提取物理线索的。

## 20230512

不叠任何 buff 的 Featurized Q-RCNN 增量性能（作为对比）

|              | t1      | t2      | t3      | t4      |
| ------------ | ------- | ------- | ------- | ------- |
| Pre mAP(↑)  |         | 48.1939 | 37.1835 | 30.5263 |
| Cur mAP(↑)  | 58.1143 | 32.4071 | 22.1830 | 16.3368 |
| Both mAP(↑) |         | 40.3005 | 32.1834 | 26.9789 |

基于上周实验的改进：

1. feature 的更新方式 ——>使用队列保存每个类别最新的20个特征样例，用该20个特征样例的平均值的 0.1 倍来更新prompt-proto；
2. $loss_{dis}$ ——>改为对特征和 prompt-proto 归一化后使用平均 L2 距离（既考虑特征和 prompt-proto 的整体距离又考虑角度距离，因为归一化之后优化 L2 距离相当于优化余弦距离）

|              | t1      | t2      | t3      | t4      |
| ------------ | ------- | ------- | ------- | ------- |
| Pre mAP(↑)  |         | 48.8296 | 37.3603 | 30.5001 |
| Cur mAP(↑)  | 58.2783 | 32.0554 | 21.6313 | 16.0219 |
| Both mAP(↑) |         | 40.4425 | 32.1173 | 26.8806 |

下一步计划：把增量的实验跑出来看看结果，然后加相似性矩阵约束损失

增量过程中使用 prompt-proto 对于已知类的检测精度影响：pre mAP 会有零点几的提升，cur mAP 会有零点几的下降，Both mAP差别不大。

## 20230519

相似性矩阵蒸馏

对于识别为已知类的预测框，其特征与对应类别的文本特征计算相似性得到 $m_{pt}$

对 gt 信息中，gt box 对应的图片部分裁剪送入 CLIP image encoder 编码得到 $g_{img}$，gt label 信息送入 CLIP text encoder 编码得到 $g_{text}$，计算  $g_{img}$ 和 $g_{text}$ 的相似性矩阵 $m_{gt}$

损失 $L_{kd} = \| m_{gt} - m_{pt}\|_{F}$    (Frobenius norm)

 $L_{kd}$ 以0.5的权重加到总损失上（$L_{total} = L_{F} + L_{close} + 0.5L_{kd}$）

|              | t1      | t2      | t3      | t4      |
| ------------ | ------- | ------- | ------- | ------- |
| Pre mAP(↑)  |         | 49.7219 | 38.0595 | 31.9715 |
| Cur mAP(↑)  | 58.1735 | 31.5323 | 21.4993 | 16.0612 |
| Both mAP(↑) |         | 40.6271 | 32.5394 | 27.9940 |

相较于加 $L_{kd}$ 前，除了t1，后续任务上，mAP都有提升。（对比baseline，除个别指标，总体上都有提升。）

后续计划，先调调损失系数，由于 $L_{kd}$ 借助了 CLIP 的对齐功能，所以后面可以试试在推理的时候使用 CLIP 文本做zero-shot分类的得分作为辅助预测。

## 20230526

0.7 $L_{kd}$（不ok）

|              | t1      | t2      | t3      | t4      |
| ------------ | ------- | ------- | ------- | ------- |
| Pre mAP(↑)  |         | 48.6577 | 37.2258 | 31.3740 |
| Cur mAP(↑)  | 58.2086 | 31.5722 | 21.4958 | 16.3605 |
| Both mAP(↑) |         | 40.1150 | 31.9824 | 27.6206 |

**受启发于 king-man+woman=queen**

是不是可以用已知类的特征组合来表示未知类呢？

初步想法，使用所有已知类的特征的平均值（或者线性组合，其中权重设置为可学习参数）作为未知类的特征表示，用于给预测框打上 unknown 的伪标签。

* 首先，用已知类的特征（CLIP编码的图像特征）的平均值作为未知类的特征表示称为 fused_feat ，在匈牙利匹配后，未匹配的候选框中若挑选与 fused_feat 相似性最大、且相似性大于某一阈值（0.5）的标为 unknown 类（label：80）

    $L_{total} = L_{F} + L_{close} + 0.5L_{kd}$

    结果分析：（阈值0.5可能太高了，模型不具备检测 unknown 的能力，甚至已知类的检测性能还会下降）

* 阈值设成0：

    结果分析：

    训练 iter = 54000

    mAP在逐渐提升，每隔6000次评估的结果分别为: 33.5151 -> 42.1409 -> 47.7817 -> 47.3875 -> 50.7755 -> 51.5424 -> 55.4549 -> 		56.1285 -> 56.3453

    U-Recall 处于波动状态，每隔6000次评估的结果分别为: 4.2795 -> 12.8001 -> 7.9502 -> 9.8241 -> 10.3516 -> 8.7950 -> 13.8036 -> 		6.6552 -> 8.7521（考虑打伪标签的方式可能有问题，待改进）

    A-OSE 太高了，模型不能很好地区分未知类和已知类的特征

    首先考虑提高训练 iter 试试：iter = 66000

    增加训练 iter 后，mAP变化：-> 56.5027 -> 7.9716    U-Recall：-> 56.3387 -> 10.3301

|              | t1      | t2 | t3 | t4 |
| ------------ | ------- | -- | -- | -- |
| Prev mAP(↑) |         |    |    |    |
| Cur mAP(↑)  | 56.3453 |    |    |    |
| Both mAP(↑) |         |    |    |    |
| WI(↓)       | 0.0670  |    |    |    |
| A-OSE(↓)    | 30264   |    |    |    |
| U-Recall(↑) | 8.7521  |    |    |    |

后续计划：重新考虑下原型的设置方式和文本的使用方式，感觉用已知类的特征组合表示未知类应该是可行的，使用方式也有待改进。

## 20230602

使用 one-hot 向量初始化类原型，使用 yu 的方式更新原型

用 已知类原型的线性组合 fused_feat（权重是可学习参数）表征 unknown 类的原型，未与当前 gt 类匹配的预测框特征若与 fused_feat 相似性超过一定阈值 $\delta$，则打上 unknown 的伪标签。

（unknown 类的原型只参与 $loss_{dis}$ 的计算）

两个 loss：

将特征归一化映射到球面空间，用 $loss_{close}$ 拉近同类的距离；用 $loss_{dis}$ 远离不同类的距离；

| 参数：$\delta$ = 0 | t1      | t1(0.1) | t2 | t2(0.1) | t3 | t3(0.1) | t4 |
| -------------------- | ------- | -- | -- | -- | -- | -- | -- |
| Prev mAP(↑)         |         |         | 46.8807 | 45.6656 | 35.8001 | 35.1958 | 30.0777 |
| Cur mAP(↑)          | 57.0520 | 56.3452 | 30.1277 | 29.5897 | 20.8213 | 20.5145 | 16.3025 |
| Both mAP(↑)         |         |         | 38.5042 | 37.6276 | 30.8071 | 30.3020 | 26.6339 |
| WI(↓)               | 0.0642  | 0.0525 | 0.0334 | 0.0207 | 0.0192 | 0.0152 |    |
| A-OSE(↓)            | 32080   | 5519 | 18070 | 2857 | 11120 | 3268 |    |
| U-Recall(↑)         | 11.5780 | 20.1715 | 9.5896 | 14.1698 | 8.2589 | 17.9911 |    |

(在推理的时候可以通过设置阈值（如0.01）的方式降低模型将未知类识别为已知类，（例如将预测分数小于0.1的框的预测标签改为unknown）)

相比于上周的方式， mAP 和 U-Recall 都有变好

原因分析：之前用CLIP编码的文本特征作为类原型时。各类之前的距离很难拉开，$loss_{dis}$不收敛；原型初始化改为one-hot向量后， $loss_{dis}$ 和 $loss_{close}$ 在训练过程中均左在变，即起了作用。



mAP在逐渐提升，每隔6000次评估的结果分别为: 36.7275 -> 44.7829 -> 48.2822 -> 49.0676  -> 51.3694 -> 52.4868  -> 56.3887 -> 56.9019 -> 57.0520

U-Recall 的波动算是平缓了？每隔6000次评估的结果分别为: 12.2813 -> 18.6535 -> 12.9245 -> 12.7873 -> 12.3799  -> 10.8704 -> 10.8919 -> 11.5351 -> 11.5780

> 对比实验（ 参数 $\delta$ 设置为动态的，即 cur_iter/100000，随着迭代次数的增大，打 unknown 伪标签的条件越严苛）

> mAP：34.6219 -> 44.1900 -> 48.6985 -> 47.5945 -> 51.6841 -> 51.8227 -> 55.7686 -> 56.6501 -> 56.8743

> U-Recall：14.6740 -> 16.3164 -> 12.5600 -> 13.5677 -> 14.4725 -> 12.4056 -> 13.5677 -> 12.5042 -> 12.3499

> A-OSE: 31649 , WI: 0.0627

> 分析：这种逐渐严苛的伪标签标注方式貌似有点 work。

Mark：由于 mAP 尚在上升趋势，后面可以考虑提高训练 iter

> 下一步计划：关注特征学习的方法，如何学习紧凑的特征

## 20230609

代码修改：

* 原来的class_logits 的输出维度，由 81 改为 t123：known + 1，t4：80
* 之前的loss_dis 应该是有误（loss_dis = 1- sim_ij，sim_ij 表示原型 i 和 原型 j 的相似性），现在改为 loss_dis =  sim_ij，并在 1020 iter 之后开始计算 loss_dis（原型更新的开始 iter = 1000）

|             | t1      | t2      | t3      | t4      |
| ----------- | ------- | ------- | ------- | ------- |
| Prev mAP(↑) |         | 44.0985 | 32.9018 | 26.8434 |
| Cur mAP(↑)  | 56.6980 | 29.8738 | 19.9360 | 15.7362 |
| Both mAP(↑) |         | 36.9861 | 28.5798 | 24.0666 |
| WI(↓)       | 0.0623  | 0.0334  | 0.0217  |         |
| A-OSE(↓)    | 30314   | 18516   | 12400   |         |
| U-Recall(↑) | 12.4142 | 11.7903 | 9.8505  |         |

分析：loss_dis （不收敛）维持在0.99左右，说明各类原型的 cosine 相似度非常高（夹角几乎为0，之前错误的 loss_dis 也是和这个现象一致），loss_dis没有起到远离各类原型的作用。

mAP 为啥会下降呢？

修改：让 loss_close 和 loss_dis 均从夹角（cosine_similarity）和距离（l2）的角度考虑

