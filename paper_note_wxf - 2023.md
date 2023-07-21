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

- **OLN-RPN**：以来自ResNet特征金字塔的特征作为输入，每个特征图都经过一个卷积层，然后是两个独立的层，一个用于边界框回归，另一个用于定位质量预测。网络架构的设计遵循了标准的 RPN 。选择中心性作为定位质量目标，并用 L1 损失训练两个头。在提议阶段学习定位而不是分类是至关重要的，因为它可以避免通过分类过度拟合前景。

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

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230315192850364.png" alt="image-20230315192850364" style="zoom:75%;" align="left"/>如图，DeFo 的构成：

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

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230403191235478.png" alt="image-20230403191235478" style="zoom:67%;" align="left" />`                                                                                                                                                                                                                                                                                                                                                           并针对以上三个挑战，分别提出相应措施。即

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

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230514220939801.png" alt="image-20230514220939801" style="zoom:80%;" />

**Heterogeneous Label Space Training**

使用多种标签空间的组合数据训练模型的三种方式：

a.  seperate label spaces，各模型在单个数据集上单独训练，然后分别对测试数据进行评估，最终结果由所有模型的结果结合得到；

b.  unified label space，将所有标签映射到同一标签空间；

c.  partitioned label space，所有数据集共享 backbone，但有各自的分类层。在推理时，利用测试label的class embedding 即可避免标签冲突。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230514222546399.png" alt="image-20230514222546399" style="zoom:80%;" />

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

![image-20230516110516969](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230516110516969.png)

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
<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230516133933577.png" alt="image-20230516133933577" style="zoom:80%;" align="left"/><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230516134043594.png" alt="image-20230516134043594" style="zoom:67%;" align="left"/> 



























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

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main\image-20230524101246340.png" alt="image-20230524101246340" style="zoom:67%;" />

**Region Prompting：**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main\image-20230524145603791.png" alt="image-20230524145603791" style="zoom:80%;" align="left"/>  

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













## 20230609

### 35_Improving Visual Representation Learning through Perceptual Understanding_ CVPR 2023 投稿_代码暂无

> 作者：

Samyakh Tukra, Frederick Hoffman, Ken Chatfield  单位：Tractable AI

> 贡献：

本文提出了用于 MAE（masked autoencoders） 的扩展，通过明确地鼓励模型学习高层次的特征来学习高质量的表示。

* 首先，本文引入生成图像和真实图像之间的感知相似性损失
* 利用对抗训练中的一些技术如 multi-scale training 以及 adaptive discriminator augmentation

结果：不仅做到了更好的像素级重构，还能更好地捕获图像中的高层次特征细节。

> 方法：

本文建立在 MAE 的基础上，探讨如何跨越高遮蔽比的障碍，明确地将高阶“语义”特征的学习纳入学习目标。

#### 3.1 MAE with Perceptual Loss

**原始 MAE** 的像素重构损失加上感知损失表示为：
$$
L^{G} = \| G(I_{m} - I) \|_{1} + L^{G}_{perceptual} \tag{1}
\\
G 是 MAE model，I是原始图像，I_{m} 是原始 I 经随机 masked 后的图像；第一项为 L_{1} 损失，第二项为感知损失
$$
**MS-SSIM：**本文的 baseline 感知损失为 MS-SSIM ( multi-scale structural similarity index)，多尺度分量有助于减少在输出重建图像I‘的边缘周围形成的伪影，感知损失项为： 
$$
L_{ssim}^{G} = \frac{1}{N} \sum_{i,j} \alpha \frac{1 - SSIM(G(I_{m})_{ij}, I_{ij})}{2} \tag{2}
\\
N 为scales数，设置为4，\alpha 是感知损失权重，同时 L_{1} 损失的权重设置为 1- \alpha
$$
**Feature matching：**第二项感知损失建立在一个单独的 loss network $\Phi$ 的基础上，鼓励 decoder 的每一层特征尽量保持与 $\Phi$ 的对应层特征相似。在本文中 $\Phi$ 为一个额外的判别网络 D，在对抗训练的设置中，区分重建图像和原始图像。直觉上是，通过这个任务学习到的特征包含了更高阶的知觉线索，可以用来指导解码器的训练。感知损失包含两个部分：
$$
L_{feat}^{G} = \delta_{f} \sum_{j=1}^{J} \frac{1}{N_{j}}[\| \Phi^{j}(G(I_{m})) - \Phi^{j}(I)\|_{1}] + \delta_{s} \sum_{j=1}^{J} \frac{1}{N_{j}} [\| \Psi(\Phi^{j}(G(I_{m}))) - \Psi(\Phi^{j}(I)) \|_{1}] \tag{3}
\\
j 是layer 索引，N_{j} 表示第 j 层的元素数， \Psi为Gram matrix function，\delta 为相应的权重；
\\
第一项迁移高层次语义，第二项迁移不同通道特征的相似性。
$$
此外，对抗性损失被添加到 $L_{G}$ 中。任何对抗性损失函数都可以使用，但在本文的基线实验中使用了 LS-GAN，它在原始的最小-最大分类损失上实现了更稳定的优化。相应的 generator-discriminator 损失对为：
$$
L_{adv}^{D} = \frac{1}{2}[(D(I) - 1)^{2}] + [D(G(I_{m}))^{2}] \tag{4}
$$

$$
L_{adv}^{G} = \frac{1}{2} [(D(G(I_{m})) - 1)^{2}] \tag{5}
$$

所以，decoder 的总 $L_{G} $:
$$
L_{G} = \| G(I_{m}) - I \|_{1} + L_{feat}^{G} + L_{adv}^{G} \tag{6}
$$
**dVAE perceptual：**利用 dVAE 作为感知学习的 baseline，使用公式(3)结合 dVAE 作为 loss network $\Phi$。

#### 3.2. Adversarial Training Variants

对于基于特征匹配的感知学习，可以使用任何对抗性损失函数。除了在我们的基线模型中使用的 LSGAN 损失外，我们还对两个进一步的变体做了实验。这两种方法都是为了解决原始 GAN 的问题，如训练不稳定性和模式崩溃。我们假设，通过这些方法学习到的更丰富的分布将为知觉学习提供更强的线索。

* **MSG-GAN：**为了稳定生成器的训练，MSG-GAN 允许梯度在多个尺度上从鉴别器流向生成器。这是通过添加从生成器的中间层到鉴别器的中间层的 skip 连接来实现的。训练D和训练G的损失函数保持不变。
* **StyleGANv2-ADA：**我们采用了StyleGANv2-ADA 中的鉴别器中所有的修改。在MSG-GAN的基础上，在解码器输入和鉴别器特征映射之间添加了感知路径正则化。对训练期间的所有样本使用自适应鉴别器增强。训练损失函数D和G保持不变。

#### **3.3. Model Architecture**

多尺度 GAN 用于 MSG-GAN 和 StyleGANv2-ADA 方法的一个问题是，多尺度学习发生通过 skip 连接鉴别器D和解码器G，这意味着只有解码器受益于多尺度梯度惩罚训练，并在预训练之后删除解码器，只有编码器用于下游任务。

为了在编码器和解码器之间更均匀地分配学习内容，类似于U-Net，我们另外引入了中间编码器和解码器层之间的跳过连接，如图1所示 **MSG-MAE** 结构。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230606180347041.png" alt="image-20230606180347041" style="zoom:80%;" />

### 36_Vocabulary-free Image Classification_CVPR 2023 投稿_有代码

作者：

Alessandro Conti, Enrico Fini, Massimiliano Mancini, Paolo Rota, Yiming Wang, Elisa Ricci

贡献：

背景：大规模视觉语言模型的出现改变了图像分类的范式并展现了不错的 zero-shot 性能，但需要假设在测试时存在一个预定义好的类别 vocabulary 作为 textual prompts。然而当语义信息事先不知道或者是不断发展时，这种假设就不切实际了。

因此，本文提出一个新的任务：Vocabulary-free Image Classification (**VIC**)。VIC 的目标是在没有预定义的词汇表前提下，为给定的一张图像分配一个类别。该任务存在以下三个挑战：

* 目标类别搜索空间巨大
* 可能包含模型难以区分的细粒度概念
* VIC 需要模型不依赖于任何 vocabulary-aware 的监督

在本工作中，我们首先通过实证验证了利用外部视觉语言数据库来表示语义空间是获得与语义相关的内容进行图像分类的最有效的方法。然后，提出从外部数据库的类别搜索（**CaSED**）方法，利用预先训练过的视觉语言模型和外部视觉语言数据库以无需训练的方式处理 VIC 问题。CaSED首先根据从数据库中检索到的标题与图像的语义相似性，从这些标题中提取一组候选类别，然后根据相同的视觉语言模型将最佳匹配的候选类别分配给该图像。

方法：

作者观察到据大规模的视觉语言数据集 Vision-Language Databases (VLDs) 如 PMD 覆盖了一般到特殊的广泛的语义空间，并且比 caption 能够提供与目标类别更相关的语义内容。因此作者提出无需训练的 CaSED，首先为给定的测试图像粗略地选择一些候选类别，然后通过多模态匹配预测最终的类别。

![image-20230607150050288](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230607150050288.png)

#### **Semantic space representation：**

作为 VIC 任务的主要挑战，如何表示巨大的语义空间至关重要。作者杜比了两种方法：1、直接使用带有自回归语言解码器的 VLM 模型（选择当前最SOTA的模型 BLIP-2）；2、通过 image-text 从 VLDs 中检索（选择PMD的一个子集）。

* *BLIP-2 VQA*：给定图像，直接询问 BLIP-2 模型该图像的类别
* *BLIP-2 Captioning*：向 BLIP-2 模型询问该图像的 Caption
* *Closest Caption*：从 database 中选择与图像最相似的描述
* *Caption Centroid*：从 database 中选择与图像最相似的 10 个描述，取平均值。

初步试验结果：如图2所示，与 VQA-enabled VLMs 相比，用 VLDs 表示大语义空间可以（通过检索）产生与输入图像语义更相关的内容，同时计算效率较高。基于此，作者提出 CaSED。

![image-20230607153758492](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230607153758492.png)

#### **CaSED: Category Search from External Databases**

如图3，CaSED是通过从大型 VLDs 的多模态数据在无约束语义空间中寻找最佳匹配类别。首先从数据库中检索语义上最相似的 Caption，并通过应用文本解析和过滤技术从中提取一组候选类别。然后进一步使用预先训练好的 VLM （即CLIP）的多模态对齐功能对候选对象进行评分，以获得最佳匹配的类别。

![image-20230607155850699](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230607155850699.png)

**第一步：Generating candidate categories**，将非常大的分类空间限制为几个最有可能的候选类。首先选择与输入图像最接近的 top-K个描述 $D_{x} = \underset{d \in D}{top_{k}} f_{VLM}(x,d) = \underset{d \in D}{top_{k}} <f_{VLM}^{v}(x), f_{VLM}^{v}(d)>$，然后通过文本解析和过滤从中提取出候选类别 $C_{x}$。

**第二步：Multimodal candidate scoring**

**Image-to-text score**： $s_{c}^{v} = <f_{VLM}^{v}(x), f_{VLM}^{t}(c)>$，该值越高，表示目标图像和候选类之间的对齐距离越近

**Text-to-text score**：虽然图像到文本的得分是有效的，但不同模态在空间Z中存在 gap，如图2所示，检索到的 Captions 的语义相关性，特别是它们的质心，与潜在的 gt 标签更相关。利用这一属性，引入单模态文本到文本的评分，以减轻跨模态评分的模态差距：$s_{c}^{t} = <\overline{d}_{x}, f_{VLM}^{t}(c)>$，其中 $\overline{d}_{x} = \frac{1}{K} \sum_{d \in D_{x}} f_{VLM}^{t}(d)$

**Final predicted candidate**：$s_{c} = \alpha \sigma(s_{c}^{v}) + (1-\alpha) \sigma(s_{c}^{t})$

## 20230616

### 37_Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts_ICML 2022_有代码

> 作者：

Yan Zeng, Xinsong Zhang, Hang Li.

> 贡献：

背景：大多数现有的视觉语言预训练方法都依赖于通过目标检测提取的以目标为中心的特征，并在提取的特征和文本之间进行细粒度的对齐。这些方法难以学习多个目标之间的关系。

为此，我们提出了一种名为 **X-VLM** 的新方法来进行 "**多粒度视觉语言预训练**"。学习多粒度对齐的关键是在给定相关文本的情况下，在图像中定位视觉概念，同时将文本与视觉概念对齐，其中的对齐是多粒度的（对象、区域和整张图像）。

实验结果表明，X-VLM在许多下游视觉语言任务中有效地利用了所学到的多粒度对齐，并优于最先进的方法。

> 方法：

X-VLM由一个图像编码器 (Itrans)、一个文本编码器 (Ttrans) 和一个跨模态编码器 (Xtrans) 组成。所有编码器都基于 Transformer，跨模态编码器通过在每一层的交叉注意，将视觉特征与语言特征融合。

作者重新制定了广泛使用的预训练数据集，这样一个图像可能有多个边界框，每个边界框都与描述一个对象或一个区域的文本相关联，表示为 $(I,T,{(V_j,T_j)}^N)$。有些图像没有相关的文本，即 $T$ 是 NaN，有些图像没有边界框，即 $N = 0$。$V_j$ 是边界框 $b_j = (c_x,c_y,w,h)$ 中的一个对象或区域，由边界框的归一化中心坐标、宽度和高度表示。当图像本身代表一个视觉概念时，$b = (0.5,0.5,1,1)$。

![image-20230608101021405](https://raw.githubusercontent.com/yuki1ssad/typora_images/main\image-20230608101021405.png)

#### Vision Encoding

图像编码器可以有效地在图像中产生多粒度的视觉概念表示。该编码器基于视觉Transformer，首先将一个图像分割成不重叠的块，并将这些块线性映射成向量。然后，这些块被传递到 Transformer 层，产生 $\{v_1,...,v_{N^I}\}$ 。对于分辨率为 224x224、块大小为 32x32 的图像，$N^I = 49$。

假设 $v_{p_i}$ 是对相应的块 $p_i$ 的信息进行编码，因此可以通过对块之间的信息进行聚合来对应于一组块，表示一个视觉概念 $V_j$ (对象、区域或图像)。具体来说，在保留块特征位置信息的同时对块特征进行重塑，表示为$\{v_{p^j_1},...,v_{p^j_M}\}$。$\{p^j_1,...,p^j_M\}$是 $V_j$ 的块。同时计算特征的平均值来表示整个视觉概念，表示为$v^j_{cls}$。

然后，图像编码器在不同的粒度下创建 N+1 个的概念表示，表示为 $Itrans(V_j)= \{v^j_{cls},v_{p^j_1},...,v_{p^j_M}\}，j∈[0,N]$。$Itrans(V^0)$ 表示所有块特征都被利用的图像表示。

#### **Cross-Modal Modeling**

**Bounding Box Prediction：**给定图像表示和文本表示，让模型预测视觉概念 $V_j$ 的边界框 $b_j$，其中$b_j = (c_x,c_y,w,h)$。通过在同一图像中定位不同的视觉概念，期望模型能够更好地学习细粒度的视觉语言对齐。边界框的预测为：
$$
\hat{b}^{j}(I,T^{j}) = Sigmoid(MLP(x_{cls}^{j}))
\\
x_{cls}^{j}为跨膜态编码器输出的[CLS]
$$
损失使用的是 $l_1$ 损失和交并比 (IoU) 损失的线性组合。

**Contrastive Learning：**预测 (视觉概念, 文本) 对，表示 $(V,T)$。视觉概念包括对象、区域和图像，随机抽取N对，并计算视觉到文本相似度和文本到视觉相似度。给定一对 $(V,T)$，$T$ 是 $V$ 的正样本，将 batch 中的其他$(N−1)$ 对视为负样本。

定义余弦相似度 $s(V,T) = g_v(v_{cls})^T \times g_w(w_{cls})$，$w_{cls}$ 是文本编码器的输出 [CLS] 嵌入。$g_v$ 和 $g_w$ 是将 [CLS] 嵌入映射到规范化的低维表示的转换。然后，计算批处理视觉文本相似度为：
$$
p^{v2t}(V) = \frac{exp(s(V,T)/\tau)}{\sum_{i=1}^{N}exp(s(V,T^{i})/\tau}
$$
同样，文本视觉相似度为：
$$
p^{t2v}(V) = \frac{exp(s(V,T)/\tau)}{\sum_{i=1}^{N}exp(s(V^{i},T)/\tau}
$$
对比损失定义为 p 和 one-hot GT y 之间的交叉熵 H:
$$
L_{cl} = \frac{1}{2} \mathbb{E}_{V,T ∼D}[H(y^{v2t}(V), p^{v2t}(V)) + H(y^{t2v}(V), p^{t2v}(V))]
$$
**Matching Prediction：**确定一对视觉概念和文本是否匹配，对于 batch 中的每个视觉概念，通过 $p^{v2t}(V)$ 对批内 hard negative 文本进行采样，与该概念更相关的文本更有可能被抽样。同时为每个文本采样一个 hard negative 样本的视觉概念，使用 $x_{cls}$，即跨模态编码器的输出 [CLS] 嵌入，来预测匹配概率 $p_{match}$，损失为：
$$
L_{match} = \mathbb{E}_{V,T ∼D} H(y^{match}, p^{match}(V, T)) 
\\
y^{match}是二维 one-hot 向量代表 gt
$$
**Masked Language Modeling：**根据视觉概念来预测文本中的掩蔽词，以 25% 的概率随机屏蔽输入标记，替换为 10% 的随机标记，10%不变，80% [MASK]。使用跨模态编码器的输出，并附加一个线性层，然后是softmax进行预测。设 $\hat{T}$ 表示一个掩码文本，$p^{j}(V,\hat{T})$ 表示掩码 token $t_{j}$ 的预测概率，最小化交叉熵的损失：
$$
L_{mlm} = \mathbb{E}_{t_{j} ∼ \hat{T};(V,\hat{T})∼D}H(y^{j}, p^{j}(V,\hat{T}))
$$
最后，X-VLM的预训练目标为：$L = L_{bbox} + L_{cl} + L_{match} + L_{mlm}$

>  总结：

1. 对比学习中既考虑到了图像-文本方向，也考虑到了文本图像方向，应该是能更好地对齐两个模态的特征空间。

### 38_Efficient Multimodal Fusion via Interactive Prompting_CVPR 2023_代码

>  作者：Yaowei Li, Ruijie Quan, Linchao Zhu, Yi Yang

> 贡献：

背景：随着预训练大模型的规模越来越大，将大模型应用到下游任务时微调的成本也越来越高。

(<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230614094214509.png" align="left"/>)   

   



当前主流方法：

*  微调：能获得比较好的性能，但是由于微调时要更新所有预训练好的参数所有计算代价很大。
*  prompt-based：保持预训练好的大模型参数不变，训练时只更新 prompts。效果不如微调的好。

 左图展示了本文方法和微调以及之前 prompt-based 方法在训练时内存占用和分类精度方面的比较。







因此本文提出一个**基于 prompt** 高效且灵活的融合多个单模态 Transformer 模型的架构 **PMF**，通过**将普通的 prompts 分解成三种不同类型的 prompts**，从而分别达到跨膜态学习中的不同优化目标。由于本文的融合方法**只应用在 Transformer 结构的深层 layer 中**，所以在内存占用方面能够非常高效。

实验结果表明，该方法与多模态微调方法相比，在可训练参数量不到后者 3%、训练内存占用节省 66%的情况下达到了相当的性能。

> 方法：

**3.1. Unimodal Transformers**

对于单模态 Transformer，图像是先被划分成  $N_{img}$ 个不重叠的 patches，并获得相应的 embedding $z_{i} \in \mathbb{R}^{d}$；对于文本，原始文本会被 token 成 $N_{txt}$ one-hot 向量然后编码得到 $N_{txt}$ 个文本 embedding。两种模态的 embedding 共享如下相同的结构：
$$
z = [CLS, z_{1}, z_{2}, ... ,z_{N}]
$$
**3.2. Unimodal Base feature Extraction**

![image-20230614102209401](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230614102209401.png)

在 Transformer 结构中的后面几层才是多模态融合层，因此单模态输入 embedding 首先通过各自的浅层 Transformer layer 获得基础特征（假设开始融合的层为 $L_{f}$）：
$$
z^{l+1} = TransLayer^{l}(z^{l};\theta) \quad \quad if \quad l < L_{f} \tag{2} \\
\theta 是预训练好的参数
$$
**3.3. Multimodal Fusion Layer**

所提取的单模态基础特征然后会经过多模态融合层，多模态融合层包括 “query stage” 和 “fusion stage”。

![image-20230614102236250](https://raw.githubusercontent.com/yuki1ssad/typora_images/main\image-20230614102236250.png)

三种 query prompt & 非线性层：

* 查询上下文提示词 QCP（$z_{qcp}$）—— ”提供答案“
* 查询提示词 QP（$z_{qp}$）的对应输出—— ”提出问题并获得回答“
* 非线性层（$f^{l}$）—— 模态间翻译器
* 融合上下文提示词 FCP（$z_{fcp}$）—— 帮助模态融合

**Querying Stage.**

将 QP 和 QCP 拼接到基础特征上：$[z^{l}\|z_{qcp}^{l}\|z_{qp}^{l}]$；

然后送入单模态 Transformer 中：$[\hat{z}^{l}\|\hat{z}_{qcp}^{l}\|\hat{z}_{qp}^{l}] = TransLayer^{l}([z^{l}\|z_{qcp}^{l}\|z_{qp}^{l}];\theta)$；

QP 的输出 $z_{qp}^{l}$ 通过非线性层映射到另一个模态特征空间中：$y_{qp}^{l} = f^{l}(\hat{z}_{qp}^{l})$；

**Fusion Stage.**

$[{z'}^{l+1}\|\hat{z}_{fcp}^{'l}\|\hat{y}_{qp}^{l}] = TransLayer^{l}([z^{'l}\|z_{fcp}^{'l}\|y_{qp}^{l}];\theta') \quad \quad [*]'表示[*]对应的另一个模态信息$；

然后两个单模态 Transformer 层的输出 ${z}^{l+1} 和 {z'}^{l+1}$ 共同作为多模态融合层的输出。

完整的多模态融合过程可表示为：$[z^{l+1}\|z^{'l+1}] = FusionLayer^{l}([z^{l}\|z^{'l}];\theta, \theta') \quad \quad if L_{f} <= l$

然后获得的 $z_{CLS}^{L} 和 z_{CLS}^{'L}$被送入到两个不同的线性层，然后平均两者得到的 softmax logits 作为分类结果。

> 结果：

* PMF 内存效率高
* PMF 优于所有现有的基于提示的方法
* PMF 与微调极限相比更有竞争力

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230614105407142.png" alt="image-20230614105407142" style="zoom:80%;" />

> 消融：

* 随着多模态融合层开始的越晚，训练显存的需要越少

* $L{f} < 10$ 的时候，模型表现没有太大变化

  <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230614105628000.png" alt="image-20230614105628000" style="zoom:67%;" />

总结，由于浅层网络主要是提取一些基本特征，因此在深层网络中做融合，不同模态之间的指导能力会更强。

## 20230623

### 39_TransHP: Image Classification with Hierarchical Prompting_Under review_

> 作者：Wenhao Wang, Yifan Sun, Wei Li, Yi Yang

​				ReLER, University of Technology Sydney, Baidu Research, Zhejiang University, Zhejiang University

> 贡献：

背景：层次图像分类（Hierarchical image classification，HIC）的目的是**利用语义层次结构来提高预测精度**。具体而言，HIC 提供了额外的粗糙标签（如 Rose），即精细的标签（如 China Rose 和 Rose Peace）的祖先。粗糙标签通常不需要手动注释，可以通过 WordNet 或词嵌入根据精细标签自动生成。由于 HIC 几乎没有增加任何注释成本，但却带来了实质性的收益，因此它具有现实价值，引起了大家极大的研究兴趣。

本文探索了一种新的层次提示机制，可以很好地模拟人类视觉识别的能力。当一个人的可参考范围很大时（例如，,the whole Plantae），他可能会混淆两个相近的视觉概念（例如，China Rose 和 Rose Peace）。然而，如果能把可能的类别范围缩小（例如， the rose family），人们就可以将他的注意力转移到粗糙类别内的细微变化上。本文基于 Transformer prompt 机制建模了这一过程，**将粗糙类提示符注入到 Transformer 的中间阶段**，指导后续对该限定粗糙类下的细节特征提取，即产生所谓的**层次提示**。

通过实验表明，TransHP  对多个流行的 Transformer 骨干和 5 个图像分类数据集带来了一致的分类精度的改进；对于训练数据比较少的情况下，TransHP 对基线方法的提升更明显；通过可视化观察到，TransHP 与人类视觉识别有一些相似的模式，例如，先进行粗粒度的整体识别，然后在提示后关注一些关键的局部区域进行后续识别。

下图（a），ViT试图通过识别整体的前景信息从 ImageNet 的1000个类别中识别金鱼类；图（b）， TransHP 使用中间模块将输入图片识别为 Fish 大类，从而使模型聚焦于  face and crown （识别金鱼的关键部位）。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230613161313453.png" alt="image-20230613161313453" style="zoom:80%;" />

> 方法：

Transformer with Hierarchical Prompting(**TransHP**)包含三步：

1. TransHP学习一组提示令牌来表示所有的粗糙类，并选择一个中间块作为注入提示的 “prompting block”；
2. “prompting block”  实时预测输入图像的粗糙类别；
3. “prompting block” 将预测类的提示标记（即目标提示标记）注入中间特征（拼接 prompt token 和 feature token）

**3.2 The Prompting Block of TransHP**

假设有 $M$ 个粗糙类， TransHP 使用 $M$ 个可学习的 prompt token $P_{M} = [p_{0}, p_{1}, ..., p_{M-1}]$ 表示这些粗糙类。如果输入图像属于第 $k$ 个粗糙类，则将 $p_{k}$ 注入到 prompting layer。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230617150931042.png" alt="image-20230617150931042" style="zoom:80%;" align="left"/>TransHP learns to automatically 

1) predict the coarse class
2) select the corresponding prompt for absorption through “soft weighting”（对目标提示具有高吸收率，非目标提示具有吸收低率）

 如左图（i）表示学习过程：
$$
[x_{cls}^{l}, X^{l},\hat{P}_{M}] = B_{l}([x_{cls}^{l-1}, X^{l-1},{P}_{M}]) \\
B_{l}表示第 l 层 transformer block；\\
\hat{P}_{M}将用于预测输入图片的粗糙类，但不会传入下一层
$$
 相似性得分：
$$
S = \{ p_{i}^{T}w_{i} \}, i = 1,2,...,M \\
w_{i}为可学习的第 i 个粗糙类的原型
$$
进一步使用 softmax 和 交叉熵损失来监督相似性评分：
$$
L_{coarse} = -log \frac{p_{y}^{T}w_{y}}{\sum_{i=1}^{M}exp(p_{i}^{T}w_{i})}
$$
**3.3 Overall Structure**

**Multiple transformer blocks for multi-level hierarchy**

对于某些数据集，其中包含多种层级结构，可以通过堆叠多个 prompting blocks 来实现多层级的提示，总体训练损失为：
$$
L = L_{fine} + \sum_{l} \lambda_{l} \cdot L_{coarse}^{l} \\
L_{fine} 是最终的分类损失，L_{coarse}^{l}为第 l 层 prompting-block 的损失，\lambda_{l} 为相应的权重参数
$$
**The position of the prompting block**

考虑到不同数据即得层级结构差异较大，因此本文并未给出精确的 prompting block 位置，但给出的建议是：如果粗糙类的数量较少，则 prompting block 的位置应尽量靠近底层，反之则尽量靠近顶层。

**3.4 TransHP Spontaneously Selects the Target Prompt**

吸收权重：
$$
w(x ← p_{i}) = \frac{exp(Q(x)^{T}K(p_{i})/\sqrt{d})} {\sum exp(Q(x)^{T}K([x_{cls}, X, P_{M}])/\sqrt{d})}
$$
实验：TransHP提高模型可解释性

Baseline： the attention score map of the the last block of ViT

(coarse)： the attention map at the prompting block (which handles the coarse-class information)

(fine)：  the last block (which handles the fine-class information)

* 底层关注大致整体
* 顶层关注相邻细粒度类别之间的关键布局

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230613162716140.png" alt="image-20230613162716140" style="zoom:80%;" />

### 40_Open-World Weakly-Supervised Object Localization_xx_有代码

> 作者： Jinheng Xie,Zhaochuan Luo,Yuexiang Li,Haozhe Liu,Linlin Shen,Mike Zheng Shou

​				1Shenzhen University, 2Tencent Jarvis Lab, 3KAUST, 4National University of Singapore

> 贡献：

本文首次引入了一个新的任务设定：开放世界弱监督对象定位**OWSOL**（Open-World Weakly-Supervised Object Localization）。训练的时候，所有带标签的数据均来自 known 类，无标签数据包含 known 和 novel 类。OWSOL 的数据分布模拟了一个通用场景，其中大量的未标记数据可以与标记数据联合使用，用于训练。

并且提出该问题的解决方法，通过对比共同学习有标签数据和无标签数据的表征来生成泛化类激活图**G-CAM**（Generalized Class Activation Map）。由于无标记数据缺乏类别标记，因此本文基于所有训练数据做聚类，并为表征学习设计了一个多语义质心驱动的损失。通过在 ImageNet-1K 、iNatLoc500以及本文提出的 OpenImages150数据集上验证了所提出的方法对 OWSOL 任务的有效性。

**在测试数据中**，本文将 Novel 类进一步划分为，与某 Known 类属于同一家族的 Nov-S 以及与任何 Known 类都不属于同一家族的 Nov-D

![image-20230620093228898](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230620093228898.png)

> 代码：https://github.com/ryylcc/OWSOL

> 方法：

首先使用预训练好的编码器（如MoCo）将训练数据 D 中的图像特征都提取出来并映射到对比学习的特征空间：$\{ z_{i} \}_{i=1}^{n+m}$（n个known类和m个novel类）；

为每个known类随机采样 $N_{z}$ （12）个表征样本构成 **memory bank P**：$P = \{ P(cat^{1}) \cup ... \cup P(cat^{|Y_{l}|}) \}$，对于标签 $y_{i}$，$P(y_{i})$表示类别 $y_{i}$ 的 $N_{z}$ 个 positive 表征；在训练过程中，P 会以先进先出的队列形式更新。

然后基于所有训练数据的特征  $\{ z_{i} \}_{i=1}^{n+m}$ 做聚类得到 $N_{c}$ 个语义质心，并构建 **memory bank C**：$C = \{ c^{1}, ... ,c^{N_{c}} \}$,我们使用 $C(z_{i}) = \{ c_{z_{i}}^{1},...c_{z_{i}}^{l}, ..., c_{z_{i}}^{L}  \}$ 来表示 $z_{i}$ 的L个语义质心，其中 $c_{z_{i}}^{l}$ 表示第 $l$ 个最靠近 $z_{i}$ 的质心。

![image-20230620095927581](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230620095927581.png)

**4.2. Contrastive Representation Co-learning**

在一个mini batch中。

有标签的数据上做监督对比学习：
$$
L_{scl} = -\frac{1}{|P(y_{i})|}\sum_{z_{j}^{+} \in P(y_{i})} log \frac{exp(z_{j}^{T} \cdot z_{j}^{+} / \tau)}{\sum_{z^{i} \in P} exp(z_{j}^{T} \cdot z^{i} / \tau)} \tag{1}
$$
one semantic centroid-driven contrastive learning形式如下：
$$
L_{ocl} = -log \frac{exp(z_{i}^{T} \cdot c_{z_{i}}^{1} / \phi(c_{z_{i}}^{1}))}{\sum_{c^{i} \in C} exp(z_{i}^{T} \cdot c_{z_{i}} / \phi(c_{z_{i}}))} \tag{2} \\
其中\phi(c_{z_{i}}) = \frac{\sum_{v=1}^{V} \| z^{v} - c^{i} \|_{2}}{V} 为密度估计，值越小表明聚类聚的越好
$$
公式2表示每个数据点只能被拉到接近一个语义质心。然而，由于对象的不同，这种语义质心可能对不同的对象区域有不同的重点。为了学习具有更多粒度信息的表示，我们使用**多个最近的语义质心**作为每个锚点的 positive template。因此，多重语义质心驱动的对比损失可以表述为：
$$
L_{mcl} = -log \frac{exp(z_{i}^{T} \cdot c_{z_{i}}^{*})}{exp(z_{i}^{T} \cdot c_{z_{i}}^{*}) + \sum_{c^{i} \in C/C(z_{i}) }exp(z_{i}^{T} \cdot c^{i}/\phi(c^{i}))} \tag{4} \\
c_{z_{i}}^{*} = \frac{1}{L} \sum_{l=1}^{L}c_{z_{i}}^{l}/\phi(c_{z_{i}}^{l}) 
$$
总体的对比训练损失为：$L = \alpha L_{scl} + \beta L_{mcl}$，权重分别为 1， 0.5.

**4.3. Generalized Class Activation Mapping**

**推理过程中**的聚类和 G-CAM：

训练后，将学习到的模型应用于整个测试数据，提取一组特征。然后执行聚类，将每个图像与 $Y_{all}$ （所有known和novel类）中的一个潜在类别关联起来，并生成类激活映射，以获得用于对象定位的边界框。

用 $m \in \mathbb{R}^{d_{1} \times h \times w}$ 表示encoder 提取的特征，传统闭集训练好的分类曾表示为 $w \in \mathbb{R}^{d_{1} \times k} $，那么第 k 个类别的 CAM 表示为： $p_{k}(i,j) = w_{k}^{T} m(i,j)$

虽然对这对已知类别进行空间激活是一种有效的方式，但它们并不适用于没有相应学习向量的新类别。因此，我们提出了广义类激活映射（G-CAM），其中使用聚类赋值和语义质心来替换固定大小的可学习矩阵。给定一个图像 $x$，假设$c_{x}∈\mathbb{R}^{d_1×1}$是分配给 $x$ 的语义质心，G-CAM可以得到如下：
$$
p_{x}(i,j) = c_{x}^{T} m(i,j)
$$
下图，验证了不同的质心聚焦于不同的对象语义。因此，参与表示学习的多个语义质心将有利于激活映射的完整性。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230620110725689.png" alt="image-20230620110725689" style="zoom:80%;" />

总结：

1. 多语义质心或许可以考虑用用。
2. 这篇文章的任务训练时需要用到无标签数据，即可以看到novel类的图片，只是没有相应标签。另一个比较奇怪的地方是，推理时，需要先将测试数据样本全部编码后做聚类等操作，这样不能来一张图片做一次推理。

## 20230630

### 41_Visual Semantic Role Labeling__有代码

> 作者：Saurabh Gupta UC Berkeley sgupta@eecs.berkeley.edu, Jitendra Malik UC Berkeley malik@eecs.berkeley.edu

>代码：https://github.com/s-gupta/v-coco

> 贡献：

任务：动作识别

背景：经典的动作识别方法要么研究在图像或视频剪辑上的动作分类任务，要么最多在做动作的人周围产生一个边界框。我们认为这样的输出是不够的，只有当我们能够将场景中的对象与动作的不同语义角色关联起来时，才能得到完整的理解。如图 1 展示了我们想要的输出：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230624214104022.png" alt="image-20230624214104022" style="zoom:80%;" />

我们想要的不仅是粗糙的活动标签，如“打棒球”，而是能够对细粒度的动作进行推理，如“击打”，并检测这个动作相关的各种语义角色，即： the agent（粉色框）、the instrument（蓝色框）和 the object（橙色框）。

* **提出数据集 V-COCO**。作者提取了 COCO2014 的部分数据进行了标注，首先确定了 26 个主要的动作，然后得到和这个动作相关的所有图片，并分析其中和 agent 进行这个动作交互的物体。由于一个人可能同时做多个事情，所以每个动作涉及的 obj 都有可能有多个，例如 table 1 中 cut 一词在 569 个图片中出现，并且这 569 张图片中有两种semantic role，一个是作为 obj 一个是作为 instr ，作为instr 出现了 477 次，并且作者还列出了具体的物体，scissors, fork, knife。标记为 * 的是当时还没有统计完成的。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230624212259779.png" alt="image-20230624212259779" style="zoom:65%;" align="left"/><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230624212722205.png" alt="image-20230624212722205" style="zoom:70%;" />











有了自己提出的新的 V-COCO 数据集，接下来就是用算法进行基准测试了。

对于其提出的两个task (**agent deteciton** 和 **role detection**)

* agent detection其实就是普通的目标检测，框出人的位置；
* role detection就是框出与人交互的物体的位置，并且要给出相应的标签（语义角色）。

首先在原来的COCO数据集上训练 80 个类别的物体检测模型，使用的模型算法其实是fast rcnn。（由于这时还没有faster rcnn所以建议框是提前生成好的，这里作者使用的是名为MCG的bounding box。）

 这里作者还具体分成了三个模型（第二个和第三个其实做的任务是一样的）。

**第一个模型 A **要做的事情就是将检测出的人对应到 26 中不同action的类别上，同时由于每个agent可能同时做许多不同的action所以这一步是一个 **multi-label tas**k。

**第二个模型 B** 就是要框出 obj 并且给出 role，这里模型 B 是直接进行回归，这里训练的也是 gt 和参考框的偏移量。

**第三个模型 C **其实和模型B要做的事情一样，只是这里不再直接进行回归而是采用了另一种方法。

> 实验结果：

几种模型在不同动作上的实验结果，这里模型 C 要比模型 B 好很多，比较有意思的是后边作者对结果进行了可视化，发现错误的检测结果中有几种相似的模式，比如一幅图片有两个agent，而且在做相同的action,检测器在检测的时候，框出一个人，但是却框出了另一个人交互的 obj 。

### 42_Learning to Detect Human-Object Interactions_WACV 2018_有代码

> 作者：Yu-Wei Chao, Yunfan Liu, Xieyang Liu, Huayi Zeng, Jia Deng

>代码：https://github.com/ywchao/ho-rcnn

>贡献：

* 提出了被广泛采用的数据集 **HICO-DET**

HICO 是对图像中是否存在 HOI 类进行分类的数据集，HICO-DET 在其基础上给实例添加边框注释从而能狗用于检测。HICO-DET 数据集的样例如下（左），每一个 person 对应 1 个或者多个交互动作和对象，可以用 <verb, object> 的二元组来表示。作者把这个二元组叫做 HOI category。

HICO-DET 数据集一共包含了 600 种常见的 HOI categories，比如 <riding, horse>, <cutting, apple>。数据集的各类数据统计如下（右），其中 positive 表示 human 的数量，instances 是总共有效交互 <verb, object> 的数量，由于 2 个不同的人可以共享同一个交互目标，因此 bbox 的数量小于 instance 数量的2倍。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230625155720817.png" alt="image-20230625155720817" style="zoom:80%;" align="left"/><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230625160247251.png" alt="image-20230625160247251" style="zoom:70%;" />















* 提出方法 **HO-RCNN**，首次融入 human-object 的**空间位置信息**来提升检测效果

> 方法

**HO-CRNN**：Human Object Region-based CNN，分为三部分：Human stream, Object stream, Pairwise stream。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230625160539255.png" alt="image-20230625160539255" style="zoom:80%;" />

**Generate Human-object proposal**

首先使用 Fast R-CNN 生成可能包含 human/object 的 proposals，分别保留 10 个 human 和 object 预测 proposals，然后这2类 proposals 两两配对，生成 100 对 person-object proposal pair。

**Human and Object Stream**

给定一个 human-object 的 proposal，human stream 从 human 的边界框中提取局部特征，并为每个 HOI 类生成置信度分数。首先使用边界框裁剪完整的图像，然后调整大小到固定的大小。然后将这个归一化的图像 patch 传递到一个卷积网络中，它通过卷积、最大池化和全连接层来提取特征。最后一层是大小为 K 的全连接层（其中K是感兴趣的HOI类的数量），每个输出对应于一个HOI类的置信度分数。object stream 遵循相同的设计。

**Pairwise Stream**

由于人和物之间会因为不同的交互产生不同的位置关系（比如说，你拿起一个苹果和你吃一个苹果具备不同的空间构型了）。

为了编码这种 human-object spatial configuration ，作者首先圈出能包围 human and object 的最小边框（下图黑框），这个黑框是一个 2 channels 的特征图，然后对黑框进行如下操作：

第一个 channel 对应的是人，将人的区域内的值都置为1，非 human 区域内的值置0
第二个 channel 对应的是物，将物的区域内的值都置为1，非 object 区域内的值置0

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230625161716416.png" alt="image-20230625161716416" style="zoom:80%;" />为了解决黑色框尺寸变化的问题，作者探讨了两种 resize 到固定尺寸的方法：

1. 将交互模式的两边调整为一个固定的长度，而不管其长宽比如何（这可能会改变注意窗口的长宽比）；
2. 在保持长宽比的同时，将交互模式的较长的边的大小调整为一个固定的长度，然后在较短的边的两侧填充零来实现。

（↓IP0表示无填充，IP1表示以0填充）

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230625162326686.png" alt="image-20230625162326686" style="zoom:80%;" />

> 实验：

很多使用该数据集作为 benchmark 的论文通常会给出形如下表的测试结果。其中，**Full** 表示采用全部 600 HOI categories 做训练；**Rare** 表示只用 138 种 HOI categories 做训练，这138个类别中每个类别的样本量少于10个；**Non-Rare** 表示 462 种 HOI categories 的数据集，其中每个 category 都有10个以上训练样本。

根据 HICO分 类基准，作者还考虑了两种不同的评估设置：

 **Known Object setting**：对于每个 HOI 类别（例如“riding a bike”），只评估包含目标对象类别的图像的检测。例如“bike”)。挑战在于定位 HOI（例如  human-bike pairs），以及区分互动（例如“riding”）。

**Default setting**：对于每个 HOI 类别，在完整的测试集上评估检测结果，包括包含或不包含目标对象类别的图像。这是一个更具挑战性的设置，因为我们还需要区分背景图像（例如，没有“bike”的图像）。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230625162645654.png" alt="image-20230625162645654" style="zoom:80%;" />

## 20230707

### 43_MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering_ CVPR 2022 _有代码

> 背景：

基于知识的视觉问答（KB-VQA）需要模型结合外部知识进行回答，相比于视觉问答，**基于知识的视觉问答**对于模型实现与人类类似的基于相关联知识进行跨模态的场景理解能力更具有挑战性。现有的模型通常从结构化的文本模态知识图谱获取相关知识，如 ConceptNet 与 DBpedia 等。该类知识**局限于**可以用自然语言明确表达的事实或基于一阶谓词的简单三元组，因此**难以表达高阶关系与多模态知识**，而多模态知识对问题引导下的场景理解至关重要。当前受到广泛关注的隐式知识（如GPT3）检索方式在效率、准确性、可信性上仍面临挑战，本文旨在实现隐式多模态知识的结构化、易检索、可解释的表征，具体回答三个问题：多模态知识如何实现结构化表征？该种形式的多模态知识如何积累？在视觉问答任务中如何检索和利用该多模态知识？

因此，本文提出了一种端到端的**多模态知识抽取与积累模型 MuKEA**（Multimodal Knowledge Extraction and Accumulation framework）。首先基于预训练多模态模型从非结构化的 *图像-问题-答案* 对样本中提取多模态知识三元组，提出三种损失函数来学习三元组的表示，分别是问题引导的视觉区域（**头实体特征**），事实答案（尾实体特征），头实体与尾实体间的隐式关系（**关系特征**）。通过域外和域内数据的训练，模型积累了广泛的多模态知识，从而能基于知识检索进行答案预测。

> 方法：

![image-20230703142357785](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230703142357785.png)

该方法主要分为三个部分：1、多模态知识三元组抽取，2、知识三元组表示学习，3、基于预训练与微调的多模态知识积累

**3.1. Multimodal Knowledge Triplet Extraction**

在VQA场景中，我们将复杂且无法表达的事实定义为三元组形式的多模态知识，即 $(h，r，t)$，其中 $h$ 为问题聚焦的图像中的视觉内容，$t$ 是给定问题图像对的答案的表示，$r$ 描述了包含多模态信息的 $h$ 和 $t$ 之间的隐式关系。

首先我们通过多模态预训练模型 **LXMERT** 建模模态间与模态内关联获得图像与问题的表示，然后基于硬注意力机制计算得到与问题最相关的视觉区域作为头实体。由于LXMERT通过注意力机制很好地捕获两个模态间的关联，所以可以用 $[CLS]$ 符号来表示隐式关系。最后将尾实体定义为问答样本中的答案，它代表了与问题对应的视觉对象有关的事实。

**3.2. Knowledge Triplet Representation Learning**

由于三元组内的每个部分都含有不同的模态信息和语义信息，本文提出三种损失函数来学习三元组的表示，以消除异构鸿沟和语义鸿沟，这三种损失函数互相补充，共同约束三元组的表示。

第一个损失：**Triplet TransE Loss**，通过对比正、负样本对，保持嵌入结构；

第二个损失：**Triplet Consistency Loss**，上述 TransE 损失的问题是，当训练过程中正负对之间的距离小于间隔 $\gamma$ 时，模型将停止学习。为了进一步使实体嵌入满足严格拓扑关系，采用均方误差损失来约束 $h + r 和 t^{+}$ 之间的距离；

第三个损失：**Semantic Consistency Loss**，通过随机初始化一个 look-up table 来学习尾实体的表示，为了将尾实体与对应的语义标签相关联，将三元组映射至尾实体词表进行分类：

**3.3. Knowledge Accumulation and Prediction**

**训练**方法上采用预训练+微调的方法进行知识的逐步积累，首先在 VQA2.0 Other 类型的问题上积累基础的视觉相关知识，然后在下游 KB-VQA 相关数据集上积累复杂的领域相关知识。在**测试**阶段采用知识图谱补全的方法进行关系预测找出最相关的尾实体，即找到与 $h + r$ 最相近的 $t$。

### 44_Detecting the open-world objects with the help of the “Brain”_投稿 2023 _有代码链接

> 作者：Shuailei Ma, Yuefeng Wang, Ying Wei, Peihao Chen, Zhixiang Ye, Jiaqi Fan, Enming Zhang, Thomas H. Li

**【注】**：本文和 CAT 作者团队一致。文中的思路（**级联**和权重减弱）思想都和 CAT 很像（CAT 也用了级联，以及伪标签驱动权重变化），CAT 对于未知类的为标签方式为 模型驱动和输入驱动（selected search）的权重随着模型训练逐渐变化（越到训练后期，模型驱动权重占比越大）。另外，阿里团队的 HOI 检测文章（Mining the Benefits of Two-stage and One-stage HOI Detection，NeurIPS 2021 ）中也用到了和本文一样的级联思想。

> 代码：Gongjie Zhang, Zhipeng Luo, Zichen Tian, Jingyi Zhang, Xiaoqin Zhang, Shijian Lu

> 贡献：

人类识别环境中未知物体的自然本能主要取决于大脑中的知识库，然而一个模型很难通过从几个小数据集的注释中学习来做到这一点。然而大型的预训练 VL 模型（如 GLIP ）对开放世界拥有丰富的知识（但用于检测还受限于文本提示，即根据文本去检测对象）。

本文则提出**利用 VL 作为开放世界探测器的“大脑”**，通过大型 VL 生成未知类的标签。但由于直接使用其生成伪标签势必会影响模型对已知类的学习，因此本文提出降低伪标签对应的损失所占权重（**down-weight loss function**）来缓解这个问题；另外“未知类”中与已知类具有高度相似特征的对象的存在会极大地影响开放世界对象的识别过程。这个问题不仅影响识别（分类），而且影响定位能力，因此本文提出级联结构来解耦分类和定位过程。

结果：对于未知类的召回率很高。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230704204103463.png" alt="image-20230704204103463" style="zoom:80%;" />

> 方法：

对于给定一张图片，会同时被送入 Detector 和 Assistant 中，Detector 输出定位框回归 $b$ 、边框分数 $bs$ 以及分类得分 $cls$；Assistant 则挖掘开放世界的信息（找出可能包含对象的区域打伪标签）。然后用 down-weight 损失函数的方式，使用已知类的注释信息以及伪标签共同训练开放世界的检测器。

![image-20230704204401055](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230704204401055.png)

**4.2. Cascade Open-World Object Detector**

解码器分为定位解码器和分类解码器两部分，其中定位解码器的输出除了用于预测 $b 和 bs$ ，还作为 class queries 输入到分类解码器中。

**4.3. Assistance from the large pre-trained model**

本文使用 GLIP 作为 Pre-trained VL， prompt 为所有 LVIS 数据集中的类别。所产生的标签先进行 NMS 处理，然后与已知类 GT 注释对齐并去除掉对应于已知类的哪些标签从而获得 unknown labels，另外，所得到的 unknown label 的  identifification confifidence 将用于后续 Down-Weight Training。

**4.4. Matching and Evolving**

匹配方式采用的是 Hungarian 算法，讲预测和 GT 匹配时，匹配代价同时考虑 边框定位损失、分类损失以及 box score 损失。匹配之后，box score 分支将用于选择伪标签：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230704210739907.png" alt="image-20230704210739907" style="zoom:80%;" />

**4.5. Down-Weight Training Strategy**

生成的伪标签不可避免地会影响模型对已知类的学习。因为生成的标签的质量不能得到保证，它们会增加模型学习的难度。因此，本文提出了 down-weight 训练策略，利用生成的识别置信度 $\hat{S}$​ 生成**软标签** ，对未知的训练损失进行减压，并以端到端方式训练如图2 (b)所示。训练损失：$𝐿 = 𝐿_𝑟 + 𝐿_{𝑏𝑠} + 𝐿_{𝑐𝑙𝑠} + 𝐿^{𝑧}_{r} + 𝐿_{𝑏𝑠}^{𝑧} + 𝐿_{𝑐𝑙}𝑠^{𝑧} + 𝐿_{𝑐𝑙s}^{p}$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230704213558137.png" alt="image-20230704213558137" style="zoom:80%;" />

【问题】：文中没有给出 $l_{z}$ 的解释，猜测应该是来自大模型的那些橙色框，但是这些框是作为什么标签用于匹配的呢？（guan与软标签的形式存疑，等代码出来再看）

> 消融

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230704213836209.png" alt="image-20230704213836209" style="zoom:80%;" />

### 45_Annealing-based Label-Transfer Learning for Open World Object Detection_CVPR 2023_有代码

> 作者：Yuqing Ma， Hainan Li， Zhange Zhang, Jinyang Guo，Shanghang Zhang, Ruihao Gong, Xianglong Liu

> 代码：https://github.com/DIG-Beihang/ALLOW

> 贡献：

背景：之前的 OWOD 工作是手动设计未知类发现策略，从背景中选择未知类的 proposals，存在没有适当先验的不确定性缺点。

在本文中，作者认为目标检测的学习可以看作是一个**对象级的特征纠缠过程**，其中未知类的特征通过卷积运算传播到已知类的 proposals 中，并可以在没有人工选择的情况下有利于未知对象的识别。

因此，作者提出了一个简单而有效的**基于退火的标签转移框架（Annealing-based Label-Transfer framework）**，它充分探索已知类的 proposals，以缓解不确定性。具体而言，首先通过**标签迁移的学习范式（Label-Transfer Learning paradigm）**来解耦已知类和未知类的特征；然后通过**锯齿型退火方法（Sawtooth Annealing Scheduling）**来构建已知类和未知类的决策边界，从而共同提高模型对已知类和未知类的检测性能。

另外，作者指出之前的 OWOD 工作忽略了已知类和未知类性能的 trade-off，因此在本文中介绍了一种新的衡量 OWOD 模型性能指标：(**EI**: Equilibrium Index，平衡指数）

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707090502484.png" alt="image-20230707090502484" style="zoom:67%;" />

结果：

![image-20230707084934026](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707084934026.png)

【注】：据作者介绍，这是第一个没有使用手动设计未知类选择的 OWOD 工作。

> 方法：

**3.2. Motivation: Object-Level Feature Entanglement in Object Detection**

动机：对象级特征纠缠。由于同一图像中包含多个物体（可能会包括未知的物体），通过 local-connected 的卷积运算，它们的 object 级特征将同时被感知并纠缠在一起（如下图）。因此，与已知的 GT 相匹配的 proposal 既包含已知的类特定特征，也包含有利于未知识别的潜在未知信息。简单地将这种纠缠的 proposal 分类为已知类，就会导致对未知实例的错误分类。

由于大多数图像同时包含已知和未知的实例，我们可以提取与已知的鉴别特征不冲突且具有**有意义语义模式的未知特征**。这样，就可以在保持已知检测性能的同时，提高未知检测。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707090943729.png" alt="image-20230707090943729" style="zoom:67%;" />

**3.3. Annealing-based Label-Transfer Framework**

标签迁移学习可以指导模型区分已知和潜在的未知信息，以推进未知学习，而锯齿退火调度策略可以通过修改解纠缠度来调整学习过程，以达到未知学习和已知学习之间的平衡。

根据锯齿退火调度，整个标签转移学习可以看作是两个连续的阶段，即**形成阶段**和**扩展阶段**。

在形成阶段，该模型倾向于形成纠缠的已知 proposal。然后，随着解纠缠度的增加来学习未知信息，已知的检测性能会受到不利影响。

因此，在扩展阶段，调整解纠缠度变为锯齿形状，在考虑未知对象的情况下重建已知的决策边界。

最后，形成了未知类和已知类的决策边界，使 OWOD 模型具有未知的识别能力，并保持了已知的识别精度。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707093800000.png" alt="image-20230707093800000" style="zoom:80%;" />

Label-Transfer Learning：一般的目标检测算法，标签是one-hot的，这种标签增强了已知类别的分类，但阻止了未知物体的检测。因此，本文将标签变为软标签，加入一个Label-Transfer term

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707095252611.png" alt="image-20230707095252611" style="zoom:80%;" />

所以分类损失的形式变成

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707095404247.png" alt="image-20230707095404247" style="zoom:80%;" />

可见 $\lambda$ 的确定直接影响性能。

因此，提出 Sawtooth Annealing Scheduling 来设计一个适应学习过程的 $\lambda$ 。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707100238013.png" alt="image-20230707100238013" style="zoom:80%;" />

### 46_Supporting Vision-Language Model Inference with Causality-pruning Knowledge Prompt_Conference ’22, October 10–14, 2022, Lisbon, Portugal_无代码

> 作者：	Jiangmeng Li, Wenyi Mo, Wenwen Qiang, Bing Su, Changwen Zheng

> 贡献：

VL 模型经过在大量图像文本对上训练后能够用于解决开放域的视觉任务，近期的工作探索了固定的或是可学习的 prompts 来将 VL 模型用于 zero-shot 任务，但是**什么样的** prompts **如何**起到提升模型推理能力的作用还没有被探索。

在本文中，作者为 prompts 包含语义信息的重要性提供了明确的解释。构建富含语义信息的 prompts 需要领域专业知识而且非常耗时，因此，本文提出因果关系修剪提示方法（CapKP：Causality-pruning Knowledge Prompt），使预训练 VL 模型适应下游视觉识别任务。CapKP 通过将文本标签作为 query 来检索本体知识图，以探索与任务相关的语义信息。为了进一步细化派生的语义信息，CapKP 通过遵循格兰杰因果关系（Granger causality）的第一原则引入了因果关系剪枝技术。

下图的 abc 分别展示了固定的、可学习的和本文提出的因果关系修剪提示的设计方式：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230705175500900.png" alt="image-20230705175500900" style="zoom:67%;" />

> 方法：

总体：

1、根据标签的到1跳标签相关的知识子图

2、根据格兰杰因果关系裁剪掉因果不相关的边和点

![image-20230705183700543](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230705183700543.png)

修剪：

因果关系修剪图表示的基本原理的一个例子。根据格兰杰因果关系的第一原理，我们通过修剪与下游任务因果解耦的边来改进得到的知识子图 G_i。通过迭代去除与关系类型 r_m 相关的边，然后检查结果的变化来确定格兰杰因果关系，这是根据一个特定的图规则计算的。只保留与因果相关的边缘，其他的则被修剪。
注：图编码器 f_G（·）在整个过程中都是固定的。

![image-20230705183736100](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230705183736100.png)

### 47_Towards Efficient Use of Multi-Scale Features in Transformer-Based Object Detectors _ 投稿中_有代码链接

> 作者：Gongjie Zhang, Zhipeng Luo, Zichen Tian, Jingyi Zhang, Xiaoqin Zhang, Shijian Lu

> 代码：https://github.com/ZhangGongjie/IMFA

> 贡献：

多尺度特征已经被证明对目标检测非常有效，但往往会带来额外巨大的计算成本，特别是对于基于 Transformer 的检测器。

本文提出一种用于 Transformer-based 检测器高效利用多尺度特征的通用方法—— 迭代多尺度特征聚合 **IMFA**（Iterative Multi-scale Feature Aggregation），其核心思想是利用来自几个关键位置的稀疏多尺度特征。

首先，IMFA重新排列 Transformer 编解码器通道，使编码后的特征可以根据解码预测的指导进行迭代更新；然后，IMFA 稀疏采样尺度自适应特征，在先验预测的指导下，仅从几个关键点位置进行细化检测。这样，采样到的多尺度特征稀疏且非常有利于目标检测。

实验表明，所提出的 IMFA ，在少量额外的计算开销情况下，能够显著提高多个基于 Transformer 的目标检测器的性能。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707155902175.png" alt="image-20230707155902175" style="zoom:80%;" />

> 方法：

如下图，展示了普通 Transformer 检测器和 IMFA 的区别。它在先验预测的指导下，从最有可能出现物体的区域采样多尺度特征，并迭代地更新编码器输出的特征。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707160342620.png" alt="image-20230707160342620" style="zoom:80%;" />



<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707160411435.png" alt="image-20230707160411435" style="zoom:80%;" />

### 48_Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection_CVPR 2023_有代码

> 作者：Xinjiang Wang,Xingyi Yang,Shilong Zhang,Yijiang Li,Litong Feng,Shijie Fang,Chengqi Lyu,Kai Chen,Wayne Zhang

> 代码：https://github.com/Adamdad/ConsistentTeacher

> 贡献：

背景：半监督目标检测的目标是利用大量无标签数据来辅助训练好一个目标检测器。通常的做法是先在有标记的数据上训练教师模型，然后在未标记的数据上生成伪标签和边框作为学生模型的 GT。以期望，无论网络随机性或数据增强如何，学生检测器都可以做出一致的预测。此外，为了提高伪标签质量，将教师模型将根据学生模型参数的进行移动平均的跟新。

在本研究中，作者指出半监督检测器的性能仍然在很大程度上受到**伪标签的不一致性**的阻碍。**不一致性**意味着伪边框可能会高度不准确，并且在不同的训练阶段差异很大。

问题1、与半监督分类不同，SSOD 有一个额外为每个 RoI 分配一组伪边框作为密集监督的步骤。常见的两阶段和单阶段 SSOD 网络采用**静态标准来进行 anchor 分配**，例如 IoU score 或 centerness。结果表明，静态分配对教师预测的边界框中的噪声很敏感，因为伪边界框中的一个小扰动可能会极大地影响分配结果；

问题2、广泛使用的**固定阈值方案**也会导致伪标签中的阈值不一致。传统的 SSOD 方法利用置信分数的静态阈值训练学生模型。但是，阈值作为一个超参数，不仅需要仔细调整，还应该根据模型在不同时候的能力进行动态调整。在 Mean-Teacher 范式中，在固定阈值方案下，伪 bbox 的数量可能会有时太少而有时过多，这将对学生造成低效和有偏见的监督。

因此，本文提出 Consistent-Teacher 来解决上述不一致性问题。**最终目的**是期望老师模型能够尽量产生足够精确的伪标签（主要在**边框准确度**这一块）。

首先，将静态分配 anchor 的方式改为 cost-aware adaptive sample assignment (**ASA**)就能很好地缓解密集伪目标带来的不一致性问题；

其次，通过三维特征对齐模块（**FAM-3D**）使分类特征感知并采用用于回归的最合适的特征来进行分类，从而达到**校准**老师模型的分类置信度对边框质量的代表性的目的；

然后，对于伪边框中的阈值不一致性，作者在训练中用高斯混合模型（**GMM**）为每个类别生成自适应阈值。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707195715527.png" alt="image-20230707195715527" style="zoom:80%;" />

### 49_MixTeacher: Mining Promising Labels with Mixed Scale Teacher for Semi-Supervised Object Detection_CVPR 2023_有代码链接

> 作者：Liang Liu, Boshen Zhang, Jiangning Zhang, Wuhao Zhang, Zhenye Gan, Guanzhong Tian, Wenbing Zhu, Yabiao Wang, Chengjie Wang

> 代码：https://github.com/lliuz/MixTeacher

> 贡献：

背景：现有的半监督目标检测方法依赖于严格的条件来从网络预测中获取高质量的伪标签，但作者观察到，具有极端尺度的目标往往置信度较低，导致对这些目标缺乏积极的监督。

在本文中，作者提出了一个新的框架，通过引入混合尺度教师来解决尺度变化问题，以改进伪标签生成和尺度不变学习。

> 方法：

混合尺度特征金字塔是通过从输入图像的常规视图和下采样视图提取两个特征金字塔来构建。然后，使用**特征融合模块**将这些金字塔组合起来，以自适应地融合适合不同尺度对象的特征。伪标签则使用教师网络中的混合尺度特征金字塔生成。

为了解决由高分数阈值引起的假阴性问题，作者还提出了一种称为有希望标签挖掘（**PLM**）的策略。PLM 通过衡量从常规尺度到融合尺度的置信度得分的提升来挖掘预测中的伪标签。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707212547976.png" alt="image-20230707212547976" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707211514130.png" alt="image-20230707211514130" style="zoom:80%;" />



### 50_Detection Hub: Unifying Object Detection Datasets via Query Adaptation on Language Embedding_CVPR 2023_无代码

> 作者：Lingchen Meng, Xiyang Dai, Yinpeng Chen, Pengchuan Zhang, Dongdong Chen, Mengchen Liu, Jianfeng Wang, Zuxuan Wu, Lu Yuan, Yu-Gang Jiang

> 贡献：

背景：结合多个数据集可以提高许多计算机视觉任务的性能。但由于检测数据集之间的**分类差异**和域差距，在组合多个数据集时目标检测中并没有出现类似的趋势。本文中，作者提出检测中心（***Detection Hub***）的概念来解决这个问题，通过将不同数据集的语义类别名称映射到统一的嵌入空间，并动态调整目标查询，实现了多个数据集的统一训练。

* 类别语义对齐：通过用单词嵌入替换 one-hot 的类别表示，并利用语言嵌入的语义一致性，将跨数据集的类别在语义上对齐到一个统一的空间中。
* 数据集感知：数据集感知的设计是通过学习一个数据集嵌入来实现的，该数据集嵌入用于适应对象查询以及检测头中的卷积内核。

结果显示在多个数据集上进行联合训练可以显著提高目标检测的性能。

> 方法：

* **Category-aligned Embedding**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707215623399.png" alt="image-20230707215623399" style="zoom:80%;" />

* **Dataset-aware Query**

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707215643403.png" alt="image-20230707215643403" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707214913502.png" alt="image-20230707214913502" style="zoom:80%;" />

### 51_Understanding and Constructing Latent Modality Structures in Multi-Modal Representation Learning_ CVPR 2023_无代码

> 作者：Qian Jiang, Changyou Chen, Han Zhao, Liqun Chen, Qing Ping, Son Dinh Tran, Yi Xu, Belinda Zeng, Trishul Chilimbi

> 贡献：

背景：多模态表示学习旨在从图像和文本中学习通用的表示，以在多模态下游应用中发挥作用。然而，如何有效地融合两种模态仍然是一个重要的问题。

本文通过信息论的论证，首先证明了在下游预测任务中，精确的模态对齐并不是最优的选择，而更好的性能取决于**有意义的潜在模态结构**。为此，作者提出了三种构建潜在模态结构的方法，并在多个任务上进行了广泛的实验验证，取得了一致的改进效果，证明了所提方法的有效性和普适性。

* 模态内正则化的深度特征分离损失
* 模态间正则化的 *Brownian-bridge*
* 模态内和模态间正则化的几何一致性损失

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707221800147.png" alt="image-20230707221800147" style="zoom:80%;" />

### 52_Revisiting Prototypical Network for Cross Domain Few-Shot Learning_CVPR 2023_有代码

> 作者：Fei Zhou, Peng Wang, Lei Zhang, Wei Wei, Yanning Zhang

> 代码：https://github.com/NWPUZhoufei/LDP-Net

> 贡献：

背景: 本文关注的问题是原型网络在跨领域少样本学习中的有限泛化能力。原型网络是一种流行的少样本学习方法，通过建立一个可推广到新的少样本任务的特征度量来解决问题。然而，当将原型网络应用于新领域的少样本任务时，其性能大幅下降，限制了其在实际应用中的可行性。
以往的研究主要集中在原型网络的设计和改进上，但很少考虑跨领域泛化的问题。现有的原型网络倾向于利用一些偏见的快捷特征，这些特征在预定义领域的元训练任务中足以区分很少的类别，但在不同领域中很难泛化。
受到神经网络中的简单性偏见问题的启发，**作者认为原型网络的有限跨领域泛化能力是由于其倾向于利用一些偏见的快捷特征所导致的**。为了解决这个问题，本文提出了**局部-全局蒸馏原型网络**（**LDP-net**），通过在全局图像和局部裁剪之间进行知识蒸馏，从而利用跨领域可传递的语义特征建立特征度量。此外，还通过对同一类别的不同图像之间进行**局部-全局语义一致性的强化**，减少了特征的类内语义变化。LDP-net在跨领域少样本学习基准上取得了 SOTA 结果。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230707223109867.png" alt="image-20230707223109867" style="zoom:80%;" />

## 20230714

### 53_Modeling Inter-Class and Intra-Class Constraints in Novel Class Discovery_CVPR 2023_有代码

> 作者：Wenbin Li, Zhichen Fan, Jing Huo, Yang Gao

​				单位：南京大学，新软件技术国家重点实验室

> 代码：https://github.com/FanZhichen/NCD-IIC

> 贡献：

背景：新类发现任务（NCD）的目的是学习一个模型，该模型将 common knowledge 从一个类不相交的标记数据集转移到另一个未标记的数据集，并在其中发现新的类。作者发现，现有的方法并没有充分利用 NCD 设置的本质（即忽视了标记和未标记类别之间的不相交特性）。

为此，本文提出了基于**对称 KL 散度**（sKLD）对 NCD 中的**类间**和**类内**约束进行建模。

* 类间sKLD约束：有效地利用标记类和非标记类之间的不相交关系，增强了嵌入空间中不同类的可分性；
* 类内sKLD约束：明确地约束样本与其相应增强样本之间的内部关系，并同时确保训练过程的稳定性。

结果：实验结果表明，所提出的方法大幅优于现有的最先进方法。该方法有效区分了未标记数据和标记数据，有助于发现新类别。类间 sKLD 约束在提高未标记类别的性能方面起到了关键作用。该方法防止将标记图像错误分类为未标记类别，反之亦然。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230708223409247.png" alt="image-20230708223409247" style="zoom:80%;" />

> 方法：

与之前的工作一致，假设未标记类 $C_{u}$ 的数量是预先已知的。

思路比较简单，对于分类头的输出，通过对称 KL 散度尽可能增大有标记数据和无标记数据的分类差异，减小有标记数据（无标记数据）的分类差异。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230708231548494.png" alt="image-20230708231548494" style="zoom:80%;" align="left"/> <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230708231619879.png" alt="image-20230708231619879" style="zoom:80%;" align="left"/><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230708232125312.png" alt="image-20230708232125312" style="zoom:80%;" /><img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230708232235545.png" alt="image-20230708232235545" style="zoom:80%;" />

除上述两个损失外，还有标准交叉熵（CE）损失：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230708231920678.png" alt="image-20230708231920678" style="zoom:80%;" />

总体训练损失函数为：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230708231941021.png" alt="image-20230708231941021" style="zoom:80%;" />

> 总结：

本文方法相当于是对分类头输出的 logits 分布进行约束，比较新颖。

但是训练的时候是明确知道哪些数据是有标签的，哪些数据是无标签的，这算是个归纳偏置吧。

### 54_CapDet: Unifying Dense Captioning and Open-World Detection Pretraining_CVPR 2023_无代码

> 作者：Yanxin Long, Youpeng Wen, Jianhua Han, Hang Xu, Pengzhen Ren, Wei Zhang, Shen Zhao, Xiaodan Liang

​				单位：中山大学深圳校区

> 贡献：

背景: 过去的开放世界检测方法（指的应该是 open vocabulary）需要在推理阶段使用预定义的类别空间，并且只能预测属于该空间的对象。对于一般的 OVD 面临两个挑战：1、很难确定一个大而全的类别列表；2、对罕见类响应低容易导致识别错误。

然而，在真正的开放世界检测中，我们希望能够预测未知类别的对象。由于密集描述任务可以生成与图像中的对象相关的自然语言描述。

因此，本文旨在将密集描述和开放世界检测任务**统一到一个框架中**，以实现更好的性能和更广泛的应用。

作者将 Open-vocabulary object detection 方法和 dense caption 方法进行结合提出一个新的 OVD 框架 **CapDet**，实现对不在类别列表中的物体能用文本描述的方式表示出来。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230709101315006.png" alt="image-20230709101315006" style="zoom:80%;" align="left"/>  





左图对比了 OWD、OVD 和 CapDet 的区别：

 OWD只能把新类检测为 unknown，无法明确描述其是啥；

 OVD检测能力受限于预定义的类别列表；

 CapDet 检测和识别常见的对象类别，并为未知的和罕见的类别生成密集的描述。









> 方法：

首先**将检测数据和密集字幕数据统一为一种数据格式**：$(x, \{b_i\}^N_{i=1}, y^N_{i=1})$, x 为输入的 RGB图片，b 为边框，y 为对应 b 的描述。

对于检测数据，y 由类别名称和其在概念字典中的描述组成，如 $y_i = “person, a\quad human\quad being.”$;

对于密集描述数据，y 为 region-grounded caption，如 $y_i = “an\quad outlet\quad on\quad the\quad wall.”$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230709134750658.png" alt="image-20230709134750658" style="zoom:80%;" />

模型架构包括双重视觉-语言编码器，其中图像编码器用于目标检测，文本编码器用于从概念集生成嵌入。训练目标包括用于匹配区域和概念的对齐损失，以及用于边界框回归和中心性的定位损失。ATSS 检测器用作图像编码器。

有两个语言头，一个是 Vision-Language 的文本头，一个是 Dense Caption 任务的文本头。输出结果和同一个 Image Encoder 输出的图像特征计算对齐损失。回归损失和中心损失用于训练一个类别无关的目标检测器。Language Modeling loss 用于训练 Dense Caption 任务。这样的结合实现了一个理想的目标，不会因为物体没在类别列表中就导致检测不出来或检测出错，而是输出一系列描述。

> 实验：

文中分别对 zero-shot 目标检测和 dense caption 任务与 baseline 进行了对比，都取得了不错的效果。由于之前没有同样任务设定的baseline，所以没有相应的实验对比，只是在附件中给出了可视化结果：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230709140535428.png" alt="image-20230709140535428" style="zoom:80%;" />

> 总结：

是个比较有意思的工作，但是没有开源；

没有评价体系：作者没有为这个任务提出相应的评价标准（应该后面会做吧）。

文中作者提到了两个缺陷：1、密集描述生成的训练花费大量时间；2、现有的密集描述数据的收集成本很高。

### 55_Learning Object Context for Dense Captioning_AAAI 2019_无代码

> 作者：Xiangyang Li, Shuqiang Jiang, Jungong Han

​				单位：中国科学院智能信息处理重点实验室，中国科学院计算技术研究所

> 贡献：

背景：密集描述是一项具有挑战性的任务，它不仅可以检测图像中的视觉元素，还可以生成自然语言句子来描述它们。以往的方法中没有充分利用图像中的对象信息。然而，对象提供了有价值的线索，可以帮助预测描述区域的位置，因为描述区域通常与对象区域高度重叠。同时，对象还提供了描述对象区域的重要信息，描述不仅描绘了其属性，还涉及其与图像中其他对象的交互。

鉴于目标与描述区域之间的相关性，本文提出通过**对象上下文编码 LSTM 网络**自动学习每个描述区域的补充对象上下文信息，将对象的知识转移到描述区域。通过优化位置预测和描述生成，使对象上下文编码 LSTM 能够捕捉和聚合有用的对象上下文。

实验结果表明，与现有方法相比，本文提出的方法具有更好的性能。

> 方法：

首先检测一组对象，然后将所有对象放入一个编码长短期记忆网络的上下文中，形成信息上下文。LSTM 单元格逐步将每个对象作为输入，并根据从其以前的状态和当前输入中捕获的信息，决定是否保留来自当前输入中的信息还是丢弃它。最后，将学习到的上下文作为引导信息，帮助生成描述和预测边界框偏移量。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710205940316.png" alt="image-20230710205940316" style="zoom:80%;" />

> 总结：

这篇应该就是马老师说的 proposal 用 LSTM 编码来学习 proposal 之间的关系的文章吧；本文的思路在于，用 LSTM 编码出上下文关系，为 region caption 提供更多的上下文信息。

能否借鉴还有待思考。（或许就是我之前的想法，用 Transformer 对 ROIs 进行特征增强，也可以理解为 proposals 之间大家相互看看彼此，交换交换信息）

### 56_Region-Object Relation-Aware Dense Captioning via Transformer_TNNLS 2022_无代码

TNNLS：IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS

> 作者：Zhuang Shao, Jungong Han, Demetris Marnerides, and Kurt Debattista

> 贡献：

背景：密集描述生成是生成复杂视觉场景详细描述的任务，近年来取得了一些成功。然而，现有方法存在两个主要限制：**1）**大多数方法采用编码器-解码器框架，使用长短期记忆（LSTM）对上下文信息进行顺序编码。然而，LSTM的遗忘门机制在**处理长序列时容易出现问题**；**2）**绝大多数方法将所有感兴趣区域（RoIs）视为同等重要，**未能关注更具信息量的区域**，导致生成的描述无法突出图像的重要内容，不够自然。

为了解决现有方法的限制，本文提出了一种基于 Transformer 的密集图像描述生成架构 **TDC**。通过引入**区域-对象关系得分单元（ROCSU）**，测量每个 RoI 的重要性，并根据检**测到的对象与区域之间的关系**以及**区域内检测到的对象的置信度**来分配权重。

实验结果表明，**TDC+ROSCU** 在 VG V1.0 数据集上相对于COCG（文献 55）方法的mAP提高了17%；在 VG V1.2 数据集上，相对于 COCG 的mAP提高了14.5%，达到了11.90，超过了现有技术的最新方法

> 方法：

1、针对 LSTM 编码上下文信息存在长序列问题，作者提出使用 Transformer 替代 LSTM

2、作者认为包含更多具有高检测置信度得分的物体的区域包含更多信息，从而更重要，因此提出 **ROCSU** （Region-Object Correlation Score Unit ），同时考虑对象检测分数和区域与对象的 IoU 为不同的 RoI 分配不同的权重。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230711105304172.png" alt="image-20230711105304172" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230711104954997.png" alt="image-20230711104954997" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230711100843777.png" alt="image-20230711100843777" style="zoom:80%;" />

> 总结：

本文是在上一篇文章基础上做的改进，其中有同名的人，应该是一个团队。

### 57_Visual-Language Prompt Tuning with Knowledge-guided Context Optimization_CVPR 2023_有代码

> 作者：Hantao Yao, Rui Zhang, Changsheng Xu

​				单位：中国科学院自动化研究所，多模态人工智能系统国家重点实验室

> 代码：https://github.com/htyao89/KgCoOp

> 贡献：

背景：提示调优是一种使用与任务相关的文本标记来使预先训练好的视觉语言模型（VLM）适应下游任务的有效方法。具有代表性的基于 CoOp 的工作将可学习的文本标记与 class tocken 结合起来，以获得特定的文本知识。然而，因为缺乏具有泛化能力的基本一般文本知识，特定的文本知识对未知类难以泛化。

为此，本文引入了一种新的知识引导上下文优化（**KgCoOp**），以增强对不可见类的可学习提示符的泛化能力。KgCoOp的关键观点是，通过减少可学习提示和手工提示之间的差异，可以减轻对基本知识的遗忘。KgCoOp 将**学习到的提示生成的文本嵌入**与**手工制作的提示**之间的**差异最小化**。在对比损失上添加 KgCoOp 可以对已知类和未知类产生有区别的提示。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230709191604702.png" alt="image-20230709191604702" style="zoom:80%;" align="left"/> 

左表对比了手工设计的 prompts 的 CLIP 和一些可学习 prompts 方法的对比：

 1、手工设计的 prompts （如 “a photo of a [Class]”）具备更好的泛化知识，表现为 New 的性能更好；

 2、本文方法 KgCoOp 相较于其他可学习的提示方式，泛化性更好，微调训练时间更少。（Base 虽然降低了点，但还是比 CLIP 好的）

通过在多个分类数据集上进行 few-shot、域泛化实验，结果：1、更高的总体性能；2、对新类泛化能力更强；3、训练耗时更少。

> 方法：

由于作者发现不可见类上的性能下降与可学习提示和固定提示之间的距离波动是一致的（如下图），所以得出结论，提高可学习提示和固定提示之间的相似性可以减轻一般（通用）文本知识的遗忘，从而提高未知类领域的通用性，这是本片工作的**核心动机**。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230709204000074.png" alt="image-20230709204000074" style="zoom:80%;" />

所以作者提出通过对可学习的文本提示和固定文本提示之间设置一个**欧氏距离**作为约束：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230709204734065.png" alt="image-20230709204734065" style="zoom:80%;" align="left"/> 

所提出的约束：

 <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230709204807137.png" alt="image-20230709204807137" style="zoom:80%;" /> 



原本的标准对比损失：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230709204830188.png" alt="image-20230709204830188" style="zoom:80%;" /> 



总训练损失：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230709204847225.png" alt="image-20230709204847225" style="zoom:80%;" />



> 总结：

1、改进比较简单，只是基于 CoOp 加了一项约束，使可学习的提示不要学飞了，还是要尽量靠近 CLIP 原本的手工设计提示（包含更一般的文本知识）；

2、写作思路好。先通过实验发现，prompt 微调方法相较于原始 CLIP 在下游任务中的表现，虽然对已知类的性能提高不少，但对未知类的泛化能力不如 CLIP；并且多个数据集呈现出来的趋势表明，学习得到的 prompt 与 CLIP prompt 差异越大，对新类的泛化能力越差，因此自然而然地想到通过减小这种差异来增加模型对未知类的泛化能力。

### 58_Texture-guided Saliency Distilling for Unsupervised Salient Object Detection_CVPR 2023_有代码

> 作者：Huajun Zhou, Bo Qiao, Lingxiao Yang, Jianhuang Lai, Xiaohua Xie

> 代码：https://github.com/moothes/A2S-v2

> 贡献：

背景：无监督显著对象检测（USOD）是基于无标签数据进行训练，达到同时正确定位和精确分割显著对象的目的。

基于深度学习的无监督显著对象检测（USOD）主要依赖于传统手工方法或预训练网络生成的噪声显著性伪标签。为了解决噪声标签问题，一类方法只关注具有可靠标签的简单样本，而忽略了硬样本中有价值的知识。

在本文中，作者提出了一种新的 USOD 方法，同时从简单和硬样本中挖掘丰富和准确的显著性知识。

首先，作者提出了一种置信度感知显著性蒸馏（**CSD**，Confidence-aware Saliency Distilling）策略，该策略以样本的置信度为条件对样本进行评分，指导模型将显著性知识从简单的样本逐步提取到硬样本。其次，作者还提出了一种边界感知纹理匹配（**BTM**， Boundary-aware Texture Matching）策略，通过匹配预测边界周围的纹理来细化噪声标签的边界。因此该方法可以产生高质量的伪标签来训练广义显著性检测器。

> 方法：

网络Φ产生的激活图可以感知输入图像中的一些鉴别区域，但由于噪声的广泛存在，其质量仍然较低。为了提高质量，作者采用了三种策略对网络Φ进行训练：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710132711049.png" alt="image-20230710132711049" style="zoom:80%;" />

$\hat{L}$ 是对不同尺度的预测的损失。首先，通过置信度感知显著性蒸馏（CSD）方案，从简单的样例到更复杂的例子中挖掘有价值的显著性知识；然后通过边界感知纹理匹配（BTM）策略对齐外观和预测的显著性图的边界。此外，多尺度一致性损失 $L_{ms}$ 确保了本文的方法对多尺度输入产生一致的预测。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710133102552.png" alt="image-20230710133102552" style="zoom:80%;" />

**Confidence-aware Saliency Distilling**

通过引入调节参数 $ρ$ （在训练阶段逐渐由0变成1），置信度表示为<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710134711106.png" alt="image-20230710134711106" style="zoom:80%;" />，则有以下式子：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710134755283.png" alt="image-20230710134755283" style="zoom:80%;" />  <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710134809335.png" alt="image-20230710134809335" style="zoom:80%;" />

具体来说，在一开始，我们的 $L_{csd}$ 将**低梯度**分配给硬样本，以从简单的样本中学习可靠的显著性知识。随着训练的进行，硬样本的梯度不断增加，以挖掘更有价值的显著性知识。四种损失的梯度对比如下：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710134948101.png" alt="image-20230710134948101" style="zoom:80%;" />

**Boundary-aware Texture Matching**

一般来说，显著性边界的外观与显著性预测具有相似的纹理。因此，匹配不同映射图之间的纹理可以指导我们的方法产生合理的显著性分数。该策略也适用于其他模式，如深度图、热图像和光流，通过在多模态数据中聚合丰富的外观信息，我们可以为我们的 USOD 方法提供更广义的指导。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710135504564.png" alt="image-20230710135504564" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710135736563.png" alt="image-20230710135736563" style="zoom:80%;" />

**Multi-scale Consistency**

显著性对象在多尺度输入中是一致的。因此，我们将输入图像的大小调整为一个参考比例尺，并鼓励我们的方法通过以下方法产生一致的预测：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710135849890.png" alt="image-20230710135849890" style="zoom:80%;" /> ，其中两个 y 为不同尺寸输入的 saliency predictions。

> 总结：

用激活图来辅助检测物体的方法可以参考。训练前期通过标签数据（对应本文的容易样本）学习一般知识，训练后期辅助无标签数据（对应本文中的难（硬）样本）知识学习。（**找时间看看代码**，关于激活图那块）

### 59_Grounding Counterfactual Explanation of Image Classifiers to Textual Concept Space_CVPR 2023_无代码

> 作者：Siwon Kim, Jinoh Oh, Sungjin Lee, Seunghak Yu, Jaeyoung Do, Tara Taghavi

​				单位：首尔国立大学数据科学与人工智能实验室（首位作者所属机构）

> 贡献：

背景：可解释的人工智能（XAI）旨在揭示黑盒深度神经网络的推理过程。在图像领域，热图式解释已被广泛研究用于解释图像分类器，但仅仅突出显示对模型结果有显著贡献的像素并不能回答直观且可操作的问题，如“区域的哪个方面很重要？是颜色还是图案？”而从突出显示的像素中得出人类可理解的理由则需要领域专家的干预，因此容易受到人类主观性的影响。相比之下，基于概念的解释可以提供更具人类可理解性和高级语义的解释。

基于概念的解释旨在为图像分类器提供简洁和人类易于理解的解释。然而，现有的基于概念的解释方法通常需要大量手动收集的概念注释图像。这是昂贵的，而且存在着人类偏见的风险。

因此本文提出一种新的**基于文本驱动概念的反事实解释方法 CounTE**，通过利用预训练的多模态联合嵌入空间，仅从文本中定义概念，而无需额外的概念标注数据集。通过生成基于文本驱动概念的概念反事实解释，可以解释目标分类器的结果，并提供对模型决策理由的语义理解，从而减少人类偏见的影响。

> 方法：

**投影**将图像从目标分类器的中间层映射到 CLIP 潜在空间。利用文本驱动的概念方向在 CLIP 空间中进行扰动；

**逆投影**将扰动嵌入映射回目标分类器空间，以便将其传递给剩余的目标分类器模块。

作者发现，由一个简单的神经网络组成的投影器/逆投影器可以有效地映射目标分类器和 CLIP 的两个潜在空间，并产生可靠的解释。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710162514091.png" alt="image-20230710162514091" style="zoom:80%;" align="left"/> <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230710162608944.png" alt="image-20230710162608944" style="zoom:80%;" />





图4展示了，无需标签数据，通过对齐映射空间中的特征来训练映射器和逆映射器，还保证了特征语义的一致性。

> 总结：

这种通过双向约束将两个空间特征对齐的方法比较新颖。

### 60_CIGAR: Cross-Modality Graph Reasoning for Domain Adaptive Object
Detection_CVPR 2023_无代码

> 作者：Yabo Liu，Jinghua Wang，Chao Huang，Yaowei Wang，Yong Xu

> 贡献：

背景：无监督域自适应目标检测（UDAOD）的目的是通过将知识从标记源域推广到未标记目标域来学习检测器。现有基于图的方法仅基于视觉特征来构建图，而**不考虑语义原型所携带的语言知识**，例如，数据集标签。

因此，本文提出跨膜态图推理自适应方法 **CIGAR** 来同时利用视觉和文本信息，还提出了一种判别性特征选择器来寻找最具判别性的特征，并将它们作为视觉图的节点，以提高效率和有效性。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230713200143307.png" alt="image-20230713200143307" style="zoom:80%;" />

> 方法：

1、通过奇异值分解将图像中包含丰富信息的特征选取出来，作为构成图的节点；

2、引入语义原型指导跨域之间的信息对齐。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230713205430965.png" alt="image-20230713205430965" style="zoom:80%;" />

> 总结：

这篇文章的主要贡献点，应该 1、是使用 SVD 挑选出信息丰富的特征作为图节点；2、利用标签的语义信息来指导跨膜态信息泛化（因为就算域不同，但域内数据属于同一批类别）

### 61_One-to-Few Label Assignment for End-to-End Dense Detection_CVPR 2023_有代码

> 作者：Shuai Li, Minghan Li, Ruihuang Li, Chenhang He, Lei Zhang

> 代码：https://github.com/strongwolf/o2f

> 贡献：

一对一（o2o， One-to-one）标签分配在基于 transformer 的端到端检测中起着关键作用，最近被引入到端到端密集检测的全卷积检测器中。然而，在基于全卷积网络（FCN）的密集检测中，o2o策略由于正样本数量有限而降低了特征学习效率。

鉴于 o2o 策略的局限性，本文旨在开发一种高效的基于 FCN 的密集检测器，实现无 NMS 的端到端训练。作者观察到，在 o2o 中，将与正样本语义相似的模糊 anchor 定义为完全的负样本是不合适的。这些模糊 anchor 可以在训练过程中同时计算正负损失，只要损失权重经设计好，就不会影响端到端能力。基于这一观察，作者提出了**一对多（o2f）**标签分配策略，为模糊 anchor 分配动态的**软分类标签**。

通过在训练的**早期**阶段使用较大的正标签度和较小的负标签度，网络可以**更有效地学习特征表示能力**；

而在训练的**后期**阶段，逐渐增加模糊 anchor 的负标签度，以指导网络**学习去除重复预测**。

通过将 o2f标 签分配方法应用于密集检测器 FCOS，实验证明了该方法在 COCO 和 CrowdHuman 数据集上的性能优于具有 NMS 的检测器。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230713231507069.png" alt="image-20230713231507069" style="zoom:80%;" />

关于 ambiguous anchor 的定义：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230713232448884.png" alt="image-20230713232448884" style="zoom:80%;" />

作者尝试了两种利用 ambiguous anchor 的方法：

方法一：把 one-to-one 改为 one-to-two

方法二：使用合适的软标签（效果更好）

结果对比：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230713232642768.png" alt="image-20230713232642768" style="zoom:80%;" />

> 总结：

软标签、标签迁移的方法可以参考

### 62_Mind the Label Shift of Augmentation-based Graph OOD Generalization_CVPR 2023_有代码

> 作者：Junchi Yu1,  Jian Liang, Ran He

> 代码：Mind the Label Shift of Augmentation-based Graph OOD Generalization

> 贡献：

背景：分布外（OOD）泛化是图神经网络（GNNs）中的一个重要问题。过去的研究主要依赖于学习一个在不同训练环境中表现良好的分类器。然而，对于非欧几里得图形的对应问题的研究相对较少。其中一个挑战是图形结构数据的环境稀缺性。一些之前的工作提出通过应用不同的图形编辑策略来生成增强的训练环境。然而，由于图形标签对图形结构敏感，图形编辑可能会改变增强图的标签，从而使图形 OOD 泛化变得困难。

本研究的动机是解决基于数据增强的图形 OOD 泛化中的标签偏移问题。

作者提出了一种名为 **LiSA （Label-invariant Subgraph Augmentation）** 的方法，通过生成标签不变的增强数据，解决增强环境之间存在的不一致的预测关系问题。

LiSA 能够**生成具有一致预测关系的多样化增强环境**，并促进学习一个不变的GNN。

通过在多个图形分类和节点分类数据集上进行广泛实验，作者证明了 LiSA 在不同的 GNN 模型上取得了不错的泛化性能。

> 方法：

LiSA 利用变分子图生成器提取局部预测模式，并高效地构建多个标签不变的子图。然后，利用这些子图构建具有一致预测关系的增强环境。

为了促进增强的多样性，LiSA 引入了基于能量的正则化，以扩大不同增强环境之间的配对距离。实验表明，LiSA 在节点级和图级 OOD 基准测试中，使用不同的 GNN 骨干网络都取得了不错的泛化性能。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230713234251345.png" alt="image-20230713234251345" style="zoom:80%;" />

> 总结：

* 在现有数据上做增强以提高泛化性能；

* 增强的度需要把握好，既要体现多样性又要保持预测一致性。

### 63_DiGeo: Discriminative Geometry-Aware Learning for Generalized Few-Shot Object Detection_CVPR 2023_有代码
> 作者：Jiawei Ma Yulei Niu Jincheng Xu Shiyuan Huang Guangxing Han Shih-Fu Chang

> 代码：https://github.com/Phoenix-V/DiGeo

> 贡献：

背景：广义 few-shot 目标检测的目的是对具有丰富注释的基类和训练数据有限的新类进行精确检测。现有方法难以平衡模型对基类和新类的检测性能。

本文作者出原因在于对所有类的**区分性特征**学习不足。因此作者提出 DiGeo, 学习类间分离类内紧凑的 Geometry-aware 特征。

为了指导特征聚类的分离，作者使用了一个离线单纯形等角框架（**ETF**， offline simplex equiangular tight frame）分类器，其权值作为类中心，最大等分离。

为了收紧每个类的集群，将**自适应的类特定的边距**纳入分类损失中，并鼓励相应特征靠近类中心。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714103651335.png" alt="image-20230714103651335" style="zoom:80%;" />

> 结果：

对两个基准数据集（VOC，COCO）和一个长尾数据集（LVIS）的实验研究表明，本文的方法可以在不影响基类检测的情况下，有效地提高对新类的泛化效果。

> 总结：

ETF 离线单纯形等角框架分类器比较有意思

### 64_DETR with Additional Global Aggregation for Cross-domain Weakly Supervised Object Detection_CVPR 2023_无代码

> 作者：Zongheng Tang， **Yifan Sun**， Si Liu， Yi Yang

> 贡献：

背景：作者认为 DETR 对跨域弱监督目标检测 CDWSOD 有很强的潜力：DETR 中的编码器和解码器都是基于注意机制的，因此能够跨整个图像聚合语义。聚合结果，即图像级的预测，可以自然地利用其进行领域对齐的弱监督。

因此，本文提出带有全局聚合的 CDWSOD 方法 **DETR-GA**，同时做出“实例级+图像级”预测，并利用“强+弱”监督。

具体做法：在 encoder、decoder 端分别加入**多个 class query** 和**一个 foreground query** 将语义信息聚合到图像级的预测当中。

首先，在编码器中，弱监督的类查询能够粗略地定位相应的位置，并排除来自非相关区域的干扰；解码器中的对象查询和前景查询在类语义上具有一致的一致性，从而使得强监督和弱监督在域对齐方面相互受益。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714111851185.png" alt="image-20230714111851185" style="zoom:80%;" />

> 总结：

class query 是所有类别每个类一个query，可学习。类别比较多的话就会导致计算量很大。

### 65_ScaleDet: A Scalable Multi-Dataset Object Detector_CVPR 2023_无代码

> 作者：Yanbei Chen, Manchen Wang, Abhay Mittal, Zhenlin Xu, Paolo Favaro, Joseph Tighe, Davide Modolo

> 贡献：

背景：现有的多数据集学习主要依赖于**手动重新标记**工作或**复杂的优化**来统一跨数据集的标签空间。

本文引入了一个简单但可用的方式  **ScaleDet** 来推导出用于多数据集训练的统一语义标签空间。

**ScaleDet** 通过**视觉-文本对齐**进行训练，以学习跨数据集的具有标签语义相似性的标签分配。一旦经过训练，ScaleDet 可以很好地在任何给定的（可见和不可见类的）上游和下游数据集上推广。

> 方法：

1、联合四个数据集的类别，共获得 2249 个类别，并通过文本编码器获得相应的 class labels；

2、由于不同数据集中对于同一对象的类名可能不同，因此提出语义相似性矩阵（下图右下角）计算 class labels 之间的相似性；

3、**硬标签分配**在概率空间中强制施加，以消除不同类别标签的歧义；**软标签分配**在语义相似性空间中施加，将每个视觉特征分配给具有不同语义相似性的文本嵌入，从而作为一个跨数据集关联相似的类标签正则化器。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714123733543.png" alt="image-20230714123733543" style="zoom:80%;" />

> 总结：

这里的软标签利用文本编码器统一不同数据集中**语义相同但标签相异**的类别语义可以借鉴。

通过CLIP text encoder 编码得到的 embedding，可以考虑从中挖掘类别之间的信息，或者是从中挖掘未知类的信息？

### 66_Universal Instance Perception as Object Discovery and Retrieval_CVPR 2023_有代码

> 作者：Bin Yan1,  Yi Jiang,  Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, Huchuan Lu

> 代码：https://github.com/MasterBin-IIAU/UNINEXT

> 贡献：

背景：以对象为中心的理解是计算机视觉中最重要和最具挑战性的问题之一。所有实例感知任务都是查找由某些查询指定的某些对象，如类别名称、语言表达式和目标注释，但是这个完整的领域被分成多个独立的子任务，如下图是一些子领域示意：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714132608998.png" alt="image-20230714132608998" style="zoom:80%;" />

本文作者认为，划分成这么多子领域并在子领域内各自研究存在以下缺点：

1. 参数冗余，独立设计阻碍了模型在不同任务和领域之间学习和共享通用知识，导致参数冗余。
2. 不同任务之间的相互协作的可能性被忽视了。例如，目标检测数据使模型能够识别公共对象，这可以自然地提高REC（表达式理解）和RES（表达式分割）的性能；
3. 由于固定大小分类器的限制，传统的对象检测器很难在具有不同标签词汇的多个数据集上联合训练，并在推理中动态改变对象类别进行检测。

由于基本上所有的实例感知任务都是根据某些查询找到特定的对象，这导致了一个自然的问题：我们能否设计一个统一的模型来一劳永逸地解决所有主流实例感知任务？

为此，本文提出了 **UNINEXT**，一个通用的下一代实例感知模型。首先根据不同的输入提示，将10个实例感知任务重组为三种类型：

1. **category names as prompts** (Object Detection, Instance Segmentation, VIS, MOT, MOTS)
2. **language expressions as prompts** (REC, RES, R-VOS)
3. **reference annotations as prompts** (SOT, VOS)

> 方法：

**UNINEXT** 首先在提示的指导下发现 N 个对象建议，然后根据实例提示的匹配分数从建议中检索最终的实例。

为了处理不同的提示模式，采用一个**提示生成模块**，它由一个文本编码器和一个视觉编码器组成；

然后使用一个**早期的融合模块**来增强当前图像的原始视觉特征和提示嵌入，为稍后的实例预测提供高度判别性的表示；

考虑到灵活的查询-实例方式，选择一个基于 transformer 的对象检测器作为实例解码器。具体来说，解码器首先生成N个实例建议，然后使用提示符从这些建议中检索匹配的对象。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714133727901.png" alt="image-20230714133727901" style="zoom:80%;" />

> 总结：

大一统模型

### 67_Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning_CVPR 2023_有代码
> 作者：Zeyin Song， Yifan Zhao， Yujun Shi， Peixi Peng， Li Yuan， Yonghong Tian

> 代码：https://github.com/zysong0113/SAVC

> 贡献：

背景：少样本类增量学习（FSCIL）的目的是在有限的样本中不断学习对新类的分类，而不忘记旧的类。处理 FSCIL 的主流框架首先是在基础阶段采用交叉熵（CE）损失进行训练，然后冻结特征提取器以适应新的类。

作者发现 CE 损失对于基础课程训练并不理想，因为它在表示方面的类分离很差，这进一步降低了对新类的泛化。作者想到，缓解这个问题的一个的方法是增加监督对比学习（SCL）。然而，通过实验发现，尽管 SCL 可以在不同的基类之间创建一个稍微更好的表示分离，但它仍然难以分离基类和新的类。

基于此，作者提出语义感知虚拟对比模型（**SAVC**），通过在 SCL 中**引入虚拟类**来方便分离新类和基类。

这些虚拟类是通过预定义的转换生成的，它们不仅作为表示空间中**不可见的类的占位符**，而且还**提供不同的语义信息**。

通过学习在虚拟类所培养的幻想空间中进行识别和对比，SAVC 显著提高了基类分离和新的类泛化，在三个广泛使用的 FSCIL 基准数据集上实现了新的最先进的性能。

> 方法：

**Semantic-aware class fantasy**

首先定义一个离散变换（即幻想）集合F，并假设其中有M个元素，即变换的数量。然后，我们可以为一个图像标签对$(x，y)$生成 M 个转换样本，表示 $F(x，y)= \{(xm，ym)\}^{M}_{m=1}$，其中$y_{m} = y×M + m$。$(x_m，y_m)$ 的下标表示第 m 次变换应用于$(x，y)$，而$(x_{1}，y_{1})$表示原始图像标签对。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714134837321.png" alt="image-20230714134837321" style="zoom:80%;" />

> 总结：

其中的语义类别想象空间本质应该也算是一种增强，后面可以细看。

### 68_Improving Weakly Supervised Temporal Action Localization by Bridging Train-Test Gap in Pseudo Labels_CVPR 2023_有代码

> 作者：Jingqiu Zhou ， Linjiang Huang， Liang Wang， Si Liu， Hongsheng Li

> 代码：https://github.com/zhou745/GauFuse_WSTAL

> 贡献：

背景：弱监督的**时间动作定位**目标是生成感兴趣动作的时间边界，同时也应对动作类别进行分类。基于伪标签的方法是一种有效的解决方案，近年来得到了广泛的研究。然而，现有方法在训练过程中生成伪标签，并在不同的管道或设置下的测试过程中进行预测，导致了训练和测试之间的差距。

本文提出从预测的动作边界中生成高质量的伪标签。首先，提出了一个**高斯加权融合模块**来保存动作实例的信息，并获得高质量的动作边界；其次，我们根据动作实例的置信度分数，将伪标签生成表示为**约束条件下的优化**问题；最后，我们引入了**∆伪标签**的思想，使模型具有自校正的能力。

> 方法：



<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714141701846.png" alt="image-20230714141701846" style="zoom:80%;" />

> 总结：

视频中伪标签的设计；

**∆伪标签**用于伪标签自校正的的思想可以考虑借鉴。

### 69_Weakly-Supervised Domain Adaptive Semantic Segmentation with Prototypical Contrastive Learning_CVPR 2023_有代码链接

> 作者：Anurag Das, Yongqin Xian, Dengxin Dai, Bernt Schiele

> 代码：https://github.com/anurag-198/WDASS

> 贡献：

背景：之前的工作在提高语义分割任务的**无监督领域自适应性能**方面已经付出了大量的努力，但与监督学习相比，性能上仍存在巨大的差距。

本文提出了一个通用的框架来结合使用不同的弱标签，例如，图像标签、点标签和粗标签来减少这种性能差距。

> 方法：

具体而言，利用这些弱标签来学习作为具有代表性的类特征的**更好的原型**，使用这些改进的原型来对类特征进行对比对齐。特别地，本文执行了**两种不同的特征对齐**：首先，将像素特征与**每个域内的原型对齐**；其次，以不对称的方式将像素特征从源对齐到目标域的原型。这种非对称对齐是有益的，因为它在训练过程中保留了目标特征。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714143357779.png" alt="image-20230714143357779" style="zoom:80%;" />

> 总结：

域内对齐和跨域对齐；

跨域对齐更好地提高泛化性，

### 70_Multi-Modal Representation Learning with Text-Driven Soft Masks_CVPR 2023_无代码

> 作者：Jaeyoo Park， Bohyung Han

> 贡献：

通过引入一种新的操作（**软掩码图像中与某个单词强相关的区域**）、损失（**图像-文本对比学习的 focal loss 版**）和数据增强策略（**掩码文本、渲染图像等多模态数据增强**），本文提出了一种在自监督学习框架内的视觉-语言表示学习方法。

> 方法：

对于给定的图像和掩码文本数据，分别编码以获得来自各自模态的嵌入标记序列，在单模态层面优化 ITC 目标，以便在融合它们之前对齐每个模态的嵌入空间；然后将嵌入标记的序列提供给多模态编码器以聚合这两种模式的信息，并学习解决 ITM（Image-Text Matching） 和 MLM（Masked Language Modeling）任务。

另外使用软掩码视觉特征从图像中学习属性。从正匹配分数的交叉注意图的Grad-CAM中生成软掩模，通过随机选择一个单词级的 Grad-CAM，并使用归一化的 Grad-CAM 作为视觉特征的软掩模。通过屏蔽重要的区域而不是完全地去除它们，生成的软掩码视觉作为一个硬但多样化的样本。此外，本文还引入了 ITC 的 focal 版本，使模型更关注困难的例子。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714145220237.png" alt="image-20230714145220237" style="zoom:80%;" />

### 71_Sample-level Multi-view Graph Clustering_CVPR 2023_无代码

> 作者：Yuze Tan, Yixi Liu, Shudong Huang, Wentao Feng, Jiancheng Lv

> 贡献：

背景：多视图聚类由于其处理异构数据的有效性而被研究。尽管最近的研究取得了验证上的成功，但仍存在一些严峻的挑战。特别是，以往的多视图聚类算法**很少考虑数据中的拓扑结构**，这是流形上数据聚类的关键。此外，现有的方法**不能充分探索不同视图之间局部结构的一致性**，因为它们以视图内的方式而不是视图间的方式揭示聚类结构。

本文提出通过**学习数据的拓扑结构来利用隐含的数据流形**。此外，考虑到多视图的一致性表现在一般相似的局部结构中，而不一致的结构为少数，我们进一步探索了**样本层次上多视图的交叉**，从而可以更好地保持交叉视图的一致性。

> 方法：

首先得到多视图变换；然后将这些视图特征进行拼接；那么就可以得到包含多视图特征的切片。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714151301395.png" alt="image-20230714151301395" style="zoom:80%;" />

> Mark：

后面可以细看一下。

### 72_Language in a Bottle: Language Model Guided Concept Bottlenecks
for Interpretable Image Classification_CVPR 2023_有代码

> 作者：Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, Mark Yatskar

> 代码：https://github.com/YueYANG1996/LaBo

> 贡献：

背景: 随着深度学习系统的发展，其在关键领域的应用受到了透明度的限制。为了解决这个问题，过去的研究主要集中在事后解释上，但这些解释可能不完整或不准确。此外，可解释的模型往往被认为性能较差。本研究旨在探索如何构建高性能的可解释模型，以及如何自动构建概念瓶颈模型。

 过去的研究主要集中在人工设计的概念瓶颈模型上，但这些模型需要手动指定概念，并且性能通常不如黑盒模型。另外，一些研究通过绕过概念瓶颈来提高性能，但这样会破坏模型的可解释性。

本文提出了一种名为 Language Guided Bottlenecks (**LaBo**)的方法来自动构建高性能的 Concept Bottleneck Models (CBMs)，而无需手动进行概念注释。通过结合语言模型(GPT-3)和语言-视觉模型(CLIP)，用于生成和选择瓶颈层的概念，旨在创建与黑盒模型性能相当的可解释模型。

> 方法：

LaBo 首先使用一个大型语言模型 GPT-3 为每个类生成一组候选概念。然后使用子模块优化来贪婪地为每个类选择一个概念的子集，从而最大限度地提高可辨别性和多样性。然后，使用 CLIP 将所选的概念与图像对齐。在概念和图像的相似性得分上应用线性层来学习一个权重矩阵，表示每个概念在最终分类中的重要性。这个权重矩阵是使用 GPT-3 之前的语言模型进行初始化的。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714160535173.png" alt="image-20230714160535173" style="zoom:80%;" />

> 总结：

模型可解释性；

粗筛 ---->  细粒度匹配

### 73_MaPLe: Multi-modal Prompt Learning_CVPR 2023_有代码

> 作者：Muhammad Uzair Khattak  Hanoona Rasheed  Muhammad Maaz  Salman Khan  Fahad Shahbaz Khan

> 代码：https://github.com/muzairkhattak/multimodal-prompt-learning

> 贡献：

背景：预训练的视觉-语言模型（V-L）如CLIP在下游任务中展现出了出色的泛化能力。然而，这些模型对输入文本提示的选择非常敏感，需要精心选择提示模板才能取得良好的性能。为了解决这个问题，一些方法通过学习提示来调整 CLIP 以适应下游任务。然而，现有方法只在 CLIP 的一个分支（语言或视觉）中学习提示，**这种单模态的提示学习方法并不理想**，因为它无法动态调整两个表示空间。

因此，本文提出了一种多模态提示学习方法（**MaPLe**），用于同时调整 CLIP 的语言和视觉分支，以改善视觉和语言表示之间的对齐。

> 方法：

与之前单模态提示的对比：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714164323598.png" alt="image-20230714164323598" style="zoom:80%;" />

MaPLe 采用联合提示方法，在视觉和语言分支中学习上下文提示，以实现对 CLIP 模型的微调，提高模型对下游任务的泛化能力。

通过在语言分支中添加可学习的上下文标记，并通过**耦合函数**明确地将视觉提示与语言提示相关联。通过不同的 Transformer 块中的单独可学习的上下文提示引入深度提示。在微调过程中，只学习**上下文提示 $P_0$**及**耦合函数 $F$**，模型的其余部分保持冻结状态。

框架：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714164752725.png" alt="image-20230714164752725" style="zoom:80%;" />

> 总结：

多模态端分别设置可学习提示;

通过耦合函数将两种模态提示建立关系，提高相互协同作用。

### 74_NeRF-RPN: A general framework for object detection in NeRFs_CVPR 2023_有代码

> 作者：Benran Hu, Junkai Huang, Yichen Liu, Yu-Wing Tai, Chi-Keung Tang

> 代码：https://github.com/lyclyc52/NeRF_RPN

> 贡献：

背景: 3D 物体检测对于机器人和自动驾驶等重要应用至关重要，需要对三维场景进行理解。然而，大多数现有的相关方法要求输入3D点云或至少从3D传感器获取的 RGB-D 图像。然而，最近在Neural Radiance Fields (NeRF)方面的进展提供了一种有效的替代方法，可以从2D多视图图像中提取底层三维场景的高度语义特征。

目前的 3D 物体检测方法要么使用单个图像，要么利用多个图像的多视图一致性。然而，这些方法仍然使用 2D 特征来指导相关的3D预测，而没有充分利用NeRF中的内在 3D 信息

 鉴于NeRF提供了 3D 场景的结构细节，并且适用于3D训练，本文的动机是将 RPN 引入 NeRF，提出一种 NeRF-RPN 框架，直接在从多视图图像中学习的 NeRF 表示上进行操作，以提供 3D 区域建议。通过采用 “3D到3D学习” 范式，充分利用 NeRF 中的 3D 信息，并直接在 3D 空间中预测 3D 区域建议，NeRF-RPN 将成为NeRF 中 3D 物体检测的强大工具。

> 方法：

NeRF-RPN 引入了 3D 感兴趣区域（ROIs），用于在给定的 NeRF 表示中提出物体的 3D 边界框。该网络将从 NeRF 中提取的 3D 体积信息作为输入，并直接输出 ROIs 的 3D 边界框。

NeRF-RPN 的架构，包括在 NeRF 中采样点、提取 RGB 和密度特征、3D 主干网络、3D 特征金字塔网络和 3D RPN 头部。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714174222723.png" alt="image-20230714174222723" style="zoom:80%;" />

### 75_MetaViewer: Towards A Unified Multi-View Representation_CVPR 2023_无代码

> 作者：Ren Wang，Haoliang Sun，Yuling Ma，Xiaoming Xi，Yilong Yin

> 贡献：

背景：现有的多视图表示学习方法通常遵循特定统一的管道，从每个视图中提取潜在特征，然后将它们融合或对齐以获得统一的对象表示。然而，手动预先指定的融合函数和对齐标准可能会降低所得到表示的质量。

因此，本文从元学习的角度提出了一种新的**统一到特定**的多视图学习框架，其中统一表示不再涉及手动操作，而是自动从一个名为 MetaViewer 的元学习器得到。具体地说，我们将视图特定潜在特征的提取和融合定义为一个嵌套优化问题，并采用双层优化方案进行求解。MetaViewer 自动将视图特定特征融合成统一的特征，并通过观察从所有视图统一到特定视图的重建过程来学习最优融合方案。

> 方法：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714175817594.png" alt="image-20230714175817594" style="zoom:80%;" />

### 76_OSAN: A One-Stage Alignment Network to Unify Multimodal Alignment and Unsupervised Domain Adaptation_CVPR 2023_代码暂无

> 作者：Ye Liu, Lingfeng Qiao, Changchong Lu, Di Yin, Chen Lin, Haoyuan Peng, Bo Ren

> 贡献：

背景：从单模态扩展到多模态是无监督域自适应（UDA）的一个关键挑战。无监督多模态域适应中两个主要问题：域适应和模态对齐。处理这两个问题的一种直观方法是在两个独立的阶段中完成这些任务：对齐模式，然后进行领域适应，反之亦然。然而，在大多数现有的两阶段研究中，领域和模式并没有关联，它们之间的关系也没有被利用来相互提供互补的信息。

在本文中，作者将这两个阶段**统一为一个阶段**，以同时对齐域和模态，提出了一个基于张量的对齐模块（**TAL**）来探索域和模态之间的关系。通过这种方法，域和模式可以充分地相互作用，并指导它们利用互补的信息以获得更好的结果。此外，为了建立域之间的桥梁，提出了一个动态域生成器（**DDG**）模块，通过以自监督的方式混合两个域的共享信息来构建过渡样本，这有助于模型学习一个域不变的公共表示空间。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714181835486.png" alt="image-20230714181835486" style="zoom:80%;" />

> 方法：

该方法包含四个部分：多模态特征提取，基于张量的对齐，动态域生成器和特定任务头部。首先，编码器作为多模态特征提取器，将源和目标多模态数据映射到不同的潜在空间。然后，利用一种有效的张量表示方法，提出了一个 TAL 模块，通过模态和域之间的连续交互来获取关系信息。因此，模态和域在一个阶段同时对齐。然后，DDG 模块混合不同领域的信息来创建新的过渡领域。最后，网络分为三个分支：类别分类、领域对抗学习和差异消除。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714182004344.png" alt="image-20230714182004344" style="zoom:80%;" />

### 77_Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection_CVPR 2023_有代码

> 作者：Fan Lu，Kai Zhu，Wei Zhai，Kecheng Zheng，Yang Cao

> 代码：https://github.com/LuFan31/ET-OOD

> 贡献：

语义一致性分布外检测（SCOOD）的目的是从预期的数据分布中识别出异常值。当不进行区别时，分布内样本和分布外样本的共存会加剧模型的过拟合。

为了解决这个问题，本文提出了一种新的不确定性感知最优传输方案。包括一个基于能量的传输（**ET**）机制，该机制估计不确定性的波动成本，以促进语义不可知表示的分配，以及一个**集群间扩展策略**，通过扩大相应的边际距离来增强不同集群之间语义属性的识别。此外，还提出了一个**t-能量评分**，以减轻并行传输和分类器分支之间的幅度差距。

> 方法：

动机，相比于基于欧氏距离的方法，基于能量的度量方式能够更好地适应不同数据集之间相同类别的标签迁移问题。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714190946710.png" alt="image-20230714190946710" style="zoom:80%;" />

框架：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714191202295.png" alt="image-20230714191202295"  />

### 78_DivClust: Controlling Diversity in Deep Clustering_CVPR 2023_有代码

> 作者：Ioannis Maniadis Metaxas，Georgios Tzimiropoulos，Ioannis Patras

> 代码：https://github.com/ManiadisG/DivClust

> 贡献：

背景: 聚类是机器学习领域的一个重要研究课题，近年来深度学习在聚类中取得了显著的成功。然而，现有的深度聚类方法没有解决一个重要问题，即如何高效地生成给定数据集的多个多样性聚类结果。多样性聚类对于共识聚类非常重要，而共识聚类已被证明比单一聚类结果产生更好、更稳健的结果。

过去的研究主要集中在生成单一聚类结果上，对于多样性聚类的研究相对较少。现有的方法通常通过多次聚类数据集来增加多样性，但这种方法无法保证或控制多样性程度，并且计算成本较高。其他方法则通过创建和聚类多样性特征子空间来实现多样性聚类，但这些方法无法直接应用于深度学习框架。

鉴于现有方法在多样性聚类方面的不足，本研究提出了 **DivClust** 方法。该方法可以直接应用于现有的深度聚类框架中，通过引入**多样性控制损失**，生成多个具有所需多样性程度的聚类结果。DivClust 在计算成本上非常低，并且不需要对基础深度聚类框架进行超参数调整，因此使用简单且计算效率高。

实验证明，DivClust 方法能够有效地控制多样性，并生成优于单一聚类结果的共识聚类解决方案，从而提高基础深度聚类框架的性能。

> 方法：

本文方法由两个部分组成：1、一个新的损失函数，可以纳入深度聚类框架中，通过应用阈值集群相似性来控制集群多样性；2、动态估计阈值，使集群模型足够多样化，根据用户定义的度量。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714194833382.png" alt="image-20230714194833382" style="zoom:100%;" />

首先使用集群A,B中的样本计算相似性矩阵 S：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714195215425.png" alt="image-20230714195215425" style="zoom:80%;" />，然后使用损失函数强制聚类A与聚类B的聚合相似度不超过 d：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714200305392.png" alt="image-20230714200305392" style="zoom:80%;" />

动态估计阈值：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714200413506.png" alt="image-20230714200413506" style="zoom:80%;" />

### 79_Rebalancing Batch Normalization for Exemplar-based Class-Incremental Learning_CVPR 2023_无代码

> 作者：Sungmin Cha1, Sungjun Cho2, Dasol Hwang2, Sunwon Hong1, Moontae Lee2,3, and Taesup Moon

> 贡献：

背景: 连续学习（CL）是一种在逐个到达的数据集上高效学习神经网络的方法，而不需要每次到达新数据集时重新训练。然而，由于模型通常在每个步骤上在偏向当前任务的数据集上进行训练，导致神经网络在CL过程中往往面临稳定性和可塑性之间的次优权衡问题。为了解决这个问题，研究人员一直致力于解决所谓的灾难性遗忘现象。

在类增量学习（CIL）中，最近引起关注的是需要在每个增量步骤中学习以前未见过的类别的设置。大多数最先进的 CIL 算法通过维护一个小的样本存储器来存储以前使用的训练数据的子集，并将其与当前任务的数据集相结合，以减轻过去知识的遗忘。然而，基于样本的 CIL 的一个关键问题是**模型预测对最近学习的类别产生严重偏差**，这是由于当前任务的训练数据与过去任务的训练数据之间的不平衡导致的。为了解决这个问题，最近提出了一些解决方案，包括偏差校正、统一分类器和分离的Softmax等，这些方法在所有已学习的类别上都显著提高了CIL方法的准确性。

尽管在 CIL 中广泛使用基于CNN的模型作为特征提取器，并且这些模型默认使用批归一化（BN），但是由于 BN 是为 CNN 的单任务训练而设计的，**直接应用于基于样本的 CIL 会导致统计数据偏向当前任务**，因为当前任务和过去任务的数据在一个小批次中是不平衡的。因此，本文提出了一种新的批归一化变体，**旨在解决CIL中的数据不平衡问题**，并通过实验证明其在多个数据集上的优越性。

> 方法：

作者提出了一种任务平衡的批量归一化（**TBBN**）机制。TBBN 在训练过程中计算任务平衡的经验均值和方差，并使用它们对测试样本进行归一化。作者认为，这种方法是必要的，以防止由于训练和测试样本分布不匹配而导致性能下降。此外，作者建议以任务平衡的方式学习 BN 的仿射变换参数。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714200828037.png" alt="image-20230714200828037" style="zoom:100%;" />

### 80_Heterogeneous Continual Learning_CVPR 2023_无代码

> 作者：Divyam Madaan, Hongxu Yin, Wonmin Byeon, Jan Kautz, Pavlo Molchanov

> 贡献：

背景: 深度神经网络的创新不断涌现，但是现有的持续学习方法主要集中在将单一架构适应新任务/类别，而忽视了将现有解决方案适应新架构的问题。传统的持续学习方法通常假设网络架构不变，而现有的方法也无法处理网络架构的变化。

随着实际应用的需求，不断升级更强大的网络架构对于提供最佳用户体验和竞争优势至关重要，但是存储以前的数据并重新训练模型是计算上昂贵且不可行的。

因此，本文提出一种**新的持续学习框架**，可以在不存储以前的数据的情况下更新最先进的深度学习架构，并保留以前学到的知识。

> 方法：

本文提出的方法通过允许学习者切换网络架构并适应新架构来解决异构持续学习（**HCL**）问题，从而提高与标准持续学习相比的性能。该方法还消除了存储过去数据经验的回放缓冲区的需求。HCL 的学习目标是在一系列任务上训练一系列网络，而不会忘记先前任务的知识。该方法结合了**知识蒸馏**和标签平滑，以改善不同架构之间的知识传递。此外，还引入了一种称为**快速深度反演**的技术，用于生成先前任务特征的合成示例，有助于提高 HCL 中的数据效率。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714201852327.png" alt="image-20230714201852327" style="zoom:100%;" />

### 81_PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning_CVPR 2023_有代码

> 作者：Huiwei Lin, Baoquan Zhang, Shanshan Feng*, Xutao Li, Yunming Ye

> 代码：https://github.com/FelixHuiweiLin/PCR

> 贡献：

背景: 在线类增量连续学习是连续学习的一种特殊任务，其目标是从数据流中不断学习新的类别，但每个样本只能看到一次，这导致存在灾难性遗忘问题。现有的基于重放的方法通过在代理或对比方式下保存和重放部分旧数据来有效缓解灾难性遗忘问题。然而，这两种重放方式都存在一些限制，如**类别不平衡问题**和**样本数量有限**等。

通过对这两种重放方式进行全面分析，作者发现它们可以互补。受到这一发现的启发，提出了一种名为 **PCR** 的新的重放方法，通过在对比方式下使用代理替换锚样本，有效解决了类别不平衡问题，并保持了模型的更快收敛性能。
方法:

> 方法：

该方法旨在通过控制旧类和新类之间的梯度传播来解决灾难性遗忘问题。发现**不平衡的梯度传播**是灾难性遗忘的主要原因，新类主导了这个过程，使得新类的样本高度可区分，而旧类的样本变得不可分。现有的基于代理的方法通过选择特定的锚点到代理对来控制梯度传播，但这可能会损害模型学习新类的泛化能力。另一方面，基于对比度的方法依赖于同一批次的样本，但缺乏代理的支持。

为了克服这些局限性，本文结合了基于代理和基于对比度的方法。在对比度损失中，不使用样本作为锚点到样本对，而使用代理。这允许启发式选择锚点到代理对。通过结合这两种方法，在减轻灾难性遗忘方面能取得更好的性能。

![image-20230714203437042](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714203437042.png)

### 82_Cloud-Device Collaborative Adaptation to Continual Changing Environments in the Real-world_CVPR 2023_代码暂无
> 作者：Yulu Gan1*, Mingjie Pan1*, Rongyu Zhang2, Zijian Ling3,Lingran Zhao1, Jiaming Liu1, Shanghang Zhang

> 贡献：

背景: 在真实世界中，环境经常发生变化，而轻量级设备上的模型在分布变化下会出现性能下降的问题。过去的设备模型存在两个主要限制：一是由于设备计算能力的限制，无法及时更新模型，导致在分布变化下性能滞后；二是轻量级模型的泛化能力有限，无法应对持续变化的环境。

鉴于轻量级模型的限制，本研究提出了一种新的学习范式，即**云设备协同持续适应**，旨在促进云端和设备之间的合作，提高设备模型的泛化能力。基于这一范式，研究者进一步提出了基于不确定性的视觉提示适应的师生模型，将云端大模型的泛化能力传递给设备模型，以应对持续变化的环境。

![image-20230714202855629](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714202855629.png)

> 方法：

本文提出的云设备协同持续适应（CCA）范式旨在改善部署在设备上的轻量级模型在不断变化的目标领域中的性能。该范式由云端的教师模型和设备上的学生模型组成，教师模型是一个具有强大泛化能力的大模型，用于**提升学生模型的泛化能力**。CCA 范式结合了基于不确定性的采样（**UGS**）和基于不确定性的视觉提示学习策略（**VPLU**），以优化师生模型。

![image-20230714204144211](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230714204144211.png)

### 总结：

* **关于prompt：**

Visual-Language Prompt Tuning with Knowledge-guided Context Optimization通过实验表明，相比较于可学习的prompt，CLIP原本**手工设计的prompt的泛化性能更好一些**；MaPLe: Multi-modal Prompt Learning与以往只从文本端（大多数）或图像端输出prompt的方式不同，这篇文章提出从文本域图像两个模态端均输入prompt，而且通过耦合函数为两个模态间的prompt建立相关联系从而**提高不同模态提示的协同作用**；

* **关于特征学习：**

Texture-guided Saliency Distilling for Unsupervised Salient Object Detection**用激活图来辅助检测物体**的方法可以参考。训练前期通过标签数据（对应文中的容易样本）学习一般知识，训练后期辅助无标签数据（对应文中的难（硬）样本）知识学习。关于激活图那块可以看看代码；Grounding Counterfactual Explanation of Image Classifiers to Textual Concept Space虽然针对的是模型可解释性任务，但其中对模型特征嵌入空间和CLIP特征嵌入空间使用**双向约束**来保持语义一致性的方式还挺新颖的；One-to-Few Label Assignment for End-to-End Dense Detection通过实验表明，将模糊anchor利用起来，在训练早期为其赋予较多的正标签较少的负标签，能够促进模型学好物体的特征；在训练后期赋予标签的形式与早期反过来则可以减少重复预测，起到NMS的作用。这种**软标签**的方式还挺好的，结合OWOD那篇文章用到的标签迁移，可以从这个角度去促进模型学习有区分度的特征；

* **关于文本：**

ScaleDet: A Scalable Multi-Dataset Object Detector将不同数据集中的类别名称输入获得相应的文本嵌入后，利用**文本嵌入之间的相似性挖掘跨数据集中表示同类别的语义关系**；可以参考这一做法挖掘统一数据集中不同类别之间的关系；

* **关于伪标签：**

Improving Weakly Supervised Temporal Action Localization by Bridging Train-Test Gap in Pseudo Labels中**∆伪标签**用于**伪标签自校正**的的思想可以考虑借鉴；

## 20230721

### 83_Random Boxes Are Open-world Object Detectors_ICCV 2023_有代码

> 作者：Yanghao Wang1, Zhongqi Yue1,2, Xian-Sheng Hua3, Hanwang Zhang1

> 代码：https://github.com/scuwyh2000/RandBox

> 贡献：

背景：现有 OWOD 方法选取候选 proposals 都是以已知类标签作为约束，这导致模型对已知类产生 bias，因此对于 unknown 的召回率较低。

由于随机化是消除许多任务和领域的偏见的有效工具，受此启发，作者提出 **Random Boxes**：通过随机产生候选框的方式来消除模型对于已知类的偏见。另外，作者还提出一种新的用于选择 unknown **伪标签的评价分数**。

本文方法的有效性源于以下两个由随机性所带来的好处。首先，由于随机化独立于有限的已知对象的分布，随机 proposals 生成为防止训练被已知对象混淆；其次，无偏训练通过使用所提出的匹配分数来鼓励模型探索更多的 proposals，该匹配分数不会惩罚那些预测分数与已知目标不匹配的随机提案。

> 方法：

**RandBox** 的 proposals 由随机方式产生，特征由 Fast R-CNN 提取，检测头使用的是  Sparse R-CNN 的检测头。（**可以考虑换成 FQ-R-CNN**）

![image-20230720153354147](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720153354147.png)

* 随机生成的 proposals：
  * 训练的时候，为每张图像从高斯分布中采样 500 个proposals；
  * 推理时，为了移除预测的随机性，为每张图像使用 **10000 个预定义的 proposals**（r 10 scales, 10 aspect ratios and 100 spatial locations）。

* 动态 k 匹配：
  * 对于batch中的所有 gt，根据其与所有 proposals 的 IOU 之和排序，和越大的分配更多的 proposals。
* unknown 选择依据——matching score：
  * 基于特征迁移的假设，已知类对象和未知类对象的相似性通常大于背景和已知类对象的相似性，在排除掉已经与 gt 匹配的 proposals 之后，通过 <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720154509088.png" alt="image-20230720154509088" style="zoom:80%;" />来选择 top-N 个 proposals 作为 unknown；
  * 公式中的 $|K| + 1$ 表示已知类和未知类位置的预测分数（注：没有背景类位置），因此该分数本质上计算了yˆ对应于前景的可能性。

> 消融：

论文中没有明说第二行的 Ours 是不是比第一行多了 动态 k 匹配（感觉应该是，到时候看看代码），如果是的话，看起来 rand box 带来的收益其实没有多少。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720154824375.png" alt="image-20230720154824375" style="zoom:80%;" />

> 总结：

* 其中的动态 k 匹配方法可以考虑加到自己的模型中；
* 动态 k 可以作为和软标签（之类的）的对比方法。

### 84_Adversarial Reciprocal Points Learning for Open Set Recognition_TPAMI'21_有代码

> 作者：Guangyao Chen, Peixi Peng, Xiangqian Wang, and Yonghong Tian

> 代码：https://github.com/iCGY96/ARPL

> 贡献：

背景：如何**同时降低**标记已知数据的经验分类风险和潜在未知数据的开放空间风险是**开放集识别 OSR** 任务的关键挑战。现有基于 softmax 或是 原型的分类器都只聚焦于已知类特征，没有考虑无标签的未知类特性。

本文提出的 ARPL 以**最大限度地减少已知分布和未知分布的重叠**，而不损失已知的分类精度。
每个**倒数点**由类外空间与相应的已知类别学习，并采用多个已知类别之间的对抗来降低经验分类风险。然后，本文还提出了一种**对抗性边际约束**，通过限制互反点构成的潜在开放空间来降低开放空间风险。为了进一步估计来自开放空间的未知分布，基于互反点和已知类之间的对抗机制，设计了一种**实例化对抗增强**方法来生成多样化的训练样本，有效地增强模型对未知类的识别能力。

![image-20230720193551655](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720193551655.png)

图1.(a)图像空间具有无限的开放空间，但大多数未知数的深嵌入表示分布在深空间的有限低幅度区域。MNIST(蓝色)用于已知训练，KMNIST(绿色)、SVHN(黄色)和CIFAR-100(橙色)用于开放集评估，它们与MNIST的相似度逐渐降低。本文研究了如何通过减少已知样本的深度特征与不同未知样本的特征之间的重叠来提高识别。

> 方法：

 **倒数点** $P^{k}$  定义为潜在表征空间中除该类别特征外其他类别的特征表示集（其他已知类+未知类），表示为一个 m 维的可学习表示，通过如下是自来学习优化：

* 每个已知的类在空间位置和角度方向上都应与其倒数点相反：

![image-20230720194137020](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720194137020.png)

* 其中 C(x) （样本 x 特征经过C 变换）和 Pk 之间的距离越大，导致样本 x 以越大的概率分配为 k：

![image-20230720194351388](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720194351388.png)

* 约束损失：

![image-20230720194420388](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720194420388.png)

**对抗边界约束**

为了降低式开放空间风险，使用对抗边际约束 (AMC) 来约束开放空间，具体为对于类 k 而言，将所有开放空间的特征 到 类 k 的倒数点的距离约束在一定范围内：

![image-20230720194847776](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720194847776.png)

**R是一个可学习的边界**

![image-20230720195128427](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720195128427.png)

![image-20230720195153837](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720195153837.png)

> 总结：

* 南哥小论文的参考文献
* 倒数点感觉就是**从原型的对立面**去考虑
* 作者文中举的一个分类猫的例子：大多数模型主要是学习“what is a cat？”，只考虑了问题的一个很小的点；而本文提出的**倒数点**表征了“non-cat”的特征，相当于考虑了问题其其他方面。

### 85_PromptCAL: Contrastive Affinity Learning via Auxiliary Prompts for Generalized Novel Category Discovery_CVPR 2023_有代码

> 作者：Sheng Zhang，Salman Khan，Zhiqiang Shen，Muzammal Naseer，Guangyi Chen，Fahad Shahbaz Khan

> 代码：https://github.com/sheng-eatamath/PromptCAL

> 贡献：

背景：虽然现有的半监督学习模型在使用无注释的数据进行学习中取得了显著的成功，但由于封闭集假设，它们大多无法对未标记新类数据进行学习。

在本工作中，作者的目标是一个实用的但未被充分探索的**广义新类别发现（GNCD）**设置。

GNCD 设置旨在通过利用部分标记的已知类的信息，对来自已知类和新类的未标记训练数据进行分类。

训练数据 $D = D_l \cup D_u$，$D_l$为带标签的 known， $D_u$ 包含不带标签的 known 和 novel。

本文提出了一种具有辅助视觉提示的两阶段对比偏好学习方法 **PromptCAL**，首先，通过**提示正则化损失 Discriminative Prompt Regularization (DPR) loss**来加强经过prompt改编的预训练视觉编码器对偏好关系的语义辨别性；然后通过**对比偏好学习 Contrastive Affinity Learning (CAL)**，基于迭代半监督偏好图生成方法来校准语义表示进行语义增强监督。

![image-20230717134729858](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717134729858.png)

> 方法：

首先，CAL 基于生成的偏好图发现大量可靠用于的 DPR 损失和对比损失的 pseudo positives。这些具有语义感知能力的伪标签进一步增强 DPR 监督的语义辨别能力。其次，DPR 对集成提示的语义表示进行了正则化处理，便于在 CAL 的下一步发现更准确的伪标签。因此，随着模型和提示表示的增强，可以获得更高质量的伪 pseudo positives 来进行进一步的自训练，并获得更好的语义聚类。

![image-20230717142638343](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717142638343.png)

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717151445585.png" alt="image-20230717151445585"  />

> 总结：

也是采用了memory bank 的思想，并且基于该 memory bank 来构建偏好子图，用来进行伪标签。

> mark：可看看代码

### 86_Momentum Contrast for Unsupervised Visual Representation Learning_CVPR 2020_有代码

> 作者：Kaiming He Haoqi Fan Yuxin Wu Saining Xie Ross Girshick

> 代码：https://github.com/facebookresearch/moco

> 贡献：

提出动量对比（**MoCo**）用于无监督的视觉表示学习。从对比学习作为字典查找的角度来看，作者构建了一个带有**一个队列**和**一个移动平均编码器**的动态字典。通过建立一个**大的**+**一致的**动态字典，促进对比无监督学习。

> 方法：

三种不同方法的对比：

![image-20230717225141092](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717225141092.png)

![image-20230717225619612](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717225619612.png)

a. 传统端到端方式：原始的自监督学习方法里面的这一批负样本就相当于是有个字典 (Dictionary)，字典的 key 就是负样本，字典的value就是负样本通过 Encoder 之后的特征，负样本的规模则是 batch size。毫无疑问是 batch size 越大效果越好的，但是由于算力的影响 batch size 不能设置过大，因此很难应用大量的负样本，效率较低。

b. 针对很难应用大量的负样本的问题，则可以采用一个较大的memory bank存储较大的字典。对于给定的一个样本 ，选择一个正样本 (这里正样本的对于图像上的理解就是输入的 data augmentation 版本)。采用一个较大的 memory bank 存储较大的字典，这个 memory bank 具体存储的是所有样本的 representation(涵盖所有的样本，比如样本一共有 60000 个，那么 memory bank 大小就是 60000，字典大小也是60000)。采样其中的一部分负样本 ，然后使用 loss function 来将输入与正样本之间的距离拉近，负样本之间的距离推开。这次只更新 Encoder 的参数，和几个采样的key值 。因为这时候右边没有了 Encoder 的反向传播，但每个 step 都更新左边 Encoder 的参数，就导致两边不一致的问题。即：每个step编码器都会进行更新，这样最新的 query 采样得到的 key 可能是好多个step之前的编码器编码得到的 key，因此丧失了一致性。

因此，始的端到端自监督学习方法的一致性最好，但是受限于 batch size 的影响。而采用一个较大的 memory bank 存储较大的字典的字典可以设置很大，但是一致性却较差。

c. **MoCo**维护一个动态队列 + key 的 encoder 采用动量更新的方式：

Encoder 的输入是一个 Batch 的样本 x 的增强版本 x_q；

Momentum Encoder 的输入是一个 Batch 的样本 x 的另一个增强版本 x_k 和 队列中的所有样本 x_queue，x_queue 通过 Momentum Encoder 得到代码中的变量 queue；

Encoder 在每个 step 都会通过反向传播更新参数，假设 1 个 epoch 里面有500 个 step，Encoder 就更新 500次。Momentum Encoder 在每个 step 都会通过动量的方式更新参数，假设 1 个 epoch 里面有500 个 step，Momentum Encoder 就更新 500次，只是更新的方式是：![image-20230717230543551](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717230543551.png)

![image-20230717230629782](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717230629782.png)

### 87_Glocal Energy-based Learning for Few-Shot Open-Set Recognition_CVPR 2023_有代码

> 作者：Haoyu Wang1* Guansong Pang2* Peng Wang3* Lei Zhang1 Wei Wei1 Yanning Zhang1†

> 代码：https://github.com/00why00/Glocal

> 贡献：

背景：少样本开集识别（FSOR）的目的是通过少量样本的学习，将一个已知类样本分类为一个预定义的，封闭集的类，同时能够拒绝来自未知类的样本。

本文提出了一个基于能量的混合模型（**GEL**）来解决这个问题，GEL 由两个分支组成：其中**分类分支**学习将一个样本分类为一个封闭集中的类，而**能量分支**明确地估计开集概率。为了实现开放集样本的整体检测，本文的模型利用**类级**和**像素级**特征来学习基于 **glocal** 能量的分数，其中使用**类级特征学习全局能量分数**，而使用**像素级特征学习局部能量分数**。无论是类级特征或像素级特征中**偏离** few-shot 例子的样本，模型将为其分配**大的能量分数**，否则就分配小的能量分数。

> 方法：

**对于 class-wise 分支**，对 support 集中的每个类别，按照 support 集中该类的 embedding 平均值计算类原型：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717205613881.png" alt="image-20230717205613881" style="zoom:80%;" />

然后对类原型使用自注意力进行增强，<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717205713658.png" alt="image-20230717205713658" style="zoom:80%;" />

通过欧氏距离计算 query embedding 与各原型的相似性分数：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717205825725.png" alt="image-20230717205825725" style="zoom:80%;" />

然后使用 softmax 得到闭集分类分数：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717205938318.png" alt="image-20230717205938318" style="zoom:80%;" />，

用交叉熵损失作为优化目标<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717210046841.png" alt="image-20230717210046841" style="zoom:80%;" />

为了对开集样例有更完整的认识，作者还提出了 **pixle-wise 分支**，首先和 class-wise 分支类似，获得类别特征map：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717212050270.png" alt="image-20230717212050270" style="zoom:80%;" />

为了减少计算量，首先通过 卷积+BN+RELU 对特征map进行维度减半操作：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717212213947.png" alt="image-20230717212213947" style="zoom:80%;" />

对于每个 query pixels，根据相似性从对应类特征图pixels中选择topk个并求和：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717212644730.png" alt="image-20230717212644730" style="zoom:80%;" />

 **Energy-based Module**

对于 query样例 x，分别根据前面两个纷纷之得到的 s 计算能量：$E = E_{c} + E_{f}$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717212930489.png" alt="image-20230717212930489" style="zoom:80%;" />

energy loss:<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717213404495.png" alt="image-20230717213404495" style="zoom:80%;" />(Mk和Mu分别为闭集样本和开集样本的边缘，文中作者说这俩值简单的设置为 -1 和 1)

总体 energy loss：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717213514329.png" alt="image-20230717213514329" style="zoom:80%;" />

最终模型优化损失：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717213538526.png" alt="image-20230717213538526" style="zoom:80%;" />

![image-20230717165355131](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717165355131.png)

> 总结：

* 自注意力用于特征增强的作用，本文中用在了原型上（据文中描述，之前已经有文章这么用了）
* 这里为了对开集样例有完整的认识，从 class 和 pixel 两个方面（可以理解为粗粒度和细粒度）来获得用于计算能量的相似性分数，可以借鉴，再目标检测中可以对应于整张图片和proposals。

### 88_PROTOCON: Pseudo-label Refinement via Online Clustering and Prototypical Consistency for Efficient Semi-supervised Learning_CVPR 2023_无代码

> 作者：Islam Nassar1 Munawar Hayat1 Ehsan Abbasnejad2 Hamid Rezatofighi1 Gholamreza Haffari1

> 贡献：

背景：基于置信度的伪标记是半监督学习（SSL）中的主要方法之一，它将基于未标记数据的高可信度预测作为训练模型的额外目标。

本文提出了一种新的 SSL 方法：**PROTOCON**，通过**利用近邻的信息来细化伪标签的表示**从而达到对模型训练更好监督的目的。近邻定义为在训练过程中，使用在线聚类的方法，在使用原型损失训练的嵌入空间中形成良好的聚类。PROTOCON 的在线特性允许它在一个训练周期中利用整个数据集的标签历史，在接下来的周期中细化标签，而不需要存储图像特征。因此，它可以以低成本无缝隙地扩展到更大的数据集。另外，PROTOCON 还通过引入一个辅助的**自监督损失**，解决训练初始阶段的缺乏训练信号（好的标签，因为有标签数据较少，而训练初期为无标签数据生成的伪标签质量也不咋好）。

![image-20230718211213048](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230718211213048.png)

> 方法：

如图2，对于一张图像，首先获得两个伪标签，一个是通过其若增强版本图像的分类器的 sotmax 输出 $p^{w}$，另一个是表示该样本近邻的聚合伪标签 $z^{a}$；为了确保这两个标签是基于不同的表征，将 image neighbourhood 定义为嵌入空间的在线聚类。

![image-20230718211239247](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230718211239247.png)

![image-20230718212258006](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230718212258006.png)

> 总结：

* 伪标签优化
* 半监督里由于有标签数据比较少，伪标签优化似乎是最近用的比较多的方法，本文中由于是分类任务，而且 batch size比较大，所以可以用聚类的方法
* 考虑一下是否能将**伪标签优化**的思想用到 OWOD 中，原型的方法可以试试

### 89_Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization_CVPR 2023_有代码

> 作者：Lian Xu1, Wanli Ouyang2, Mohammed Bennamoun1, Farid Boussaid1, and Dan Xu3

> 代码：https://github.com/xulianuwa/MMCST

> 贡献：

背景：之前用于解决弱监督密集目标定位的方法主要依赖于Class Activation Mapping (CAM)来生成定位图，但CAM无法很好地处理像素级别的内类别变化。

本文提出了一种新的方法，通过**学习多模态类别特定 tocken 来指导密集目标定位**，以提高定位的准确性。通过结合视觉和文本信息，以及样本特定的上下文信息，该方法能够更好地感知目标的**类内差异**，从而实现更准确的密集定位。

具体而言，作者提出了一个统一的 transformer 框架来学习**类别特定** tocken 的两种模式，即类特定的**视觉 tocken** 和**文本 tocken** 。前者从目标视觉数据中捕获语义信息，而后者利用 CLIP 中的类相关语言先验，提供互补信息来更好地感知类内的多样性。

此外，还利用包含视觉上下文和图像语言上下文的**样本特定**上下文来丰富多模态的类特定标记。

![image-20230718232308461](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230718232308461.png)

> 方法：

![image-20230718233424572](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230718233424572.png)

Encoder 的输入为：C 个 class tokens + N^2 个 图像 patches + C 个 text tokens（C为类别总数）

为了允许 class tokens 和 text tokens 分别都与 patches 完全交互，但 class tokens 和 text tokens 之间不交互，因此加上了掩码机制，如图：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230718233802636.png" alt="image-20230718233802636" style="zoom:67%;" /> ，得到的 class tokens 输出以 channel 维度取平均作为分类分数；

受上下文感知提示]的启发，作者还使用视觉上下文来细化文本嵌入。<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230718235211851.png" alt="image-20230718235211851" style="zoom:80%;" />

为了进一步确保输出 text tokens 能够捕获有意义的特定于样本的上下文，通过利用预先训练的 CLIP 图像模型，对输出文本令牌施加对比损失：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230718235438462.png" alt="image-20230718235438462" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230718235336306.png" alt="image-20230718235336306" style="zoom:80%;" />

> 总结：

* 本文从 class-tockens 以预训练的 ViT 的 [CLS] tocken 初始化进行学习，text-tockens 就固定为 CLIP 对应的文本特征；
* 本文中的 text tockens，与 patches 交互之后得到**与图像相关的 text tockens 输出**再与 各自对应的图像做对齐（事后对齐），以确保 text tockens 输出是与本图像高度相关的；
* 事后对齐的思路可以参考借鉴，虽然暂时还没有想到怎么用，但插个眼。（联系文献59的双向约束对齐）

### 90_Prototypical Residual Networks for Anomaly Detection and Localization_CVPR 2023_无代码

> 作者：Hui Zhang1,2 Zuxuan Wu1,2 Zheng Wang3 Zhineng Chen1,Yu-Gang Jiang

> 贡献：

背景：异常检测和定位以其效率和有效性在工业制造中广泛应用于工业制造。异常是罕见的，很难收集，通过监督模型容易过合所收集到的少数异常样本。另一方面，异常通常是微妙的且外观多样，难以辨别，这使得检测都很难，更不用说定位了。

为了解决这些问题，本文提出了一个名为**原型残差网络（PRN，Prototypical Residual Network）**的框架，通过学习异常模式和正常模式之间不同尺度和大小的特征残差，以准确地重建异常区域的分割图。PRN 主要由两部分组成：**多尺度原型**明确地表示异常到正常模式的残余特征；**多尺寸的自我注意机制**学习可变大小的异常特征。此外，作者还提出了各种异常生成策略，考虑可见和不可见的外观异常，以扩大和多样化异常。

> 方法：

首先，将图像输入基于 ImageNet 预训练的网络进行聚类学习多尺度原型，学好的原型在后续训练过程中保持不变；

训练时，对于输入的一张图像，根据得到的特征图和对应原型获得多尺度异常残差表示：![image-20230717160032486](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717160032486.png)

为了实现跨尺度表示的信息交换，执行多尺度融合<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717160234171.png" alt="image-20230717160234171" style="zoom:80%;" />

由于异常区域的大小变化较大，为了进一步检测融合后的特征图中的局部不一致，引入了一种多尺寸自注意（MSA）机制。如下图，将融合后的特征图按不同比例划分成 patch，分别进行自注意力后通过再进行融合。

![image-20230717153906321](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717153906321.png)

> 总结：

从不同尺度特征的角度分别维护原型的方法比较新颖，但目前没有什么可以借鉴的思路。

### 91_Retentive Network: A Successor to Transformer for Large Language Models__有代码

> 作者：Yutao Sun∗ †‡ Li Dong∗ † Shaohan Huang† Shuming Ma† Yuqing Xia† Jilong Xue† Jianyong Wang‡ Furu Wei†⋄

> 代码：https://github.com/microsoft/unilm/tree/master/retnet#

> 贡献：

背景：“不可能三角问题”：Transformer 的并行处理机制以低效推理为代价，而且是内存密集型模型，序列越长，占用的内存越多；线性 Attention 可以降低推理成本，但性能较差；循环神经网络则无法进行并行训练。

![image-20230720224245892](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720224245892.png)

本文从序列建模的角度，构建了一种类似Transformer且更加高效的结构。在语言任务上展现出了良好的效率和性能。

* 利用类似于Transformer的并行组件实现了对于GPU并行能力的利用；
* 利用循环机制确保了O ( 1 ) O(1)O(1)级别的存储和计算复杂度；
* 利用分块循环策略执行有效的长序列建模。

> 方法：

具体而言，RetNet 在 Transformer 的基础上，使用**多尺度保持（Retention）机制**替代了标准的自注意力机制。与标准自注意力机制相比，保持机制有以下特点：

* 引入位置相关的指数衰减项取代 softmax，简化计算的同时使前步的信息以衰减的形式保留下来；
* 引入复数空间表达位置信息，取代绝对或相对位置编码，容易转换为递归形式；
* 保持机制使用多尺度的衰减率，增加模型的表达能力，并利用 GroupNorm 的缩放不变性来提高 Retention 层的数值精度。

![image-20230720224800753](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720224800753.png)

每个 RetNet 块包含两个模块：**多尺度保持（MSR）模块**和**前馈网络（FFN）模块**

保持机制支持三种形式表示序列：

* 并行，并行表示使 RetNet 可以像 Transformer 一样**高效**地利用 GPU 进行并行训练；
* 递归，递归表示实现了O(1)的推理复杂度，降低了内存占用和延迟；
* 分块递归（即并行表示和递归表示的混合形式，将输入序列划分为块，在块内按照并行表示进行计算，在块间遵循递归表示），分块递归则可以更高效地处理长序列。

### 92_TOWARDS ROBUST OBJECT DETECTION INVARIANT TO REAL-WORLD DOMAIN SHIFTS_ICLR 2023_无代码

> 作者：Qi Fan1, Mattia Segu2,3, Yu-Wing Tai1, Fisher Yu2, Chi-Keung Tang1,Bernt Schiele3, Dengxin Dai3

> 贡献：

背景：现有的分类域泛化（DG）方法不能有效地解决目标检测问题，因为它们要么依赖于多个风格差异大的源域，要么破坏了原始图像的内容结构。

由于作者观察到目标域图像的特征通道统计量偏离了源域统计数据，因此提出 Normalization Perturbation (**NP**) 即插即用模块，通过**扰动源域 low-level 的特征**来合成多种潜在风格，即使训练时不接触 target 域的情况下也能让模型感知到多种潜在域并具有很好的泛化能力。

如下图，(a). 不同域的 stage 1 输出特征存在差异

(b). 本文方法，相较于 baseline，在 stage1 、stage5 ，源域和目标域的特征混合的更好（更难分开，说明模型对目标域泛化能力好）

![image-20230720231132617](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720231132617.png)

> 方法：

由于特征通道统计量，如均值和标准差，与图像样式密切相关，改变特征通道统计量可以视为隐式地改变输入图像样式。NP 则通过在 low-level 特征上对特征进行处理，其中 $\mu$ 为统计量，$\alpha ,\beta$ 为随机噪声。

![image-20230720232550897](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720232550897.png)

![image-20230720232847245](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720232847245.png)

![image-20230720233049935](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230720233049935.png)

> 总结：

* 通过对比不同域低层特征的统计量（均值）对比，作者发现，不同域的低层特征统计量存在差异；
* 对低层特征根据其通道统计量加噪能提高模型对域泛化的能力；

* 能否以类似的方式给已知类特征加噪，然后提高模型对未知类的泛化能力？

### 93_THALAMUS: A BRAIN-INSPIRED ALGORITHM FOR BIOLOGICALLY-PLAUSIBLE CONTINUAL LEARNING AND DISENTANGLED REPRESENTATIONS _ ICLR 2023_有代码

> 作者：Ali Hummo

> 贡献：

背景: 动物在不断变化的环境中茁壮成长，并利用时间结构来学习因果关系的分解表示。

传统的神经网络在序列经验中容易遗忘，并且在旧学习和新学习之间存在干扰，限制了大多数训练范式只能使用打乱的数据。许多最近的方法提高了神经网络的灵活性，但除了减轻遗忘之外，最近提出了一些理想的持续学习代理的可取特性，包括：在学习周期结束时或至少在最小额外训练下快速适应和恢复准确性的多个任务上的准确性。理想的代理还应该能够向前传递知识，适应未来任务和以前学习的任务，同时还能传递给具有稍微不同计算或输入输出分布的任务。该算法应该能够随着任务数量的增加而扩展，并保持进一步学习的能力。最后，代理理想情况下应该能够无监督地工作，而不依赖于任务标签和任务边界。

本研究受到大脑丘脑皮层回路的启发，提出了一种简单的算法，通过推断过程中的优化来动态生成当前任务的内部表示。该算法通过交替更新模型权重和潜在任务嵌入，使代理能够将时间经验流解析为离散事件并组织对其的学习。

![image-20230717105229569](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717105229569.png)

> 方法：

每个任务由输入 $x^{k}$ 、输出 $y^{k}$ 和任务标识符 $i^{k}$ 的数据集 $D^{k}$ 描述；

学习代理过程中，预测<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717105021437.png" alt="image-20230717105021437"  />；

计算损失：![image-20230717105041742](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717105041742.png)

交替更新模型权重和潜在嵌入：![image-20230717105148999](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717105148999.png)

将模型权重更新和表示任务 id 的潜在嵌入更新结合在一个简单的算法中，该算法可以恢复环境中的潜在变量，因为它遵循数据分布的动态。从一个普通的 RNN 开始，试图解决流入的未标记任务。这个网络是新初始化的，之前没有接受过任何任务上的训练。它通过权重更新来学习第一个任务，当平均精度开始增加时，运行平均值突然下降超过指定数量（例如0.1）都会触发切换到潜在更新；在潜在空间中采取一定数量的梯度下降步骤。如果平均精度没有恢复，那么算法将通过权重更新返回学习。更直观地说，一旦当前的潜在嵌入不再解决当前的任务，就使用潜在的更新来到达另一个任务。

![image-20230717104632431](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230717104632431.png)

### 94_Adaptive Plasticity Improvement for Continual Learning_CVPR 2023_无代码

> 作者：Yan-Shuo Liang and Wu-Jun Li*

> 贡献：

背景：许多研究都试图解决持续学习中的灾难性遗忘（CF）问题。然而，在旧任务上追求不遗忘可能会损害模型对新任务的可塑性。虽然已经提出了一些方法来实现稳定性-可塑性的权衡，但**没有一种方法考虑评估模型的可塑性和提高可塑性**。

在这项工作中，作者提出了一种新的方法，称为**自适应可塑性改进（API）**。除了能够克服对旧任务的 CF 外，API 还试图评估模型的可塑性，然后在必要时自适应地提高模型的可塑性，以便学习新任务。

> 方法：

如下图，以一个简单的三层神经网络为例解释 API：除了最后一层，其他层都根据相应需要来决定是否增加输入维度，并增加相应的权重参数。

对于每个 task t，API 在客服遗忘的基础上，首先评估模型的可塑性，然后根据评估结果考虑是否增加输入维度，即扩张权重参数，如果可塑性足够，那么就不需要扩张权重。

![image-20230721100143169](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721100143169.png)

API 采用梯度修正策略 GPM 来克服 CF。基于该策略的方法修正了新的任务梯度，使其不影响模型在旧任务上的表现。但由于 GPM 存在一个问题，即内存占用会一直增加，因此作者使用 **DualGPM** 来解决内存占用一直增加的问题：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721102803997.png" alt="image-20230721102803997" style="zoom:80%;" />

![image-20230721103227991](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721103227991.png)

**Plasticity Evaluation**

gradient retention ratio (GRR)：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721102903418.png" alt="image-20230721102903418" style="zoom:80%;" />

扩张标准：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721103019037.png" alt="image-20230721103019037" style="zoom:80%;" />

![image-20230721103310368](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721103310368.png)

> 总结：

* 从梯度的角度去考虑问题，在克服遗忘的前提下同时保持模型持续学习过程中的可塑性；
* 不知道应用到自己的任务上，这种从梯度去考虑的方法能否比样例回放更好呢；

### 95_AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning_CVPR 2023_有代码

> 作者：Runqi Wang1,2∗, Xiaoyue Duan1*, Guoliang Kang1,4, Jianzhuang Liu2,Shaohui Lin3, Songcen Xu2, Jinhu Lv1,4, Baochang Zhang1,4†

> 代码：https://gitee.com/mindspore/models/tree/master/research/

> 贡献：

背景：持续学习的目的是使一个模型能够从按顺序到达的数据中逐步学习知识。以往的工作采用了传统的分类架构，由一个特征提取器和一个分类器组成。特征提取器在按顺序到达的任务或类之间共享，但是与新类对应的分类器的特定权重组应该逐步扩展，因此**参数逐渐增加**。此外，由于分类器包含所有历史到达的类，通常需要一定大小的内存来存储回放数据，以减轻分类器偏差和灾难性遗忘。

本文则提出了一个非持续增加的学习器：**AttriCLIP**逐步提取新类或任务的知识。AttriCLIP 建立在预训练过的视觉语言模型 CLIP 之上，保持图像编码器和文本编码器固定，以从图像和文本中提取特征。文本由**类别名称**和一个**固定数量的可学习参数**组成，这些参数从**属性词库**中选择并作为属性。因为是根据视觉和文本相似度进行分类，所以 AttriCLIP 是一个非增量学习器。**属性提示，即对分类有用的常识进行编码，可以有效地减轻灾难性遗忘，避免构建重放记忆**。

![image-20230721105730010](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721105730010.png)

> 方法：

![image-20230721105923697](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721105923697.png)

> 总结：

* 构建属性 prompt
* 根据图片的指导学习属性并作为prompt的组成部分
* 属性 prompt 如果能学好，对于挖掘未知类特征也是有帮助的

### 96_Exploring Data Geometry for Continual Learning_CVPR 2023_无代码

> 作者：Zhi Gao1, Chen Xu2*, Feng Li, Yunde Jia2,1, Mehrtash Harandi3, Yuwei Wu1,2*

> 贡献：

背景: 持续学习旨在从非稳态数据流中高效学习，同时避免遗忘旧数据的知识。然而，现有方法很少研究数据的几何结构对持续学习的影响。过去的持续学习方法通常假设数据是欧几里得的，并使用欧几里得几何来处理数据流。然而，实际应用中的数据往往具有非欧几里得的几何结构，导致使用欧几里得几何会产生不理想的结果。

鉴于现有方法对数据几何的研究较少，本文从**数据几何**的角度出发，探索了持续学习的新视角，提出了一种称为几**何增量搜索方案（GIS）**的连续学习方法。通过动态扩展底层空间的几何结构，以适应新数据引起的几何结构变化，并通过**角度正则化损失**和**邻居鲁棒性损失**来训练模型，从而实现了对旧数据几何结构的保留。

> 方法：

GIS 动态增加混合曲率空间中的常曲率空间的数量，以建模更复杂的结构。该方法涉及使用常曲率空间（CCS）和混合曲率空间。CCS 是具有恒定曲率的平滑黎曼流形，可以是双曲空间、欧几里得空间或超球面空间。混合曲率空间是多个 CCS 的笛卡尔积。该方法使用指数映射和对数映射在 CCS 和其切空间之间进行转换。混合曲率空间配备了诱导距离和角度的度量。该方法还包括使用投影到混合曲率空间上的分类器的基于距离的分类方法。通过使用交叉熵损失进行分类以及角度正则化损失和邻居鲁棒性损失来维持几何结构，更新 backbone 和分类器。

![image-20230721115600208](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721115600208.png)

![image-20230721120229600](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721120229600.png)

### 97_Improving Image Recognition by Retrieving from Web-Scale Image-Text Data_CVPR 2023_无代码

> 作者：Ahmet Iscen Alireza Fathi Cordelia Schmid

> 贡献：

背景：检索增强模型在自然语言处理问题上取得成功后，在计算机视觉任务中越来越受欢迎。其目标是**通过从外部记忆集检索视觉输入的类似例子来增强模型的识别能力**。

作者引入了一个**基于注意力的记忆模块**，它从记忆中学习每个检索到的例子的重要性。与现有的方法相比，本文的方法**消除了不相关的检索示例的影响**，并**保留了那些对输入查询有益的示例**。

![image-20230721122141352](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721122141352.png)

> 方法：

通过将图像-文本对输入相应的编码器获得的特征表示作为 memory 中的 key-value 对；

对如输入的一张图，首先通过视觉编码器获得特征表示作为 query embedding，并以此从 memory 中找到最近的 k 个 key-value 对；

然后将 query 和 knn 的 key-value 对送入记忆注意力模块，区分不同 key-value 对的重要程度；

最后将 refined embedding 送入分类器进行分类。

![image-20230721121727977](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721121727977.png)

> 总结：

* memory 的构建非常直白，直接将图文对编码获得 key-value 对；
* memory attention module 做的工作可以理解为使用相关记忆对现有特征表示的增强。

### 98_Position-guided Text Prompt for Vision-Language Pre-training_CVPR 2023_有代码

> 作者：Jinpeng Wang2 Pan Zhou1* Mike Zheng Shou2* Shuicheng Yan1

> 代码：https://github.com/sail-sg/ptp

> 贡献：

背景：作者观察到VLP模型往往缺乏视觉定位能力，这对于许多下游任务如视觉推理至关重要。

本文提出了一种新的位置引导文本提示（**PTP**，Position-guided Text Prompt）范式，增强使用 VLP 训练的跨模态模型的视觉定位能力。具体来说，在 VLP 阶段，PTP 将图像分割为 N × N 个块，并通过 VLP 中广泛使用的对象检测器来识别每个块中的对象。然后通过鼓励模型预测给定块中的对象或给定对象找到对应的块，即填充 *“The block [**P**] has a [**O**]” 中的 “[**P**]” 或 “[**O**]”。

> 方法：

首先使用预训练的 Faster-RCNN 检测出图像中的 top k 个对象，并将图像分成 N × N 个块，若检测到的对象中心落在某个块中，则构建 prompt：The block [**P**] has a [**O**]，对于某个块中可能包含多个对象的情况，则随机选择一个 O。

然后文本端的输入就变成了 caption + 生成的 prompt。

![image-20230721143442719](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721143442719.png)

> 总结：

* 加先验信息；

### 99_Open-set Fine-grained Retrieval via Prompting Vision-Language Evaluator_CVPR 2023_无代码

> 作者：Shijie Wang1, Jianlong Chang2, Haojie Li1,3*, Zhihui Wang1, Wanli Ouyang4, Qi Tian2

> 贡献：

背景：开放集细粒度检索需要在评估过程中使用额外的能力来检索未知的子类别。然而，目前的工作集中在视觉概念上，其中所有的子类别都是预定义的，这使得很难从未知子类别中获取有区别的知识，因此无法处理开放世界场景中的未知子类别。

在这项工作中，本文提出了一种新的提示视觉语言评估（**PLEor**）框架，基于最近引入的对比语言图像预训练（CLIP）模型，用于开放集细粒度检索。PLEor 利用预训练的 CLIP 模型来推断包含预先定义和未知子类别的差异，即**类别特定差异**，并将它们转移到在闭集场景中训练的主干网络中。为了使预训练的 CLI P模型对特定类别的差异敏感，作者设计了一个**双提示**方案来学习特定类别差异的视觉提示，并将文本提示中带有类别名称的随机向量转换为特定类别的差异描述。此外，还提出了一种基于 CLIP 模型对视觉和文本提示进行**语义对齐，并相互强化**。最后，提出了一种开放集的知识转移方法，利用**知识蒸馏**机制将特定类别的差异转移到主干网络中。

> 方法：

本文提出的 PLEor 方法由四个模块组成：**检索模块**、**双提示方案模块**、**视觉-语言评估模块**和**开放集知识传递模块**。

检索模块使用骨干网络提取图像表示并生成检索嵌入；

双提示方案模块修改预训练的CLIP模型的输入，使其对类别特定的差异敏感；

视觉-语言评估器模块在训练过程中使用预训练的CLIP模型作为评估器；

开放集知识传递模块将差异传递给在闭集场景中训练的模型。该模块旨在将图像投影到嵌入空间，并在开放集场景下提高检索性能。

整体训练目标将这些组件以不同的权重结合起来：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721152936750.png" alt="image-20230721152936750" style="zoom:80%;" />

![image-20230721145816132](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721145816132.png)

### 100_Region-Aware Pretraining for Open-Vocabulary Object Detection with Vision Transformers_CVPR 2023_无代码

> 作者：Dahun Kim Anelia Angelova Weicheng Kuo

> 贡献：

背景: 开放词汇检测任务通过利用丰富的图像-文本对进行预训练，在测试时使用用户提供的文本查询来预测未见过的对象。过去的方法主要集中在图像级别的任务，如分类和检索，而本文旨在增强区域级别的下游任务，即开放词汇目标检测。现有的预训练模型往往没有充分利用图像中的对象/区域信息，因此需要一种新的方法来在图像-文本预训练中融入区域信息。

本文提出了**RO-ViT**，通过区域感知的预训练方式，为开放词汇目标检测的视觉 transformer 提供更好的性能。通过在预训练阶段使用裁剪的位置嵌入，与检测微调阶段的区域裁剪相匹配，此外，还使用 focal loss 替代 softmax 交叉熵损失，以更好地学习有信息但难以处理的示例。最后，利用 novel 目标提议方法改进开放词汇检测的微调过程。

> 方法：

由于开放词汇表检测的识别发生在区域级，这需要全图像位置嵌入泛化到他们在预训练过程中从未见过的区域。为了弥补这一差距，我们提出了裁剪位置嵌入（**CPE**），首先对预训练中整张图的位置嵌入进行上采样，然后从上采样的结果中随机裁剪一个区域，并将裁剪的结果上采样到整张图片的位置嵌入的大小。所裁剪的区域坐标从正态分布中采样得到。直观地说，这使得模型查看的图像本身不是一个完整的图像，而是从一些更大的未知图像中裁剪出的区域。这更好地匹配了检测的下游任务，即识别发生在区域-而不是图像级别。

![image-20230721170524874](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721170524874.png)

> 总结：

* 这种随机性裁剪虽然说能够一定程度上缓解 image-level 到 region-level 的 gap，但感觉随机可能不是最优的方法吧，或许可以考虑加点先验信息之类的。

### 101_Multi-level Logit Distillation_CVPR 2023_有代码

> 作者：Ying Jin1  Jiaqi Wang2  Dahua Lin1,2

> 代码：https://github.com/Jin-Ying/MultiLevel-Logit-Distillation

> 贡献：

背景:知识蒸馏方法可以分为两类：逻辑（logit）蒸馏和特征蒸馏。逻辑蒸馏方法只利用逻辑输出进行知识蒸馏，而特征蒸馏方法则在中间层次上进行蒸馏，以匹配特征分布和逻辑输出。相较于逻辑蒸馏，特征蒸馏方法性能更好，但在某些实际应用中存在隐私和安全等问题。
针对这一困境，本文提出了一种逻辑蒸馏方法，通过更好地利用逻辑输出，实现了多层次的预测对齐。预测对齐不仅在**实例级别**进行，而且在**批处理**和**类别级**进行，通过这些级别，学生模型同时学习实例预测、输入相关性和类别相关性。

> 方法：

预测增强机制，通过使用温度缩放进行模型校准，将单个输出扩展为多个输出。温度缩放方法用于将输入在每个类别上的概率值转换为一组新的概率值。通过使用一组温度，可以将一个预测增强为K个不同的输出。
多级对齐方法，包括实例级、批次级和类别级对齐。实例级对齐通过最小化教师和学生模型的增强预测之间的KL散度来实现。批次级对齐通过量化输入相关性使用logit预测并计算Gram矩阵来实现。类别级对齐通过使用一批数据的预测来建模类别相关性来实现。
多级对齐损失是实例级、批次级和类别级对齐损失的组合。学生模型在实例级预测、批次级输入相关性和类别级类别相关性方面都被训练成模仿教师模型。

![image-20230721183514932](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721183514932.png)

![image-20230721183623448](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721183623448.png)

### 102_BiCro: Noisy Correspondence Rectification for Multi-modality Data via Bi-directional Cross-modal Similarity Consistency_CVPR 2023_有代码

> 作者：Shuo Yang1 Zhaopan Xu2 Kai Wang3 Yang You3 Hongxun Yao2 Tongliang Liu4 Min Xu1*

> 代码：https://github.com/xu5zhao/BiCro

> 贡献：

背景: 多模态数据集的收集和准确标注非常困难，因此通常采用从互联网收集的图像-文本对作为替代。然而，这种廉价收集的数据集不可避免地包含许多不匹配的数据对，这对模型的性能有害。

因此，本文旨在解决这个问题，提出了一种**基于双向跨模态相似性一致性**的通用框架 **BiCro**，通过估计噪声数据对的软标签来反映它们的真实对应程度，从而提高跨模态匹配模型对噪声数据的鲁棒性。 以往的噪声鲁棒学习方法主要针对分类任务中的噪声标签，而不适用于跨模态匹配问题中的对应错误。现有的跨模态匹配方法往往无法直接应用于噪声数据对的准确软对应标签估计。

> 方法：

该框架的动机是**相似的图像应该具有相似的文本描述**，反之亦然。因此作者提出使用 beta 混合模型（BMM）来近似混合了干净样本和噪声样本的损失分布。BMM 由具有参数 γ 和 β 的 beta 分布定义。使用期望最大化（EM）过程将 BMM 拟合到观察到的每个样本损失值分布上。根据拟合的BMM计算数据对是干净还是噪声的概率。根据阈值 δ 从噪声数据集中选择锚点，其余的数据被认为可能是噪声的，并且它们的标签被丢弃。干净数据和带有**估计软标签**的噪声数据用于模型训练。

![image-20230721184346620](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721184346620.png)

![image-20230721184234099](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230721184234099.png)

> 总结：

* 根据相似的图片应该有相似的标签来为噪声图片创造软标签。

### 总结：

* **原型（特征学习）& 伪标签**

Adversarial Reciprocal Points Learning for Open Set Recognition中的**互倒点，也算是另一种原型**。比较有意思，不直接从什么样的东西是猫去判别猫，而是从什么东西不是猫去排除不是猫的对象；Glocal Energy-based Learning for Few-Shot Open-Set Recognition将**自注意力用于特征增强的作用用在了原型上**，这是个可以借鉴的点；PROTOCON: Pseudo-label Refinement via Online Clustering and Prototypical Consistency for Efficient Semi-supervised Learning中使用了伪标签优化，利用原型和聚类对伪标签进一步精细化，**伪标签优化**的思想可以加到OWOD中，考虑使用别的先验信息来校准伪标签；

* **prompt**

AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning**根据图片的指导**学习属性并作为prompt的组成部分；Improving Image Recognition by Retrieving from Web-Scale Image-Text Data**直接将图文对编码的特征作为 key-value 对**来构建memory bank；PromptCAL: Contrastive Affinity Learning via Auxiliary Prompts for Generalized Novel Category Discovery**使用 teacher 的输出构建memory bank**，提供用以构建偏好子图的内容，用于伪标签；

* **其他**

Random Boxes Are Open-world Object Detectors中的 proposals **动态k匹配方式**似乎很能提点，可以考虑用一下（和之前看的文献中的对 proposals 进行软标签对比）；Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization 中的 text tockens，与 patches 交互之后得到**与图像相关的 text tockens 输出**再与各自对应的图像做对齐（事后对齐），以确保 text tockens 输出是与本图像高度相关的，事后对齐的思路可以参考借鉴；TOWARDS ROBUST OBJECT DETECTION INVARIANT TO REAL-WORLD DOMAIN SHIFTS* 通过对比不同域低层特征的统计量（均值）对比，作者发现，不同域的低层特征统计量存在差异； **对低层特征根据其通道统计量加噪能提高模型对域泛化的能力**；考虑能否以类似的方式给已知类特征加噪，然后提高模型对未知类的泛化能力；

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

作为参考，尝试 HungarianMatcher 时，将损失项 loss_clip 设置为 0 ：

实验结果，模型没有检测 unknown 的能力，**可见这项损失的必要性**。

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

↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑一次意外的超参错误实验结果↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

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

#### 不叠任何 buff 的 Featurized Q-RCNN 增量性能（作为对比）

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

修改：让 loss_close 和 loss_dis 均从夹角（cosine_similarity）和距离（l2）的角度考虑（loss_dis越训练越大，没办法减小不同类原型的距离和相似性）

修改：去掉loss_dis，loss_close = l2 + 1-sim

| 去掉loss_dis | t1      | t2      | t3      | t4      |
| ------------ | ------- | ------- | ------- | ------- |
| Prev mAP(↑)  |         | 44.8516 | 32.2227 | 26.6415 |
| Cur mAP(↑)   | 57.2129 | 30.7961 | 19.4214 | 15.1947 |
| Both mAP(↑)  |         | 37.8238 | 27.9556 | 23.7798 |
| WI(↓)        | 0.0660  | 0.0369  | 0.0221  |         |
| A-OSE(↓)     | 32651   | 19281   | 12626   |         |
| U-Recall(↑)  | 12.6672 | 12.6133 | 11.5066 |         |

存在的问题：学出来的不同类之间的原型 cosine 相似性非常高，0.9 左右，几乎为 1，l2 距离存在，说明使用 l2 损失来拉近特征和原型之间的距离被模型钻了空子。

修改：使用文献 20 的方法来学习紧凑又分散的特征（最小化负对数损失：$L_{comp} = − \frac{1}{N} \sum_{i=1}^{N}log\frac { exp ( z^{T}_{i} \mu_{c(i)} /τ)}{\sum_{j=1}^{C} exp (z^{T}_{i} \mu_{j} /τ)}$，它鼓励样本与它对应的类原型靠的更近；最大化不同类原型之间的距离：$L_{dis} = \frac{1}{C} \sum_{i=1}^{C} log\frac{1}{C-1} \sum_{j=1}^{C} 1\{j \neq i\} e^{\mu_{i}^{T} \mu_{j} / \tau}$，暂时加损失的比例为 $L_{comp} + L_{dis}$ ）

| $L_{comp}+ L_{dis}$ | t1      | t1(0.1->uk) | t2      | t2(0.1->uk) | t3      | t3(0.1->uk) | t4      |
| ------------------- | ------- | ----------- | ------- | ----------- | ------- | ----------- | ------- |
| Prev mAP(↑)         |         |             | 46.6980 | 45.0248     | 35.3456 | 34.5140     | 28.9764 |
| Cur mAP(↑)          | 57.0662 | 56.2815     | 31.4185 | 30.6911     | 20.6794 | 20.3172     | 16.1459 |
| Both mAP(↑)         |         |             | 39.0582 | 37.8579     | 30.4569 | 29.7817     | 25.7688 |
| WI(↓)               | 0.0576  | 0.0483      | 0.0279  | 0.0176      | 0.0187  | 0.0128      |         |
| A-OSE(↓)            | 32493   | 5021        | 16734   | 2135        | 11180   | 2553        |         |
| U-Recall(↑)         | 8.6535  | 20.8662     | 9.0708  | 16.5314     | 5.2908  | 18.7116     |         |

下一步计划，调整一下超参（loss的权重比例）跑跑看效果；再试试用原型校正已知类mAP。

## 20230616

降低 $ L_{dis}$ 的比例不太行（0.2、0.5效果都较差）权重为0.7时结果如下

| $L_{comp}+ 0.7 L_{dis}$ | t1      | t2      | t3      | t4      |
| ----------------------- | ------- | ------- | ------- | ------- |
| Prev mAP(↑)             |         | 47.2058 | 35.8315 | 28.8379 |
| Cur mAP(↑)              | 57.1332 | 31.3412 | 20.1931 | 15.6293 |
| Both mAP(↑)             |         | 39.2735 | 30.6187 | 25.5357 |
| WI(↓)                   | 0.0599  | 0.0302  | 0.0187  |         |
| A-OSE(↓)                | 33486   | 17474   | 11194   |         |
| U-Recall(↑)             | 9.6012  | 6.1843  | 5.8931  |         |

校准方式1：（pass）

预测时，先计算输出feature与原型的相似性，与已知类相似性介于0.3和0.5之间的将相应未知类位置上的值设为0.99，然后经过sigmoid计算概率的结果如下（对已知类精度影响太大），t1的结果如下：

Current class AP50: 52.067338461774895

WI：0.0539

Absolute OSE：20875.0

Unknown Recall50: 18.554888507718697

#### 校准方式2：

预测时，将特征与已知类原型的相似性作为附加值加到模型输出的 output_cls，然后经过 sigmoid 的logits，t1的结果如下：

​																								（右边为进一步将预测为已知类但 logits 小于0.1的预测为unknown）

​																									阈值0.1                      阈值0.05

Current class AP50: **57.4708**43607692316  #                      57.1571                      57.4019

WI：0.0588                                                        #                      0.0496                        0.0550

Absolute OSE：33257                                      #                      7754                           14170.0

Unknown Recall50: 8.936535162950257      #                      19.3653                     16.9854     

| $L_{comp}+ 0.7 L_{dis}$ | t1      | t1(0.1->uk) | t2      | t2(0.1->uk) | t3      | t3(0.1->uk) | t4      |         |
| ----------------------- | ------- | ----------- | ------- | ----------- | ------- | ----------- | ------- | ------- |
| Prev mAP(↑)             |         |             | 47.2058 | 46.5586     | 35.8315 | 35.7917     | 28.8379 | 28.9014 |
| Cur mAP(↑)              | 57.1332 | 57.1571     | 31.3412 | 31.5158     | 20.1931 | 20.1173     | 15.6293 | 15.6854 |
| Both mAP(↑)             |         |             | 39.2735 | 39.0372     | 30.6187 | 30.5669     | 25.5357 | 25.5974 |
| WI(↓)                   | 0.0599  | 0.0496      | 0.0302  | 0.0199      | 0.0187  | 0.0133      |         |         |
| A-OSE(↓)                | 33486   | 7754        | 17474   | 4042        | 11194   | 4375        |         |         |
| U-Recall(↑)             | 9.6012  | 19.3653     | 6.1843  | 14.4859     | 5.8931  | 14.6037     |         |         |

结果分析：

使用预测特征和原型的相似度来校准类别预测对已知类预测略微有一点作用

预测时通过设置阈值排除掉误分类为已知类的未知类，对未知类召回率比较有帮助

存在的问题：

感觉框架可以重新考虑一下，匹配之后再做特征的对齐貌似只是让特征学的更具有区分度一些，但是没有帮助到分类头更精确地分类。

## 20230623

受文献39（TransHP: Image Classification with Hierarchical Prompting）启发，试图通过给 query 加点信息帮助它快速找到要检测的对象。

方法：在 query 和 roi feature 交互之前，以 query 和 原型的相似度作为权重，将原型加到 query 上，这一步称为pre-attn，然后将得到的新query按照原计划进行后续交互操作。

结果：反向优化了，无论是对已知类还是未知类的检测能力都下降了。（pre-attn的操作无论是放在第一次交互之前还是第二次交互之前，性能都会下降。。。）

后续计划：考虑采用拼接的方式给 query 加提示。

## 20230630

实验计划：

1. 使用 caption 信息辅助检测 unknown：随机选取带有 caption 信息的图片中的某一条 caption，提取出其中的名词并获得相应文本特征，已知类相关特征用于在语义特征空间对齐跨膜态信息；其他文本特征用于辅助检测 unknown 类。（为了减少模型训练耗时，文本信息采用线下提取的方式）
2. 对 roi 特征进行增强：方案一，做自注意力；方案二，与文本信息做交叉注意力；或者两者结合。

目前进度，已经将 caption 信息加入到注释文件中，caption 数据差不多整好了；

下一步：修改网络模块，将上述想法融进去。

## 20230714

由于要用到COCO Caption信息，因此在 OW-DETR 提出的 **MS-COCO split** 划分方式的数据集上进行实验：

（注：ORE-EBUI 和 OW-DETR 的数据来自文章 OW-DETR）

| 行号 | t1                                  | Cur-mAP(↑) | WI(↓)  | A-OSE(↓) | U-Recall(↑) |
| ---- | ----------------------------------- | ---------- | ------ | -------- | ----------- |
| 1    | ORE-EBUI                            | 61.4       |        |          | 1.5         |
| 2    | OW-DETR                             | 71.5       |        |          | 5.7         |
| 3    | FQ+Caps(1+0.7)                      | 57.5855    | 0.0575 | 11470    | 21.4515     |
| 4    | FQ（增加epoch）                     | 63.4851    |        |          |             |
| 5    | FQ+Caps(1+0.7)（增加epoch, 214161） | 61.9299    | 0.0519 | 10924    | 16.0065     |
| 6    | FQ+Caps(1+0.7)（增加epoch, 81）     | 62.2226    | 0.0524 | 11356    | 16.4613     |



| 行号 | t2                                  | Pre-mAP(↑) | Cur-mAP(↑) | Both-mAP(↑) | WI(↓)  | A-OSE(↓) | U-Recall(↑) |
| ---- | ----------------------------------- | ---------- | ---------- | ----------- | ------ | -------- | ----------- |
| 1    | ORE-EBUI                            | 56.5       | 26.1       | 40.6        |        |          | 3.9         |
| 2    | OW-DETR                             | 62.8       | 27.5       | 43.8        |        |          | 6.2         |
| 3    | FQ+Caps(1+0.7)                      | 42.3397    | 43.7737    | 43.0567     | 0.0369 | 6074     | 26.5255     |
| 4    | FQ（增加epoch）                     | 54.5814    | 46.5594    | 50.5704     |        |          |             |
| 5    | FQ+Caps(1+0.7)（增加epoch, 214161） | 49.8437    | 45.0574    | 47.4506     | 0.0360 | 7763     | 19.5398     |
| 6    | FQ+Caps(1+0.7)（增加epoch, 81）     | 53.7340    | 44.9092    | 49.3216     | 0.0330 | 6525     | 20.7823     |



| 行号 | t3                                  | Pre-mAP(↑) | Cur-mAP(↑) | Both-mAP(↑) | WI(↓)  | A-OSE(↓) | U-Recall(↑) |
| ---- | ----------------------------------- | ---------- | ---------- | ----------- | ------ | -------- | ----------- |
| 1    | ORE-EBUI                            | 38.7       | 23.7       | 33.7        |        |          | 3.6         |
| 2    | OW-DETR                             | 45.2       | 24.9       | 38.5        |        |          | 6.9         |
| 3    | FQ+Caps(1+0.7)                      | 38.9024    | 38.2643    | 38.6897     | 0.0282 | 5906     | 22.6527     |
| 4    | FQ（增加epoch）                     | 49.3461    | 40.4426    | 46.3783     |        |          |             |
| 5    | FQ+Caps(1+0.7)（增加epoch, 214161） | 42.3166    | 37.4080    | 40.6804     | 0.0279 | 7427     | 11.7237     |
| 6    | FQ+Caps(1+0.7)（增加epoch, 81）     | 46.8364    | 38.5558    | 44.0762     | 0.0228 | 5499     | 14.6547     |



| 行号 | t4                                  | Pre-mAP(↑) | Cur-mAP(↑) | Both-mAP(↑) |
| ---- | ----------------------------------- | ---------- | ---------- | ----------- |
| 1    | ORE-EBUI                            | 33.6       | 26.3       | 31.8        |
| 2    | OW-DETR                             | 38.2       | 28.1       | 33.1        |
| 3    | FQ+Caps(1+0.7)                      | 36.3329    | 36.6812    | 36.4200     |
| 4    | FQ（增加epoch）                     | 44.9894    | 40.0600    | 43.7571     |
| 5    | FQ+Caps(1+0.7)（增加epoch, 214161） | 40.5325    | 39.6260    | 40.3058     |
| 6    | FQ+Caps(1+0.7)（增加epoch, 81）     | 43.5188    | 39.4311    | 42.4969     |

分析：

* 实验1：只通过阈值给 unknown 打伪标签（sim > 0.5）,不控制数量的时候，t1的结果：Cur-AP: 33.3939+U-Recall: 37.7108--> 说明模型太猖狂了，把什么都当作 unknown（可能原因：1、阈值0.5设置的太低；2、没有控制 unknown labels 的数量导致一张图片中被误标为 unknown 的样例很多）
* 实验2（第 3 行）：topk=1+unknown_threshold=0.7，结果如上表 FQ+Caps(1+0.7) 行
* 实验3（第 5 行）：ours 的 cur-map 和 pre-map 的差距相比于对照试验要小，both-map 更高。
  * 问题：各任务间的 u-recall 不稳定
  * 另（训练的 epoch 等超参应该还有得调）

## 20230721

* 实验4（第 6 行）：第5行的实验，t1，t2，t3任务的分类头输出维度分别是21，41，61，想了想应该是不太合理，所以全部改成了81，和ORE保持一致
  * mark：t2train的epoch可以考虑再加一点
* 直接将 OCPL 的愿你选哪个方法挪过来跑出来的效果很差，调整学习轮次、学习率、优化器都没有用，结果如下表；和南哥沟通了一下，需要做一些调整，后面稍微改动一下再看效果；

| OCPL初步原模原样搬运 | t1      | t2      | t3   | t4   |
| -------------------- | ------- | ------- | ---- | ---- |
| Prev mAP(↑)          |         | 42.1396 |      |      |
| Cur mAP(↑)           | 52.0953 | 31.3479 |      |      |
| Both mAP(↑)          |         | 36.7437 |      |      |
| WI(↓)                | 0.0549  | 0.0269  |      |      |
| A-OSE(↓)             | 17161   | 1505    |      |      |
| U-Recall(↑)          | 2.8730  | 0.0656  |      |      |

下一步实验：

FQ 在  **MS-COCO split** 上跑一个对比数据出来（得重新调学习率和 epoch）

第 4 行结果：初步调了一版epoch的结果（初始学习率与 FQ 原文一致为2.5e-5）

FQ+Caps 在**OWOD split** 上的性能跑一下（只有来自 COCO 的数据才有 Caption，由于 t1 任务的图片全部来自 voc，考虑使用图生文模型为图片生成描述）

baseline
