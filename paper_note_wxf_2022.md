# 20221007

### 1_AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE_ ICLR Jun 2021_有代码

作者：**Alexey Dosovitskiy***∗**,**†* **, Lucas Beyer***∗* **, Alexander Kolesnikov***∗* **, Dirk Weissenborn***∗***,****Xiaohua Zhai***∗* **, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,*****Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby***∗**,**†*

code：https://github.com/google-research/vision_transformer

贡献：将Transformer运用到图像分类任务。

由于Transformer的输入是一维序列，所以需要对图片进行预处理。即将图片（HxWxC）划分成一系列patch ((H/P)x(W/P)xP<sup>2</sup>xC),P为patch size。每个patch经过一个可训练的线性投影层后得到patch embedding（Nx(P<sup>2</sup>xC)）。给所有的patch embedding加上位置信息后即可送入Transformer Encoder。为了后续做分类，借助BERT当中的 [class] token，在patch embedding前面拼接可学习的class embedding，由于在Transformer Encoder中，class  token可以学到其他patch embedding的特征，因此在最后根据class  token对应的Transformer Encoder输出进行分类即可。

![image-20221005003353109](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221005003353109.png)

![image-20221005011759897](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221005011759897.png)

### 2_VirTex: Learning Visual Representations from Textual Annotations_CVPR 25 Sep 2021_有代码

作者：Karan Desai Justin Johnson University of Michigan *{*kdexd,justincj*}*@umich.edu

code：https://github.com/kdexd/virtex

目标：从更少的图像中学习高质量的视觉表示

贡献：本文主要贡献是，表明使用文本注释学习可转移的视觉表示，比其他方法有更好的数据效率(**Data Effificiency**)且节省注释( 

**Annotation Cost Effificiency**)。(如下图)

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221006211732871.png" alt="image-20221006211732871" style="zoom: 80%;" />

常见学习视觉表示的方法：

1）先在ImageNet上进行图片分类任务预训练CNN网络，然后将学到的视觉表示应用于下游任务。（缺点：预训练阶段需要大量人工标注，耗费太大而难以扩大规模。）

2）无监督预训练，使用未标记图像学习视觉表示然后转移到下游任务。

本文基于监督预训练，更有效地使用每幅图像，即用更少的图像学习高质量的视觉表示。

**Method**



<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221006205518553.png" alt="image-20221006205518553" style="zoom:80%;" />

如图 1所示：首先，从头开始联合训练 ConvNet 和 Transformer ，学习视觉表示（visual representations）。然后，将学习到的特征转移到下游的视觉识别任务中。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221006205800009.png" alt="image-20221006205800009" style="zoom:80%;" />

如图2，Image Captioning包含更多的图片信息，可以学到更多的信息，包括：多目标（如猫、蛋糕）、属性（如橘黄色的猫）、目标的关系（如盯着苹果）、动作、空间布局（如猫在盘子旁边）。因此，该预训练方法可以学到更丰富有效的视觉特征。

![image-20221006210506973](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221006210506973.png)

如图3，预训练包含两个模块：视觉模块（左）和语言模块（中）（右边是语言模块内部结构）

1）视觉模块（ResNet-50 ）：学习视觉特征，随机初始化（不用预训练好的模型）

经过ResNet-50后输出的视觉特征v1（7\*7\*2048）， 经过全连接层映射到v2（7\*7\*H）

（需要注意的是预训练的时候使用v2作为视觉特征输入到语言模块，而在训练下游任务时使用的是v1作为视觉特征）

2）语言模块：双向transformer
这里采用的是transformer的Decoder部分，输入为视觉特征（ image features from the visual backbone）和图片描述 （*C* = (*c*0*, c*1, . . . , c<sub>T</sub> , c<sub>(T +1)</sub> ），训练目标是生成图片描述。

双向生成：

​	从左向右：不断根据前面的词来预测后面的词（只能看到左边的词）

​	从右向左：与上面相反



# 20221014

### 3_IMAGINARYNET: LEARNING OBJECT DETECTORS WITHOUT REAL IMAGES AND ANNOTATIONS_Under review as a conference paper at ICLR 2023_暂无代码

作者：暂未公布

目标：给定一个合适的预训练模型，做到无需用真实的图像和手动注释来学习目标检测。

贡献：

1. 提出框架IMAGINARYNET——据作者所说，本文是第一篇不用真实图像和手工注释来学习目标检测的工作。
2. 提出一个新颖的目标检测范式——Imaginary-Supervised Object Detection(ISOD)
3. 通过结合真实图像和手动注释，ImaginaryNet显著改进了基线模型，同时实现了最先进的或相当的性能。

**RELATED WORK**

1. 目标检测：大多数目标检测方法（FSOD）都需要大量带有边框级别注释（box-level annotations）的训练数据；弱监督目标检测（FSOD）仅需要图片级别（image-level annotations）的标签；半监督目标检测（SSOD）结合无标签数据和边框级别的标签数据。这些方法的缺点在于都**依赖真实图片和手工注释**。
2.  SIM2REAL 模拟到真实：通过模拟来让模型学到真实技能——常用在机器人领域。在视觉领域也有人这么尝试，但是由于模拟和真实图片之间存在domain gap，因此模型还是**需要真实的图片或注释**，将知识迁移到真实领域。
3. 预训练模型

**METHODOLOGY**

ImaginaryNet框架如下：

![image-20221013141653632](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221013141653632.png)

随机采样类别标签c，添加到一个前缀模板如A photo of a *c*，喂到语言模型（GPT-2）中产生完整描述如A photo of a bird standing with others。图片生成模型（DALLE-mini）基于该完整描述（结合随机采样服从高斯分布的噪声）来生成虚拟图片。使用卷积网络（基于ImageNet数据集预训练好的ResNet50）从生成的图片中提取feature map，用Selective Search产生proposals，经过RoI Pooling层获得proposal representations（表示为{f<sub>i</sub>}）。将f<sub>i</sub>和类别标签一起提供给检测头来优化目标检测。

训练的时候只更新Representation Generator和Detection Head的参数。

推理时，去掉语言模型与图片生成模型。将图片和Representation Generator产生的 proposal representations作为检测头的输入进行目标检测。

 **EXPERIMENTS**

与弱监督的OICR以及无监督的CLIP对比：本文方法是有效的Figure3说明虚拟图片包含了与真实图片中相似的特征。



![image-20221013143854705](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221013143854705.png)

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221013144403769.png" alt="image-20221013144403769"  />

与弱监督方法对比：

![image-20221013143656965](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221013143656965.png)

虚拟图片vs真实图片：

![image-20221013144016029](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221013144016029.png)

**ABLATION**

1. IMAGINARY SAMPLES的影响，随着采样量的增加效果稳步提升。![image-20221013144243611](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221013144243611.png)
2. LANGUAGE MODEL的影响：![image-20221013144605752](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221013144605752.png)使用语言模型能提升总体性能。对于个别类别性能反而下降，作者解释为the failure of the language model。
3. DATA AUGMENTATIONS的影响：![image-20221013144753187](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221013144753187.png)

本文的缺陷：

本文的工作受到生成模型的限制，生成的图片可能不符合文本描述，会给学习带来额外的噪声。

### 4_Dataset Distillation by Matching Training Trajectories_CVPR 22 Mar 2022_有代码

作者：George Cazenavette<sup>1</sup> Tongzhou Wang<sup>2</sup> Antonio Torralba<sup>2</sup> Alexei A. Efros<sup>3</sup>   Jun-Yan Zhu<sup>1</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup>Massachusetts Institute of Technology  <sup>3</sup>UC Berkeley

code：https://github.com/georgecazenavette/mtt-distillation

目标：将大数据集蒸馏成一个小的数据集（合成数据集），使得模型在小数据集上的测试准确性能够媲美在大数据集（真实数据集）上的测试准确度。

贡献：本文的方法超越了现有方法，并且能够在分辨率更高（相较于32\*32和64\*64更高的分辨率128\*128）的图片上进行数据蒸馏。

如下图：数据蒸馏思想

![image-20221014085502218](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221014085502218.png)

数据蒸馏现有方法：

1. 使用端到端训练，但往往需要大量的计算和内存，并存在不精确松弛以及展开多次迭代的训练不稳定。
2. 为了降低优化难度，一些方法关注短程行为（focus on short-range behavior），使在蒸馏数据上的单步训练匹配在真实数据上的。然而由于蒸馏数据会被多次迭代，在验证过程中错误可能会被累积。

本文方法：

为了解决上述问题，本文通过直接模拟在真实数据集上训练的网络的长期训练动态。本文将基于合成数据训练的参数轨迹片段与基于真实数据训练的模型中预先记录的轨迹片段进行匹配，从而避免短视(即专注于单个步骤)和难以优化(即对完整轨迹建模)。

本文将在真实数据集上训练的参数序列表示为expert trajectory（当中保存了每个epoch的参数），用来指导合成数据集的蒸馏。由于expert trajectory是完全基于真实数据集训练的，所以可以预先训练好得到expert trajectory，从而节省了蒸馏时间。

具体蒸馏方法如下图：

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221014092302133.png" alt="image-20221014092302133" style="zoom:80%;" />

算法如下：

![image-20221014092528257](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221014092528257.png)

在每个蒸馏步骤中，我们首先从我们的专家轨迹（ expert trajectories）中的一个随机时间步θt<sup>*</sup>采样参数并使用它们初始化我们的学生参数θˆt: =θt<sup>\*</sup>.

在初始化学生参数后，基于合成数据的分类损失来对学生参数进行N步梯度下降更新<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221014093106833.png" alt="image-20221014093106833"  />

其中A是可微分增强操作。α是个可学习的学习率。

然后计算更新后的学生参数![image-20221014093446189](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221014093446189.png)和expert trajectory的模型参数（带*的）的匹配损失![image-20221014093227557](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221014093227557.png)

针对内存问题：

由于D<sub>syn</sub> 每个类图片数量过多，一次性输入会存在内存占用过高问题，为了解决该问题，本文将D<sub>syn</sub>  划分为多个batch，即对应于Algorithm第10行。

# 20221021

### 5_CLIP MODEL IS AN EFFICIENT CONTINUAL LEARNER_暂未发出\_有代码

作者：Vishal Thengane<sup>1</sup>, Salman Khan<sup>1,2</sup>, Munawar Hayat<sup>3</sup> , Fahad Khan<sup>1,4</sup>

<sup>1</sup>Mohamed bin Zayed University of Artifificial Intelligence, UAE (MBZUAI)

<sup>2</sup>Australian National University, Australia (ANU)

<sup>3</sup>Monash University, Australia (蒙纳士大学)

<sup>4</sup>Linkoping University, Sweden (林雪平大学)

vishal.thengane@mbzuai.ac.ae 

代码：https://github.com/vgthengane/Continual-CLIP

问题：主要问题是探索是否可以用一种简单的通用方法取代最先进的窄模型（narrow models），这种方法不需要对每个增量步骤进行训练，不需要任何示例内存存储，可以跨所有现有的增量学习协议工作，只需要很少或不需要超参数调优。

贡献：(CLIP涨点神器)

1. 本文展示了一个冻结的CLIP（对比语言-图像模型）在没有任何微调的情况下提供了惊人的持续学习能力，第一个在持续学习领域报告CLIP zero_shot性能的工作；
2. 本文的研究结果旨在巩固持续学习领域中在特定环境下工作（TIL 任务增量, CIL 类增量, DIL 域增量, and TFCL 任务不可知）的碎片化努力；
3. 强调可以在多个环境下工作的通用方法的必要性。

![image-20221018111347857](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221018111347857.png)

可能存在的缺陷：信息泄露

1. CLIP预训练时可能已经见过下游任务的图片，因此效果好

2. 碰到相似的语义，模型可能会困惑，见Figure4。（例如，模型将类索引“50”（类名是“mouse”鼠标）预测为“74”（“shrew”地鼠）。）

   ![image-20221018111750327](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221018111750327.png)



### 6_CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention\_暂未发出_未公布代码

作者：Ziyu Guo<sup>1,2\*</sup>, Renrui Zhang<sup>2,3\*</sup>, Longtian Qiu<sup>4\*</sup>, Xianzheng Ma<sup>3\*</sup>,Xupeng Miao<sup>2</sup>, Xuming He<sup>4</sup> , Bin Cui<sup>2</sup>

单位：北京大学，上海AI Lab，上科大

**贡献**：提出了CALIP，据作者称，是第一个通过无参数注意模块对CLIP进行zero-shot增强的工作。通过做实验证明CALIP在广泛的2D和3D基准测试中获得了较好的性能；进一步，作者在无参的基础上增加了可学习的线性层，引入参数版本CALIP-FS，进一步获得了更好的效果。

**当前问题**：由于CLIP显示出学习可转移的视觉特征的能力，并且能够在zero-shot分类任务上取得不错的准确度。为了进一步提高下游任务的表现性能，现有工作都是通过在CLIP之上增加可学习的模块（缺点：增加额外的训练损失，并且需要额外的训练数据）以及微调来达到zero-shot能力，这严重阻碍了模型部署和知识转移的效。率。

**CALIP思路**：本文引导视觉和文本表征相互作用，并通过注意力来探索跨模态的信息特征。由于CLIP在预训练时，已经大大减少了两种模式（视觉和文本）之间的嵌入距离，因此本文抛弃了注意力中的可学习参数（ discard all learnable parameters in the attention）并使视觉和文本模块相互更新（bidirectionally update the multimodal features）。

无参数zero-shot版本：

![image-20221019233618740](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221019233618740.png)

有参数few-shot版本：

![image-20221019233941333](C:\Users\xuefei\AppData\Roaming\Typora\typora-user-images\image-20221019233941333.png)

消融实验中提到，增加第四中结合方式（<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221019234417313.png" alt="image-20221019234417313" style="zoom:80%;" />)反而会对效果起反作用。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221019234343069.png" alt="image-20221019234343069" style="zoom:80%;" />

本文的实验只针对下游分类任务，下一步将进一步推进到其他视觉领域如目标检测和实例分割。

# 20221028

### 7_UC-OWOD: Unknown-Classified Open World Object Detection_ECCV 2022_暂未上传代码

作者：[Zhiheng Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Z), [Yue Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Y), [Xingyu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+X), [Zhengxing Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Z), [Liwen Kang](https://arxiv.org/search/cs?searchtype=author&query=Kang%2C+L), [Junzhi Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+J)

code：https://github.com/johnwuzh/uc-owod

问题：OWOD（Open World Object Detection ）需要检测未知对象，并逐渐学习识别出的未知类。

​	问题1：当前方法存在的问题是，将所有未知的类识别为unknown 类，不能将未知实例区分为多个未知类；

​	问题2：未知类别物体分类的评价标准不成熟，无法对两个不同类的未知对象被检测为同一类的情况进行评估。

贡献：

1. 本文引入了一个新的问题设置，即未知分类的开放世界对象检测（UC-OWOD），以启发未来对现实世界对象检测的研究；
2. 提出了一种基于未知标签感知建议（unknown label-aware proposal）、未知判别分类头（unknown-discriminative classification head）、基于相似度的未知分类（similarity-based unknown classification）和未知聚类细化（unknown clustering refinement）的UC-OWOD问题求解方法；
3. 提出了一种新的UC-OWOD评价指标，用于评价未知目标的定位和分类。

对象检测器M<sub>C</sub>能够识别属于任何已知类的测试实例，还可以将新的或不可见的类实例作为不同的未知类进行检测。人类用户可以从未知的实例集U<sup>t</sup>中识别出感兴趣的新类，并提供相应的训练示例。更新已知类集K<sup>t+1</sup> = K<sup>t</sup>∪{C+1，…C + u}。通过在下一个任务中增量地添加u个新类，学习者创建了一个更新的模型M<sub>C+u</sub>，而不需要在整个数据集上重新训练模型。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221022201858918.png" alt="image-20221022201858918" style="zoom:80%;" />

总体架构：

1. ULP和UCH从背景板中发现位置类别；
2. SUC将未知对象检测为不同的类；
3. UCR优化未知对象的分类，并增强算法的鲁棒性。

**UCH（Unknown-Discriminative Classification Head）：**

为了解决原始分类策略不能分类多个未知对象类别，本文修改分类损失为：![image-20221022202806465](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221022202806465.png)

**SUC（ Similarity-Based Unknown Classification）：**

相似性矩阵：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023190514614.png" alt="image-20221023190514614" style="zoom:80%;" />(E是UCH的输出，代表类别信息)

supervised method for known classes，类别矩阵：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023190724027.png" alt="image-20221023190724027" style="zoom:80%;" />（l<sub>i</sub>是第i个实例的类别），相似性损失：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023191059801.png" alt="image-20221023191059801" style="zoom:80%;" />

self-supervised method for unknown classes，<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023190848514.png" alt="image-20221023190848514" style="zoom:80%;" /> <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023190933435.png" alt="image-20221023190933435" style="zoom:80%;" />，相似性损失：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023191144937.png" alt="image-20221023191144937" style="zoom:80%;" />，惩罚项为<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023191243569.png" alt="image-20221023191243569" style="zoom:80%;" />

**UCR（Unknown Clustering Refinement）：**

基于之前的网络输出改进未知分类，

第一步：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023191808121.png" alt="image-20221023191808121" style="zoom:80%;" />，*P<sub>ij</sub>*可以解释为将实例i分配给集群j的概率(软分配)，*Φ<sub>j</sub>*代表集群中心；

第二步：辅助目标分布Q，<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023192228663.png" alt="image-20221023192228663" style="zoom:80%;" />(其中<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023192337546.png" alt="image-20221023192337546" style="zoom:80%;" /> is soft cluster frequencies)

第三步：使用软分配P和辅助分布Q的KL散度来优化聚类损失，<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023192556764.png" alt="image-20221023192556764" style="zoom:80%;" />。

训练损失：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023194223478.png" alt="image-20221023194223478" style="zoom:80%;" />

对未知类别的Refine阶段，仅使用KL散度损失来进行优化：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023194407282.png" alt="image-20221023194407282" style="zoom:80%;" />

### 8_Open Vocabulary Object Detection with Pseudo Bounding-Box Labels_ECCV 2022_有代码

作者：[Mingfei Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+M), [Chen Xing](https://arxiv.org/search/cs?searchtype=author&query=Xing%2C+C), [Juan Carlos Niebles](https://arxiv.org/search/cs?searchtype=author&query=Niebles%2C+J+C), [Junnan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Ran Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+R), [Wenhao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+W), [Caiming Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+C)

代码：https://github.com/salesforce/pb-ovd （代码多个包依赖冲突）

当前问题：现有的开放词汇目标检测和零样本目标检测方法通过训练预定义好的基类，推广到检测新的类别。但是受限于可用于训练的基类样本数量较少，当推理时碰到的类别很不一样时效果很差。

贡献：为了扩大基类数量，本文利用*vision-language models*（本文用 ALBEF model pre-trained with 14M data）的定位能力来产生伪标签（pseudo bounding-box labels），并以此来训练目标检测器。对比于传统使用人工标注的基类数据，本文方法可以轻易产生大量带有伪标签的数据。

据作者所说，本文是第一篇将伪标签用于开放词汇检测的工作。

近期工作：

1. Joseph et al. 引入ORE（Towards open world object detection  https://arxiv.org/abs/2103.02603），通过对比聚类和基于能量的未知识别来增量地学习未知目标；
2.  Zareian et al （Open-Vocabulary Object Detection Using Captions https://arxiv.org/abs/2011.10678）迁移一个预训练好的视觉-语言模型知识，使用视觉-语言模型的image encoder的参数来初始化检测器，达到开放词汇目标检测的先进水平；
3. Gu et al. 提出ViLD（Open-vocabulary Object Detection via Vision and Language Knowledge Distillation https://arxiv.org/abs/2104.13921），通过蒸馏大型视觉-语言模型的知识实现好的zero-shot性能。

本文方法：

![image-20221023161712469](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023161712469.png)

Method分成两部分：1、生成伪标签；2、用伪标签训练目标检测器

![image-20221023175202510](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023175202510.png)

1、生成伪标签：

通过在预定义的对象词汇表中维护感兴趣的对象。

视觉注意评分(visual attention scores )<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023165229246.png" alt="image-20221023165229246" style="zoom:80%;" />可以直接反映不同视觉区域对token x<sub>t</sub>的重要性;

s是一个标量，表示图像与其标题之间的相似性；

如果在交叉注意层中有多个注意头，我们将所有注意头的激活映射Φ<sub>t</sub>取平均值作为最终的激活映射;

选择与Φt重叠最大的一个框作为伪标签框 <img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023165612002.png" alt="image-20221023165612002" style="zoom:80%;" />;

2、开放词汇的目标检测：

![image-20221023170548099](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221023170548099.png)

###  9_OW-DETR: Open-world Detection Transformer_CVPR 2022_有代码

作者：[Akshita Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+A), [Sanath Narayan](https://arxiv.org/search/cs?searchtype=author&query=Narayan%2C+S), **[K J Joseph](https://arxiv.org/search/cs?searchtype=author&query=Joseph%2C+K+J),** [Salman Khan](https://arxiv.org/search/cs?searchtype=author&query=Khan%2C+S), [Fahad Shahbaz Khan](https://arxiv.org/search/cs?searchtype=author&query=Khan%2C+F+S), [Mubarak Shah](https://arxiv.org/search/cs?searchtype=author&query=Shah%2C+M)

code：https://github.com/akshitac8/ow-detr  （Ubuntu 16.0`, `CUDA 10.2`, `GCC 5.4`, `Python 3.7)

贡献：本文提出了一个基于Transformer端到端的框架***OW-DETR***，用于OWOD（*open-world object detection*）开放世界对象检测。

​			在Deformable DETR（https://arxiv.org/abs/2010.04159）的基础上引入三个模块来进行开放世界的对象检测，分别是：

1. 一个注意力驱动的伪标签机制（an attention-driven pseudo labeling mechanism），用于选择可能的未知类别查询候选；
2. 一个新颖分类分支（ a novelty classification branch），学习将对象查询（the object queries）分类为已知类当中的某一类或者未知类；
3. 一个对象分支（an ‘objectness’ branch），学习从背景中分离出前景对象（包括ground-truth known and pseudo labeled unknown instances）

问题描述：

![image-20221025203800050](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025203800050.png)：t 时刻已知对象类别

![image-20221025203911402](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025203911402.png)：一个包含N张对应于标签![image-20221025204058553](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025204058553.png)的图片![image-20221025204014815](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025204014815.png)的数据集。

​	（![image-20221025204136313](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025204136313.png)代表K个对象实例标签，y<sub>k</sub>=[l<sub>k</sub>, x<sub>k</sub>, y<sub>k</sub>, w<sub>k</sub>, h<sub>k</sub>])

![image-20221025204652359](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025204652359.png)：代表测试时可能遇到的未知类别

OWOD的设定为，在 t 时刻训练模型M<sup>t</sup>，在检测之前碰到过的已知类别C之外，还要识别未见过的类别实例（denoted by label 0）。由M<sup>t</sup>识别出的未知类别实例![image-20221025205048715](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025205048715.png)会交给人类，挑出感兴趣的 n 个类别并相应给出新的训练样本，然后将这些新的类别添加到原来的已知类别中，那么![image-20221025205435257](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025205435257.png)![image-20221025205500025](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025205500025.png)，对于先前的类别K<sup>t</sup>，仅有少量样本存储在有限的内存中（be stored in a bounded memory)。那么，M<sup>t</sup>不用从头开始训练整个数据集，通过增量训练来获得更新的模型M<sup>t+1</sup>。

![image-20221025210246791](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025210246791.png)

本文方法的总体框架如上图Figure 2所示。输入一张图片（H X W）（图片中由Y个对象实例）到特征提取backbone中。获得D维多尺度的特征，并输入到带有可变注意力模块的transformer的encoder-decoder中，输出![image-20221025211739094](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025211739094.png)（编码了图片中潜在的对象实例)，然后q<sub>e</sub>分别被送入3个分支。

首先，用基于分类和边界框预测的二分类损失选出unique queries作为与ground-truth (GT)最匹配的已知类别实例，剩余的object queries用来选出候选未知类别实例。

- **Attention-driven Pseudo-labeling**

f 代表从backbone提取的中间层D维大小为h x w的特征图。（特征激活的幅度给出了在该空间位置存在对象的指示，因此可以用于计算窗口内对象的可信度）；

**b** = [x<sub>b</sub>, y<sub>b</sub>, w<sub>b</sub>, h<sub>b</sub>] 代表一个box proposal，通过回归分支得到M个b；

A ∈ R<sup>h×w</sup> is the feature map f averaged over the channels D

![image-20221025213528825](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025213528825.png)：![image-20221025213603155](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221025213603155.png)

若一张图片中包含K个已知类别的实例，那么会从M-K个未知类别中选择top<sub>k</sub>个 objectness scores *s<sub>o</sub>*得分最高的pseudo-labeled as unknown objects。

- **Novelty Classification**

该分支对 the selected pseudo-unknown objects进行训练

-  **Foreground Objectness**

如上，新类别分类分支 Fcls 是特定于类的，它将一个 query 嵌入到一个 C+1 类中：C 个已知类 或 1个未知类或背景。虽然这允许学习已知类和未知类之间特定于类的可分离性，但它不允许将知识从已知目标转移到未知目标，而这对于理解在 OWOD 任务中是什么构成未知目标至关重要。此外，由于缺乏未知类的监督，注意力驱动伪标注的准确性可能较低，这将导致大多数 query 嵌入都将在背景中进行预测。为了缓解这个问题，引入了一个前景目标性分支 Fobj，它对 query 嵌入 qe 的 “目标性” 进行评分，以便更加好地将前景目标 (已知的和未知的) 从背景中分离出来。学习把与前景目标相对应的 queries 评分高于背景，这样可以改进对未知目标的检测，否则这些目标将被检测为背景。这种类无关的评分还有助于模型将知识从已知类转移到未知类，即构成前景目标的特征。

**训练：**

端到端损失：$L=L_n+L_r+\alpha{L_o}$

# 20221104

### 10_End-to-End Object Detection with Transformers_ECCV2020_有代码

作者：![image-20221031131112462](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031131112462.png)

code：https://github.com/facebookresearch/detr

贡献：本文提出直接将目标检测作为集合预测问题，DETR(DEtection TRansformer)主要由两部分组成，一个是基于集合的全局损失，通过二分图匹配达到独特的预测；一个是transformer的encoder-decoder架构。该方法简化了目标检测流程，去掉了许多手工设计的部分，如需要先验知识的非极大值抑制程序以及anchor产生的程序。

DETR存在的缺陷：虽然在大目标上检测效果较好（可能是得益于transformer的自注意力机制能够看到全局信息），但在小目标上的检测效果较差。

如图Fig.1展示了DETR整体框架

![image-20221031133558538](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031133558538.png)

![image-20221031215629455](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031215629455.png)

DETR两个主要成分：

**1、 a set prediction loss that forces unique matching between predicted and ground truth boxes**

第一步：匹配![image-20221031143108740](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031143108740.png)L<sub>match</sub>代表ground truth y<sub>i</sub>和第*σ*(*i*)个预测的y^匹配损失，最有匹配通过匈牙利算法（Hungarian algorithm)计算得到。（N代表输出预测框的个数，远大于一张图片一般情况下包含的目标个数；y<sub>i</sub>的size也为N（图片中目标的个数加上∅填充为N个 ）；)

L<sub>match</sub>既考虑了类别预测也考虑了预测的边框与ground truth的相似度。

令![image-20221031212242352](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031212242352.png)为第*σ*(*i*)个预测是类别c<sub>i</sub>的可能性，![image-20221031212350464](E:\SHU\paper\wang\0_note\new\typora-user-images\image-20221031212350464.png)是第*σ*(*i*)个预测框，则![image-20221031212438695](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031212438695.png) = ![image-20221031212508408](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031212508408.png)

**2、an architecture that predicts (in a single pass) a set of objects and models their relation**

基于**第一步算出来的最优匹配**来算匈牙利损失（用于回传的损失）：![image-20221031213401376](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031213401376.png)（![image-20221031213429744](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031213429744.png))

![image-20221031213701881](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031213701881.png) = ![image-20221031213730009](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031213730009.png)（其中![image-20221031213800424](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221031213800424.png))

文中提到：

​	1、为了让分类损失和box损失在一致的取值空间，所以作者把分类loss的log去掉了，这样能使得结果好一些；

​	2、对于box的loss，一般用L1 loss就可以了（一般来说框越大，L1 loss就容易越大，但由于DETR使用transformer能够关注到全局的特征，所以容易出大框，那么会导致不容易优化。因此本文在Box loss中加入了GIOU loss（与框大小无关））

**Trick：**

1、在Transformer每一个Decoder里，都会先做一个object queries的自注意力操作（第一层的decoder里可以省略），主要是为了移除冗余框；

2、为了让模型收敛更快，训练更稳定，在每层decoder后面加FFN并计算auxiliary loss。

### 11_Towards Open World Object Detection_CVPR 2021_有代码

作者：![image-20221102192939152](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102192939152.png)

code：https://github.com/JosephKJ/OWOD

贡献：

1. 提出一个新的问题设定：开放世界目标检测（Open World Object Detection）
   - 测试集中可能出现未知类别的物体，网络需将其识别为unknown类别 ；
   - 如果之后给出某个未知的标签，需要网络能够增量学习新的类别。
2. 针对OWOD问题提出了一个解决方案ORE（![image-20221102193534930](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102193534930.png))，该方法在基于Faster-RCNN的基础上增加了对比聚类（contrastive clustering)和基于能量的识别（energy based unknown identifification)；
3. 引入了一个全面的实验设置，它有助于测量对象检测器的开放世界特性，并在其上与竞争基线方法进行ORE基准测试；
4. 本文方法意外地在持续目标检测（Incremental Object Detection）上实现了SOTA性能。

本文框架如下：![image-20221102203646400](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102203646400.png)

**Contrastive Clustering**

在隐空间进行类别分离应该对OWOD识别未知类有好处，因此本文通过增加对比聚类来使得相同的类别应该距离相近，不同的类别距离应该较远。通过最小化聚类损失来达到隐空间的类别区分，聚类损失定义为：![image-20221102204143880](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102204143880.png)，其中D是距离函数，∆ 定义了相似和不相似项的接近程度，p<sub>i</sub>是 prototype vector，**f**<sub>c</sub> *∈* R <sup>d</sup>是中间层提取的类别c的特征向量。

具体实现：对每个类别的隐空间特征，统计一段时间内迭代样本的特征均值（每I<sub>p</sub>次迭代更新一次当前的特征均值），作为聚类中心（I<sub>b</sub>是不叠加对比聚类损失的轮数，用以初始化已知类别的特征向量）。并约束期望该类样本特征都靠近该中心，其他类的样本特征远离该中心；聚类中心在训练过程中不断更新。
优势：1. 可以帮助网络辨别unknown类别的物体与已知类别的表示有何不同 2. 促进网络在不覆盖潜在空间中原有类别表示情况下学习未知类别的潜在表示

![image-20221102204731968](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102204731968.png)



 **Auto-labelling Unknowns with RPN**

利用RPN的建议框类别无关特性，将RPN提取的置信度最高的（最有可能是object的）前K个背景建议框作为位置对象的建议框，作为unknown objects。

**Energy Based Unknown Identififier**

当训练结束，推测阶段，基于一个输入特征向量F，该模型利用一个基于能量的模型去计算获取其预测类别。基于能量的模型（EBM）：把我们所关心变量的各种组合和一个标量能量联系在一起。我们训练模型的过程就是不断改变标量能量的过程，因此就有了数学上期望的意义。比如，如果一个变量组合被认为是合理的，它同时也具有较小的能量。基于能量的概率模型通过能量函数来定义概率分布，可以用梯度下降法来训练。

![image-20221102211716961](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102211716961.png)

对所有L中的值的能量表示为![image-20221102211836487](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102211836487.png)，其中T是温度参数。

![image-20221102213259694](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102213259694.png)

这里的g就是网络的全连接层的输出，普通的网络就是把g接到softmax去做分类的，这里利用了g构建了E，这个E的意义在于，对于不同的输入f，有不同的能量标量值与之对应，而由于在隐空间将已知类别和未知类别分得很开，所以这里已知类别的特征向量输入产生的能量和未知类别的特征向量输入产生的能量值有较大区分，如下图所示：
![image-20221102213536661](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102213536661.png)

为了找到区分known和unknown的能量值阈值，用韦伯分布去拟合两部分数据，得到两个韦伯分布，因此对于一个特征向量输入，产生一个能量值，能量值在两个分布上对应了两个概率密度，当在known分布上的概率密度值高于unknown分布上的概率密度值时，认为该特征来自一个已知类别，否则来自未知类别。之所以是韦伯分布，是实验出来的，比伽马分布、正态分布、指数分布都更优。这样一来，在实现上其实只是分为两步，首先通过能量值判断是不是known；若是known，再看哪个类别的输出值更高。

**对新类别检测能力的评估指标**：![image-20221102220222264](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102220222264.png)和Absolute Open-Set Error (**A-OSE**) （report the number count of

unknown objects that get wrongly classified as any of the known class）

**消融实验结果：**![image-20221102220405176](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221102220405176.png)

总结：

1. 目标的遮挡和拥挤是分类器容易产生混淆的情况； 

2. 困难的视角（例如背面）也会导致一些错误的分类（如文中图4、12中的长颈鹿）；

3. 很难检测到与较大的已知目标同时发生的小型未知目标。 

   由于ORE是朝这个方向迈出的第一步，因此本文希望这些已发现的缺点将成为进一步研究的基础。

# 20221111

### 12_ORDER: Open World Object Detection_on Road Scenes_NeurIPS-W 2021_无代码

**作者**：![image-20221103202602089](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221103202602089.png)

**贡献**：本文主要基于该团队的前一篇论文（Towards Open World Object Detection提出的方案ORE）存在的缺点做出改进，对于在道路上做OWOD遇到的困难（小对象占比很多以及对象类内规模变化大）提出应对方案。

1. 引入Feature-Mix增大潜在特征空间中已知类与未知类之间的距离，从而提高检测未知对象的能力；
2. 提出Focal regression loss来加大对小目标边框回归损失的惩罚并根据目标大小动态改变该损失来解决小目标和类内规模变化的问题。

本文使用的数据集是**BDD**（Berkeley Deep Drive datasets， 包含15个类，）和**IDD**（ Indian Driving Dataset，包含10个类），数据划分如下：

![image-20221103203750657](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221103203750657.png)

**ORDER框架：**![image-20221103204019078](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221103204019078.png)

**Feature-Mix**：通过混合已知和未知的特征并抑制已知特征引起的激活，来提高对未知类的识别能力。

![image-20221103204247438](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221103204247438.png)f<sub>k</sub>代表已知类的特征，f<sub>unk</sub>代表未知类的特征。

feature-mix loss表示为![image-20221103204839155](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221103204839155.png)y代表真实标签，C<sub>unk</sub>是未知类分类器。本文通过一个小验证集来训练Feature-Mix。

*L*<sub>*unk*</sub> 在总损失中占的权重分别为0.001（IDD）和0.01（BDD）。

**Focal Regression Loss**：解决两个问题

- 通过惩罚小的边界框来检测小的物体；
- 通过包含边界框信息来解决类内部（对象大小）变化问题。

![image-20221103205808636](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221103205808636.png)

通过 (1*−*IoU)<sup> *γ* <sup>*</sup></sup>来调节TOU损失， *γ* <sup>∗</sup> *∈* [0*,* *∞*) ， *Ar*<sub>*bbox*<sub>*gt*</sub></sub>是unnormalized bounding box area.

*γ* 的值分别为0.4（IDD）和0.1（BDD）。

使用double logarithmic的原因：

- 它可以防止当<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221103210420904.png" alt="image-20221103210420904" style="zoom:80%;" />很小（对象所占面积很大）时*γ* <sup>∗</sup>超调。（可以理解为，对于大面积目标就不调整了）
- 它可以平滑<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221103210420904.png" alt="image-20221103210420904" style="zoom:80%;" />的变化区域使训练更加稳定。

相较于大边框物体，小边框的*γ* <sup>∗</sup>值更大，会导致更多的惩罚。（？）

**Curriculum Training**：由于小目标被认为比大目标更难检测，因此本文还采用Curriculum Learning（课程学习）策略，让网络从简单样本到困难样本渐进式地学习。

Curriculum Learning会根据样本的难易程度，给不同难度的训练样本分配不同的权重。初始阶段，给简单样本的权重最高，随着训练过程的持续，较难样本的权重将会逐渐被调高。**这样一个对样本进行权重动态分配的过程被论文称之为课程（Curriculum），课程初始阶段简易样本居多，课程末尾阶段样本难度增加。**

![image-20221103211913988](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221103211913988.png)I1, I2和I3是每组集合训练的迭代次数。

*Ar*<sub>*easy*</sub> and *Ar*<sub>*medium*</sub> are the area thresholds for selecting large and medium bounding boxes.

本文使用的评价标准和ORE一样：WI, A-OSE, mAP

### 13_Incremental Object Detection via Meta-Learning_IEEE（TPAMI） 2021_有代码

**作者**：![image-20221107195118308](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107195118308.png)

**code**：https://github.com/JosephKJ/iOD

**贡献**：目标检测应用于真实世界中的设定会存在学习新类忘记旧类别的问题，目前用于解决该问题的方法（蒸馏）虽然帮助保留了先前的学习，但是限制了模型快速适应新类的能力。本文提出一个基于元学习的方法来重构模型梯度，使得增量任务的信息得到最佳共享。

贡献点：

1. 本文出了一种基于梯度的元学习方法，该方法学习重塑梯度，从而对新旧任务和类增量目标检测问题实现最优更新；
2. 本文提出了一种新的损失公式，通过学习一组可推广的梯度方向，来对抗由于知识蒸馏而造成的不妥协性（intransigence）。

**METHODOLOGY**

本文框架基于Faster-RCNN

![image-20221107202528737](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107202528737.png)

**Meta-Learning the Gradient Preconditioning**

标准的目标检测训练过程中，参数更新公式为![image-20221107202824908](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107202824908.png)；

本文引入元学习的预处理矩阵*P(θ;φ)*,它将梯度扭转到最陡峭的方向，因此参数更新变为![image-20221107203148422](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107203148422.png)，作为warp layers加入到模型中。

那么模型参数可分为两个部分**θ** = **ψ** *∪* **φ**，其中**ψ** *∩* **φ** = Ø，**ψ** 为任务参数（task parameters）， **φ** 为（warp parameters）

warp layers隐式地有助于建模引入到检测器的所有任务的联合任务分布，这使得更好地推广到新任务，更快的收敛和减轻灾难性遗忘。

**Loss**

![image-20221107203945860](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107203945860.png) 

![image-20221107204013754](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107204013754.png)（λ = 1)

![image-20221107204119367](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107204119367.png) 

![image-20221107204159192](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107204159192.png) 

![image-20221107204226680](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107204226680.png) 

![image-20221107204302719](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107204302719.png)

![image-20221107204333993](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221107204333993.png)

# 20221202

### 14_Vision GNN: An Image is Worth Graph of Nodes_CVPR2022_有代码

作者：**Kai Han**<sup>1*,*2*∗*</sup> **Yunhe Wang**<sup>2*∗*</sup> **Jianyuan Guo**<sup>2</sup> **Yehui Tang**<sup>2*,*3</sup> **Enhua Wu**<sup>1*,*4</sup>

<sup>1</sup>State Key Lab of Computer Science, ISCAS & UCAS

<sup>2</sup>Huawei Noah’s Ark Lab

<sup>3</sup>Peking University <sup>4</sup>University of Macau

{kai.han,yunhe.wang}@huawei.com, weh@ios.ac.cn

code：https://github.com/huawei-noah/Efficient-AI-Backbones

贡献：本文提出将图像表示为图（Graph）结构，并引入一种新的视觉图卷积 ViG（Vision GNN, vision graph neural network ）体系结构来提取视觉任务的图级（Graph-Level）特征。相较于常用的两大主流网络，CNN将图片视为网格、transformer将图片视为序列，本文将图片视为图结构，具有更灵活的特点。

本文是第一个将图神经网络应用于大型视觉任务的工作。

方法：

1、首先，将图片表示成图

若将每个像素作为图中的一个节点（node），则会需要大量的存储消耗。因此本文参考ViT的思想，先将图片分成N个patches，一个patch作为一个节点。

使用图（graph）表示图片的优势：

- 图是一个普遍使用的数据结构，网格和序列都可视为特殊的图；
- 对象一般都不是方形的，所以相较于网格和序列，图结构对于建模复杂对象具有更好的优势；
- 一个对象往往有某些组成部分，图（的边edge）可以表示这些组成部分之间的关系；
- 可以将图神经网络的优势用于解决视觉任务。

2、网络框架![image-20221201103942459](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221201103942459.png)

一个ViG Block分为两部分：

- 一个图卷积网络（GCN），负责处理图数据，聚合并更新相邻节点中的特征；

- 一个前馈神经网络（FFN），是包含两个全连接层的MLP，负责进行节点特征转换，鼓励节点多样性。

在计算机视觉领域，常见的结构类型包括各向同性结构（如Transformer）和金字塔结构（如CNN），本文为了对比ViG与过去这两种常见结构的经典模型之间的性能，为 ViG 建立了两种网络结构，即各向同性结构ViG和金字塔结构ViG。

各向同性结构:

![image-20221201104345403](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221201104345403.png)

金字塔结构:

![image-20221201104322337](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221201104322337.png)

本文分别在ImageNet数据集上做了图像分类实验和COCO数据集上做了目标检测实验。

对于分类任务，实验结果展现出了两个信息：1、将图片视作Graph能够在计算机视觉任务中取得非常好的结果；2、金字塔结构的ViG具有更好的性能，说明其能够更好地利用图片中的多尺度信息。

对于目标检测任务，本文采用RetinaNet和Mask R-CNN框架，分别替换各种不同的backbone，ViG也表现出了非常出色的性能。

# 20221209

### 15_Deformable DETR: Deformable Transformers for End-to-End Object Detection_ICLR 2021 Oral_有代码

code：https://github.com/fundamentalvision/Deformable-DETR

作者：[Xizhou Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+X), [Weijie Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+W), [Lewei Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+L), [Bin Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+B), [Xiaogang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Jifeng Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+J)

贡献：

DETR将Transformer引入到目标检测领域，实现了第一个端到端目标检测器并且不需要众多手工设计组件（如anchor、固定规则的标签分配策略、NMS后处理等）。

但DETR存在两个重要缺陷：**收敛慢**且**能够处理的特征分辨率有限**，这些缺陷都是由Transformer导致的，原因如下：

1. **Transformer在初始化时，分配给所有特征像素的注意力权重几乎是均等的**，这就造成了模型需要长时间去学习关注真正有意义的位置，这些位置应该是稀疏的；
2. **Transformer在计算注意力权重时，伴随着高计算量与空间复杂度。**特别是在编码器部分，与特征像素点的数量成平方级关系，因此难以处理高分辨率的特征（这点也是DETR在小目标检测上效果较差的原因）。

方法：

作者认为，既然要学习稀疏的空间位置，能不能用可变形卷积呢？但可变形卷积同时也缺乏关系建模能力，而这点恰好是Transformer最擅长的，于是作者最终就将两者结合在一起，提出了**可变形注意力模块（deformable attention module）**，相比于Transformer，在这里，每个特征像素不必与所有特征像素交互计算，只需要与部分基于采样获得的其它像素交互即可，这就大大加速了模型收敛，同时也降低了计算复杂度与所需的空间资源。另外，该模块能够很方便地应用到多尺度特征上，所以不需要FPN。

另外，作者还提出**迭代边界框优化机制（*iterative bounding box refifinement* mechanism）**以及**两阶段Deformable DETR（*two-stage Deformable DETR*）**来提高检测性能。

Figure 1展示了使用多尺度可变形注意力替换部分Transformer Attention模块

![image-20221206145600928](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221206145600928.png)

**Method：**

1. **Deformable Attention**

   ![image-20221206154111640](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221206154111640.png)

   其中，

    z<sub>q</sub> 看作query，由 x 经过线性变换生成; q 是对应的索引; **p**<sub>q</sub>代表 z<sub>q</sub> 的位置（可理解成坐标）, ∆**p**<sub>mqk</sub>(由query经全连接层得到，是可学习的)是采样集合点相对于参考点的位置偏移。

   k 是采样的key的索引; Ω<sub>k</sub> 即所有的 k 集合（K << HW）; 

   m 代表是第几个注意力头部; Wm 是对注意力施加在value后的结果进行线性变换从而得到不同头部的输出结果，W<sub>m</sub>′ 用于将 x<sub>k</sub> 变换成value;

    A<sub>mqk</sub> 代表**归一化**的注意力权重（由query经全连接层得到）。

2. **Multi-scale Deformable Attention Module**

   ![image-20221206155535700](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221206155535700.png)

   **p**^<sub>q</sub>代表归一化到[0,1];

   φ<sub>*l*</sub> 代表将归一化的坐标映射到（re-scale）各个特征层去。这样，每个参考点在所有特征层都会有一个对应的(归一化)坐标，从而方便计算在不同特征层进行采样的那些点的位置。

3. **Deformable Transformer **

   这里的Transformer和DETR中的大体过程一致，最主要的区别在于用可变形注意力替代了Encoder中的自注意力（self-attention）以及Decoder中的交叉注意力（cross-attention）。

改进策略：

1. **Iterative Bounding Box Refinement**

   在这里需要注意两点：1. 各层的检测头部是不共享参数的；2. 校正后的bbox梯度会被阻断（detach），不会跨层传播。

2. **Two-Stage Deformable DETR**

   2-stage模式下，Encoder会输出一批proposals（并不是基于网络预测，而是像anchor一样计算出来的），boxes中心就是各特征点的中心，而宽、高的设置则与所在的特征层相关，base值设置为0.05这时的proposals相对于anchors的角色。

   然后，使用检测头部的分类分支对Encoder编码的特征（memory）进行预测，对应各个proposals的分类结果；同时使用回归分支也对编码特征也进行预测，得到相对于proposals(xywh)的偏移量，接着将偏移量加在proposals的中心坐标和宽、高上得到第一阶段预测的proposals。

   最后，取top-k分数最高的那批预测proposals作为Decoder的参考点。并且，Decoder的object query和query embedding都由参考点通过位置嵌（position embedding）来生成。

### 16_Open-Vocabulary DETR with Conditional Matching_ ECCV 2022 Oral_有代码

code：https://github.com/yuhangzang/OV-DETR

作者：[Yuhang Zang](https://arxiv.org/search/cs?searchtype=author&query=Zang%2C+Y), [Wei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+W), [Kaiyang Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+K), [Chen Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+C), [Chen Change Loy](https://arxiv.org/search/cs?searchtype=author&query=Loy%2C+C+C)

贡献：本文是第一篇将DETR端到端目标检测运用到open-vocabulary领域的。本文相较于之前该领域的工作，除了可以使用文本提示，本文还可以用示例图片作为提示。作者提出了框架OV-DETR，将学习目标表示为输入查询（类名class name或范例图像exemplar image）和相应对象之间的**二进制匹配目标**，经过训练后，OV-DETR可以检测任何给定其类名或范例图像的对象。

![image-20221206223639273](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221206223639273.png)

由于新类对象缺乏label，所以无法直接像DETR那样计算闭集损失，因此本文改造Transformer Decoder部分的输入为conditional inputs。

**Conditional Matching for Open-Vocabulary Detection**

1. **Conditional Inputs.**

   给定数据集中ground-truth annotation的 bounding box **b**<sub>i</sub>和类别名称 **y**<sub>i</sub> <sup>class</sup>，本文使用CLIP模型来产生相应的图像和文本emdedding：![image-20221207142726778](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221207142726778.png)由于经过CLIP，图像和对应文本embedding是已经对齐的，所以可以选用任意以上任意embedding作为query（本文以50%的概率随机选取)。

2. **Conditional Matching.**

   将上述得到的embedding经过一个全连接层变成与query q相同的维度，并与q相加。则类不可知的对象查询q转变为由类特定的q‘。

   ![image-20221207144116394](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221207144116394.png)

   实际训练中会碰到这么一种情况：如果图片里有两只皮卡丘，但是只在一个query，那么就会覆盖不到第二只皮卡丘。因此作者提出“feature cloning”的方法来解决这一问题。

   ![image-20221207144850073](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221207144850073.png)

   如图4(b)，在前向过程中，输入到DETR Decoder中的query个数总共就有N*R个。

3. **Optimization**

   ![image-20221208200137052](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221208200137052.png)
   
   其中L<sub>match</sub>为用Focal Loss衡量的匹配损失；L<sub>box</sub>由L<sub>L1</sub>和L<sub>GIOU</sub>构成；L<sub>embed</sub>为predict embedding e和conditional input embedding z（z<sup>text</sup>或z<sup>image</sup>）之间的L1损失，用于进一步提高检测性能。

# 20221216

### 17_On Hyperbolic Embeddings in Object Detection_CVPR2022_无代码

作者：[Christopher Lang](https://arxiv.org/search/cs?searchtype=author&query=Lang%2C+C), [Alexander Braun](https://arxiv.org/search/cs?searchtype=author&query=Braun%2C+A), [Abhinav Valada](https://arxiv.org/search/cs?searchtype=author&query=Valada%2C+A)

贡献：现有目标检测主要基于欧氏空间或球体几何距离来衡量图像区域与对象类别原型的相似性。本文研究了双曲几何是否会更加适合当前的对象分类空间的架构。本文在现有 two-stage、keypoint-based以及transformer-based目标检测框架下，分别加入了双曲分类器，并在large-scale、long-tail以及zero-shot目标检测benchmark下测试其性能。实验结果表明，本文的方式能够降低分类错误并且提高目标检测的整体性能。

![image-20221211175624532](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221211175624532.png)

方法：

**Hyperbolic Embeddings**

n维双曲空间（*n*-dimensional hyperboloid model H<sup>*n*</sup>）定义为$H^n=\{x\in R^{n+1}:<x,x>_l=-1,x_0>0\}$，其中，$g_l(x)=diag(-1,1,...,1) \in R^{n+1 x n+1}$，因此洛伦茨标量积（Lorentzian scalar product )可表示为:
$$
<x,y>_l=-x_0y_0+\displaystyle \sum_{i=1}^{n}x_ny_n, x,y \in R^{n+1}

\$$
$$
用指数映射（exponential map）将从欧氏空间模型中提取的visual feature **v**∈ R <sup>*n*+1</sup>转换到Hyperbolic 中的点:
$$
exp_0^k(v)=sinh(||v||)\frac{v}{||v||}
$$
Hyperbolic 中两点之间的距离表示为：
$$
d_{H^n}=arccosh(-<x_i,t_c>_l), x_i,t_c \in H^n
$$
**Focal Loss Integration**  

Since distance metrics only span the non-negative real numbers, we compute logits by shifting the distances by a negative value as：
$$
s_{i,c}=\Delta-\frac{\Delta}{d_{min}}d_{i,c}
$$
其中s<sub>i,c</sub>是proposal i 和class c之间的分类得分； d<sub>i,c</sub>是转换后视觉特征向量与类别 c 原型的距离；*∆*是激活函数的偏移量；d<sub>min</sub>是一个标量超参数（定义为决定分类置信度p=0.5的距离；对于固定的类别原型，设置为小小类间距离；对于可学习的类别原型，设置为一个标量常量d<sub>min</sub>=1）；

**Generalization to Existing Object Detectors**

将hyperbolic classification head分别结合到two-stage、keypoint-based以及transformer-based目标检测框架。

实验：

在COCO 2017 *test-dev*和LVIS v1 benchmark上评估了本文方法的2D目标检测性能；

在 zero-shot 检测任务中（COCO 2017 dataset using the *65/15* classes split）评估了本文方法视觉——语义的映射性能（ visual-to-semantic mapping performance）

### 18_Open World DETR: Transformer based Open World Object Detection_暂未发布

作者：[Na Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+N), [Yongqiang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Mingli Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+M), [Gim Hee Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+G+H)

贡献：多buff叠加涨点

作者总结OWOD要克服的3个挑战：

1. 需要在无监督的情况下为未知类生成高质量的proposals；
2. 需要防止模型将未知类作为背景类（闭集中的方式），即从背景类中区分出unknown classes；
3. 需要避免模型将未知类识别为已知类别，即从已知类中区分出未知类。

方法：

基于*Deformable DETR*的two-stage训练方法：*Open World DETR*

**第一阶段：Model Pre-training Stage**

基于当前标注的数据，预训练一个模型（ the modified Deformable DETR，modified by adding a class-agnostic binary classification head）来检测已知类；并同时训练一个而二分类器（binary classifier）来区分前景和背景，这有利于模型构建无偏特征表示，以便后续对未知类的检测；

![image-20221214175206643](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221214175206643.png)

第一阶段匹配损失函数：

$L_{hg}^{bin}(y,y^*,\hat{y})=\sum_{i=1}^N[L_{cls}(c_i),\hat{c}_{\hat{\sigma}(i)}+1_{\{c_i\neq\emptyset\}}L_{box}(b_i,\hat{b}_{\hat{\sigma}(i)})+\lambda_{b\_{cls}}L_{b\_{cls}}(c_i^*,\hat{c}_{\hat{\sigma}^*(i)}^*)]$

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221214162825719.png" alt="image-20221214162825719" style="zoom: 33%;" />带*的表示二分类部分<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221214163205772.png" alt="image-20221214163205772" style="zoom: 33%;" />

训练时，将与已知类GT匹配的predictions作为前景类（label=0），否则作为背景类（label=1）

蒸馏损失：

$L_{feat}^{kd}=\frac{1}{2N}\displaystyle\sum_{i=1}^w\sum_{j=1}^h\sum_{k=1}^c(1-mask_{ij})||f_{ijk}^{cur}-f_{ijk}^{pre}||^2$

$L_{cls}^{kd}=L_{kl\_div}(log(p^{cur},p^{pre}))$

**第一阶段总损失：**
$$
L_{overall}^{pt}=L_{hg}^{bin}(y,y^*,\hat{y})+\lambda_{feat}L_{feat}^{kd}+\lambda_{cls}L_{cls}^{kd}
$$
**第二阶段：Open World Learning Stage**

使用多视角自标记策略（multi-view self-labeling strategy）和一致性约束方法（consistency constraint）来对模型的class-specific部分（projection layer, classification head and the class-agnostic binary classification head）进行微调；

多视角自标记策略利用预先训练的类不可知二值分类器（class-agnostic binary classifier）和选择性搜索算法（SS）生成的未知类的伪目标对这些组件进行精细细化；

一致性约束通过数据增强的自正则化（ self-regularization with data augmentation），提高了表示的质量。

进一步，当unknown classes 的标签增量地加进来的时候，通过知识蒸馏以及样例回放的方法来缓解灾难性以往问题。

![image-20221214163743301](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221214163743301.png)

- **Multi-view self-labeling：**

为了解决未知类缺乏**标签**的问题，本文提出多视角自标注的策略来为未知类创建伪GT。

首先对图像 $I$ 进行random crop & resize得到$I^{aug}$， $I$ 与对应的$I^{aug}$N个预测结果分别为$\hat{y}=\{\hat{y_i}\}_{i=1}^N=\{(\hat{c_i},\hat{b_i})\}_{i=1}^N$ 和$\hat{y}^{aug}=\{\hat{y_i}^{aug}\}_{i=1}^N=\{(\hat{c_i}^{aug},\hat{b_i}^{aug})\}_{i=1}^N$ 

通过class-agnostic binary classifier将得分最高且不与已知类的GT overlap的框作为未知类的label。

$I$ 用来为$I^{aug}$生成伪GT  $y'$ ，$I^{aug}$用来为 $I$ 生成伪GT $y'^{aug}$

那么对于$I$，联合训练标签为$y_u=[y,y'^{aug}]$；对于$I^{aug}$，联合训练标签为$y_u=[y,y']$；

训练时，将与**已知类GT匹配** & **与未知类伪GT匹配** 的predictions 作为前景类（label=0），否则作为背景类（label=1）

- **Supplementary pseudo proposals**：（SS）

进一步使用selective search algorithm产生额外的proposals作为未知类的补充伪标签。

具体来说，通过选择性搜索生成的与当前已知类的GT不重叠，也不与二值分类器生成的伪GT重叠的proposals，作为对未知类的额外伪proposals。

- **Consistency constraint**

为了提高特征表示的质量，在$I$ 和$I^{aug}$的object query features之间（与已知类匹配的object query features 以及  由SS算法生成的伪GT之间）做一致性约束。

损失：$L_{con}(\hat{q},\hat{q}^{aug})=\sum_{i=1}^Nl_1(\hat{q}_{\hat{\sigma}(i)},\hat{q}_{\hat{\sigma}^{aug}(i)}^{aug})$，其中$q$为decoder的输出。

第二阶段损失函数：

$L_{total}=L_{hg}^{bin}(y_u,y_u^*,\hat{y})+L_{hg}^{bin}(y_u^{aug},y_u^{*aug},\hat{y}^{aug})+\lambda_{con}L_{con}(\hat{q},\hat{q}^{aug})$

$L_{hg}^{bin}(y_u,y_u^*,\hat{y})$和$L_{hg}^{bin}(y_u^{aug},y_u^{*aug},\hat{y}^{aug})$分别是$I$ 和$I^{aug}$相应的匹配损失。

- **Alleviating catastrophic forgetting**

只是蒸馏+样本回放 

蒸馏损失：

$L_{feat}^{kd}=\frac{1}{2N}\displaystyle\sum_{i=1}^w\sum_{j=1}^h\sum_{k=1}^c(1-mask_{ij})||f_{ijk}^{cur}-f_{ijk}^{pre}||^2$

$L_{feat}^{kd,aug}=\frac{1}{2N}\displaystyle\sum_{i=1}^w\sum_{j=1}^h\sum_{k=1}^c(1-mask_{ij}^{aug})||f_{ijk}^{cur,aug}-f_{ijk}^{pre,aug}||^2$

$L_{cls}^{kd}=L_{kl\_div}(log(p^{cur},p^{pre}))$

$L_{cls}^{kd，aug}=L_{kl\_div}(log(p^{cur,aug},p^{pre,aug}))$

**第二阶段总损失：**
$$
L_{overall}^{owl}=L_{total}+\lambda_{faet}L_{feat}^{kd}+\lambda_{cls}L_{cls}^{kd}+\lambda_{faet}^{aug}L_{feat}^{kd,aug}+\lambda_{cls}^{aug}L_{cls}^{kd,aug}
$$

# 20221223

### 19_PyramidCLIP: Hierarchical Feature Alignment for Vision-language Model Pretraining_2022_有代码

code：https://github.com/Yuting-Gao/PyramidCLIP

作者：[Yuting Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+Y), [Jinfeng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Zihan Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Z), [Jun Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Ke Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+K), [Rongrong Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+R), [Chunhua Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+C)

贡献：本文提出了金字塔CLIP（PyramidCLIP），在ImageNet zero-shot分类任务上，在143M图像-文本对上训练的PyramidCLIP-ResNet50，超过了使用400M数据训练的CLIP的性能。

方法：

现在的大规模视觉-语言预训练（Large-scale vision-language pre-training，VLP）都基于一个假设，即从互联网上抓取的图像-文本对是完全一对对应的。然而，在实际场景中，这一假设很难成立。通过对图像的关联元数据进行爬取获得的文本描述通常存在**语义不匹配**（semantic mismatch）和**相互兼容性**（mutual compatibility）问题。如下图Figure1.

![image-20221218223811841](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221218223811841.png)

<p>Figure 1: Problems in the web-crawled image-text pairs.
(a)(b)(c) suffer the semantic mismatch between visual modality and linguistic modality, while (d) shows an example of the mutual compatibility with (a). Note that, in (a) the red caption is redundant; in (b) the image outside the red bounding box is the redundant; in (c) the descriptions for the casts in the red boxes are missing; and in (d) the red caption is compatible with the image of (a).</p>

为了解决上述问题，本文提出了金字塔CLIP（PyramidCLIP）。通过同层次语义对齐（via peer-level semantics alignment）和跨层次的关系对齐（ cross-level relation alignment），（这两者处理语义不匹配问题），达到以层次的形式实现视觉和语言的对齐效果；此外，软化负样本损失来调整目标损失函数，（处理相互兼容问题），以削弱预训练阶段的严格约束，从而降低模型过度约束的风险。

总体架构：

![image-20221218224535862](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221218224535862.png)

 **Peer-level Semantics Alignment**（同层次语义对齐）

- 粗粒度的全局信息对比学习：$(v^g , l^s)$ , $L_{GS}$

- 细粒度的局部信息对比学习：$(v^l , l^t)$ , $L_{LT}$

  本文也尝试了$(v^r , l^a)$ 的对其，但是并没有带来额外的收益，因此没有使用这部分。

 **Cross-level Relation Alignment**（跨层次关系对齐）

为了进一步提高对齐精度，作者引入了图像中显著对象的ROI特征序列，以提供更多的监督。具体地说，给定一幅具有M个显著对象的图像I，作者使用预训练的对象检测器Faster R-CNN来提取每个对象区域的视觉语义。

$(v^g , l^a)$ , $L_{GA}$

$(v^r , l^s)$ , $L_{RS}$

$(v^l , l^a)$ , $L_{LA}$

$(v^r , l^t)$ , $L_{RT}$

 **Softened Objective Function**

对于一个batch中的N个图片-文本对$\{I_i,T_i\}_{i=1}^N$

以第一个损失 $L_{GS}$为例，对于第$i_{th}$对，归一化的vision-to-language similarity $p_i^v(G)=\{p_{ij}^v(G)\}_{j=1}^N$ 和language-to-vision similarity $p_i^l(T_S)=\{p_{ij}^l(T_s)\}_{j=1}^N$ 分别表示为：
$$
p_{ij}^v(G)=\frac{exp(sim(v_i^g,l_j^s)/\tau)}{\sum_{j=1}^{N}exp(sim(v_i^g,l_j^s)/\tau)}
$$

$$
p_{ij}^l(T_S)=\frac{exp(sim(l_i^s,v_j^g)/\tau)}{\sum_{j=1}^{N}exp(sim(l_i^s,v_j^g)/\tau))}
$$



其中$\tau$是可学习的温度参数（初始化为0.7）；sim()表示点积

在原来的方法中，使用one-hot标签（将positive pair标为1，其他为0）的方法没有考虑语义相容性问题，因此本文对ground-truth做如下修饰：

$\tilde{y}_{ij}^v(G)=(1-\alpha)y_i^v(G)+\alpha/(N-1)$ , $\tilde{y}_{ij}^l(T_S)=(1-\alpha)y_i^l(T_S)+\alpha/(N-1)$

因此$L_{GS}$为：
$$
L_{GA}=-\frac{1}{2N}\sum_{i=1}^{N}\sum_{j=1}^{N}(\tilde{y}_{ij}^v(G) \cdot log(p_{ij}^v(G))+\tilde{y}_{ij}^l(T_S) \cdot log(p_{ij}^v(T_S)))
$$
总体目标损失函数$L$为：
$$
L=(1-\lambda-\mu)L_{peer}+\lambda L_{cross}^{global}+\mu L_{cross}^{local}
$$
其中$L_{peer}=(L_{GS}+L_{LT})/2$ , $L_{cross}^{global}=(L_{GA}+L_{RS})/2$ , $L_{cross}^{local}=(L_{LA}+L_{RT})/2$ , $\lambda = \mu = 1/3$

### 20_DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pre-training for Open-world Detection_NeurIPS 2022_无代码

作者：[Lewei Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+L), [Jianhua Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+J), [Youpeng Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen%2C+Y), [Xiaodan Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+X), [Dan Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+D), [Wei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Zhenguo Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Chunjing Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+C), [Hang Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+H)

贡献：1）提出以并行输入处理多数据源物体 - 文本联合训练的框架，优化训练效率；2）构建一个额外的物体知识库辅助开放域检测训练。

随着使用基于网上爬取的图片文本对训练的多模态预训练模型 (如 CLIP) 的流行，以及其在 zero-shot 分类领域体现出的卓越性能，越来越多的方法尝试将这种能力迁移至开放域的 dense 预测 (如任意类别检测、分割等)。现有方法往往使用预训练好的分类大模型进行特征层面的蒸馏，或通过对 caption 打伪标签加自训练的方式进行学习，但这样往往会受限制于分类大模型的性能以及 caption 标注不完全的问题。

现有 SOTA 开放域检测模型 GLIP通过将检测数据转化为 Grounding 数据进行多数据源的联合训练，充分利用不同数据源的优势(检测数据集对常见类别有较为完全的标注，而 Grounding 数据集对类别 cover 区间的范围更大)。然而，本文发现将类别名词拼接的方式导致模型整体的学习效率降低，同时直接使用类别单词作为文本输入无法提供细粒度的类别之间的先验关系。

因此本文提出DetCLIP，一种通过从 a designed concept dictionary中获取丰富知识，并行视觉概念的用于开放世界的检测预训练方法。

模型框架：

![image-20221221201519584](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221221201519584.png)

如上图所示，基于 ATSS单阶段检测模型搭建，DetCLIP 包含了一个图像编码器，用于获得检测框的图像特征，以及一个文本编码器

，用于获得类别的文本特征。然后基于这些图像特征及文本特征来计算对应的分类对齐损失、中心点损失以及回归损失。
$$
F^I=\Phi_i(X), F^T=\Phi_i(P^*), S=<F^I,Transpose(F^T)> \tag{1}
$$

$$
L=L_{ALI}(S,G)+\alpha L_{CEN}+\beta L_{REG} \tag{2}
$$

**Paralleled Concept Formulation**

![image-20221221195928279](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221221195928279.png)

相对于 GLIP 中将 detection 数据通过拼接类别名词的方式转化为 grounding 形式(串行)，本文通过将 grounding 数据中的对应名词词组抽取出来和 detection 中的类别作为独立的输入，输入到 text encoder 中(并行)，避免不必要的 attention 计算，实现更高的训练效率。

 **Concept Dictionary**

见上述Figure 5 (b)

- 构建（ **Constructing the Concept Dictionary**）

  综合检测数据中的类别、image-text pair 中的名词词组以及对应定义来构建物体知识库。

- 使用（**Knowledge Enrichment with Concept Dictionary**）

  1. **Concept Enrichment**

     使用物体知识库的定义对现有的检测数据中的类别单词进行扩充，以提供类别之间关系的先验信息

  2. **Partial Annotation Enrichment**

     由于 grounding 数据以及 image-caption 中数据存在 caption 标注不完全的问题(图片上出现的类别在 caption 中并没有出现)，导致训练这些图片的时候可以作为负样本的类别数目极少，进而使得模型对于一些不常见类别的区分度较少。因此我们从物体知识库中随机选取物体名词作为负样本类别，提升模型对稀少类别特征的区分度（+Negative Samples）。

# 20221230

### 21_Revisiting Open World Object Detection_2022_有代码

code：https://github.com/RE-OWOD/RE-OWOD

作者：[Xiaowei Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+X), [Xianglong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Yifan Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+Y), [Yixuan Qiao](https://arxiv.org/search/cs?searchtype=author&query=Qiao%2C+Y), [Yuqing Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+Y), [Duorui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+D)

贡献：

1. 对OWOD开篇之作ORE的工作指出几点设置的不合理，并提出OWOD应该遵循的5点原则，从unknow类的角度提出两个新的评价标准。
2. 提出了一个新的有效的OWOD框架，在Faster-RCNN的基础上增加了*an auxiliary Proposal ADvisor (PAD) and a Class-specifific Expelling Classififier (CEC)*。PAD（无参数）在没有监督的情况下帮助RPN识别准确的未知建议（ *unknown proposals*）；CEC通过一个特定于类的驱逐函数校准过度自信的激活边界，过滤掉令人困惑的预测

5点原则：

1. 类别的开放性（Class Openness）：训练类别只包含有标签的已知类，推理时碰到的类别包含已知类和未知类；

2. 任务的递增性（Task Increment）：未知类不断地被识别并作为新增的已知类

3. 标注的特异性（Annotation Specificity）：

   标签：Y = [ L , B ] 其中Y为类别，B为边界框
   训练、验证阶段：仅有已知类别参与训练
   测试阶段：未知类别也参与测试，L赋值为“Unknown”
   原OWOD未遵循此原则，使用了包含未知标签的验证集来训练

4. 标签的完整性（Label Integrity）

5. 数据的特异性（Data Specificity）

2个新的评价指标：

根据OWOD的定义，有三个关键的挑战：1)**Unknown Objectness**：从背景中区分一个未知的实例。2)**Unknown Discrimination**：区分一个未知的实例和一个类似的已知的类。3)**Incremental Conflflict**：在学习现有的已知类和新注释的已知类之间的平衡。

1. **UDR**(Unknown Detection Recall)：未知类的准确定位率
   $$
   UDR=\frac{TP_u+FN_u^*}{TP_u+FN_u}
   $$
   
2. **UDP**(Unknown Detection Precision)：所有定位到的未知实例的准确分类率
   $$
   UDP=\frac{TP_u}{TP_u+FN_u^*}
   $$
   

   （$$TP_u$$：未知类预测为未知类 $$FN_u$$：未知类预测为背景 $$FN_u^*$$：未知类预测为已知类）

框架：

![image-20221228093613707](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221228093613707.png)

PAD模块：未知样本标签的缺失，导致RPN很难生成未知样本的proposal，因此本文添加了PAD，辅助RPN针对此类proposal生成。

- 对RPN的未知proposal结果进行再确认

$$
\overline{S_i}=S_i * I\{\underset{1\leq j\leq |\tilde{P}^+|}{max}(IOU(P_i^{(u)+},\tilde{P}_j^+)>\theta\}, \tag{4}
$$

​	objectness scores **S**, $$I(·)若条件满足取1，否则取0$$，IOU 大于阈值，表示advisor和RPN均认为该区域是positive proposal

- 指导未知类别定位任务的无监督训练过程

  获取到精确的proposal后，将对应类别确认改为“前景”，将对应Anchors从negative anchor set中移动到positive anchor set中
  将新的anchor set 输入到RPN的分类器中，计算损失：
  $$
  L_{RPN}^{cls}=\sum_{a \in A^+ \cup A^{(u)+}}BCE(f(a),1)+\sum_{a \in A^- \A^{(U)+}}BCE(f(a),0), \tag{5}
  $$
  BCE: Binary Cross Entropy loss

CEC模块：将那些，被预测为已知的某类的未知类别（且赋予较高的置信度）的，进行调整

- 计算驱逐指标
  $$
  \Phi(\overline{L}_i^c)=\overline{L}_i^c-\alpha \frac{1}{M}\sum_{j}^{|\tilde{B}|}\sum_{k}^{|B^c|}[I(IOU(\tilde{B_j},B_k^c)>\phi)*\tilde{L_i^c}] \tag{6}
  $$

- 对已知类别重新分配预测别类
  $$
  \overline{L}_i^{c^,}=I(\Phi(\overline{L}_i^c)>0)*\overline{L}_i^c
  $$

超参数设置：We respectively select top-50 potential unknown proposals from RPN and top-50 auxiliary proposals from the auxiliary proposal advisor. 

*θ*, *ϕ* and α are empirically set as 0.7, 0.9 and 0.5.

### 22_Towards Open-Set Object Detection and Discovery_CVPRW 2022_无代码

作者：[Jiyang Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+J), [Weihao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+W), [Jie Hong](https://arxiv.org/search/cs?searchtype=author&query=Hong%2C+J), [Lars Petersson](https://arxiv.org/search/cs?searchtype=author&query=Petersson%2C+L), [Nick Barnes](https://arxiv.org/search/cs?searchtype=author&query=Barnes%2C+N)

贡献：在之前的 open-set object detection (OSOD) 中，除了检测识别已知物体外，还会检测一些未知类别的物体，但把所有未知的物体都归到 “未知类”。本文提出的 Open-Set Object Detection and Discovery (OSODD)，不仅可以检测未知类，还可以发现它们潜在的类别。OSODD 提出了一种两阶段的方法，首先使用开放集对象检测器来预测已知和未知对象。然后，以无监督的方式研究预测对象的表示，并从未知对象的集合中发现新的类别。

不同设定下的检测效果对比：

![image-20221228105226158](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221228105226158.png)

框架： OSODD 包含两个部分，分别是 Object Detection and Retrieval (ODR) 和 Object Category Discovery (OCD)

- ODR 是一个带有两个记忆缓存的开集检测器，对于已知物体，检测器预测他们的位置信息和类别，对于未知物体，只预测其位置信息。其中已知物体和类别信息储存在 known memory 中，未知物体则储存在 working memory 中。
- OCD 则是主要利用 working memory 来发现未知物体的类别，包含了一个特征编码器和聚类辨别器。首先使用非监督对比学习方式，从 known 和 working memory 中训练一个编码器，在 latent space 中学习更好的物体表征。最后用 constrained k-means 来进行聚类。

![image-20221228105502843](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221228105502843.png)

Object Detection and Retrieval：

开放集目标检测器主要是对所有物体进行定位，同时对已知物物体进行分类，且把未知物体归到“unknown” 一类。本文使用 Faster-RCNN 作为模型的 backbone，利用了 RPN 对类别无感知的特性，把那些与 ground-truth 没有重叠且置信度比较高的候选框作为位置物体。为了让物体的特征更具有区别性，作者使用了对比损失，也就是计算从 ROI pooling 中得到的特征和类别原型( class prototype)之间的距离：
$$
l_{pcl}(f_c)=\sum_{i=0}^cl(f_c,p_i) \\
l(f_c,p_i)=
\begin{cases} 
||f_c,p_i||\quad\quad\quad\quad\quad\quad\quad\quad \quad\quad  if\quad i=c \\ \\
max(0,\Delta-||f_c,p_i||)\quad\quad\quad\quad otherwise
\end{cases}
\tag{1}
$$
其中$p_i$是移动平均的类别原型，$f_c$为特征向量；

从而，ROI head的损失变为：$l_{roi}=\alpha_{pcl}·l_{pcl}+\alpha_{cls}·l_{cls}+\alpha_{reg}·l_{reg}$

Object Category Discovery：

因为未知物体的类别是不确定的，本文采用 DCT，通过一种特殊的无参数学习的 k-mean 来估计潜在的类别数目。
为了更好地发现未知物体的潜在类别，作者在 OCD 中加入了一个 encoder，用来学习更有判别性的 embedding。在encoder 中使用 known memory 和 working memory中的对象来进行对比学习，增大 positive pairs 的相似度，而减小 negative pairs的相似度，类似减小类内差而增大类间差，这样更有益于后面的聚类操作。

# 20230106

### 23_Simple Open-Vocabulary Object Detection with Vision Transformersn_ECCV 2022_有代码

code：https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit

作者：[Matthias Minderer](https://arxiv.org/search/cs?searchtype=author&query=Minderer%2C+M), [Alexey Gritsenko](https://arxiv.org/search/cs?searchtype=author&query=Gritsenko%2C+A), [Austin Stone](https://arxiv.org/search/cs?searchtype=author&query=Stone%2C+A), [Maxim Neumann](https://arxiv.org/search/cs?searchtype=author&query=Neumann%2C+M), [Dirk Weissenborn](https://arxiv.org/search/cs?searchtype=author&query=Weissenborn%2C+D), [Alexey Dosovitskiy](https://arxiv.org/search/cs?searchtype=author&query=Dosovitskiy%2C+A), [Aravindh Mahendran](https://arxiv.org/search/cs?searchtype=author&query=Mahendran%2C+A), [Anurag Arnab](https://arxiv.org/search/cs?searchtype=author&query=Arnab%2C+A), [Mostafa Dehghani](https://arxiv.org/search/cs?searchtype=author&query=Dehghani%2C+M), [Zhuoran Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+Z), [Xiao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Xiaohua Zhai](https://arxiv.org/search/cs?searchtype=author&query=Zhai%2C+X), [Thomas Kipf](https://arxiv.org/search/cs?searchtype=author&query=Kipf%2C+T), [Neil Houlsby](https://arxiv.org/search/cs?searchtype=author&query=Houlsby%2C+N)

Google Research

贡献：本文利用基于Transformer模型的可扩展性以及其在闭集目标检测领域的成功，构建了一个两阶段、简单且可扩展的开放词汇目标检测模型。

首先，在大规模图像-文本数据上预训练图像和文本编码器；

然后，添加检测头，并在检测数据上进行微调。

推理时，模型可以根据不同方式的查询，执行开放词汇表目标检测或few-shot目标检测。

![image-20230106154947978](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230106154947978.png)

模型结构：

本文的模型使用标准的 Vision Transformer 作为图像编码器，并使用类似的 Transformer 架构作为文本编码器（如上如）。为了使图像编码器适应检测，本文删除了token池和最终投影层，而是线性投影每个输出token表示以获得用于分类的每个对象图像嵌入（如上图右）。因此，预测对象的最大数量等于图像编码器的token数（序列长度）。这在实践中不是瓶颈，因为模型的序列长度至少为 576（输入大小为 768 × 768 时的 ViT-B/32），这大于当今数据集中的最大实例数（例如LVIS中294 个实例）。边框坐标是通过 MLP(token表示)来获得的。本文的设置类似于 DETR，但通过删除解码器进行了简化。

### 24_Graph Few-shot Class-incremental Learning_ WSDM 2022_有代码

code：https://github.com/Zhen-Tan-dmml/GFCIL

作者：[Zhen Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+Z), [Kaize Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+K), [Ruocheng Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+R), [Huan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+H)

贡献：本文提出新颖的图少样本增量学习问题（ Graph Few-shot Incremental Learning problem），并将其表示为对节点的分类任务。提出专为 Graph FCL设计的层次图注意力模块（Hierarchical Graph Attention modules），并设计了一个图伪增量学习范式（Graph Pseudo Incremental Learning paradigm），使有效的训练能够模拟在评估阶段环境。

问题陈述：

对于一张图，表示为$G=(V,\epsilon ,X)$，其中，$V,\epsilon ,X$分别为节点集，边集和节点特征，图也可方便的表示为$G=(A ,X)$，其中A为邻接矩阵。Graph FCL任务假定同质数据集：$D=\{D^0,D^1,...,D^i,...,D^T\}$，对于不同的session i和session j，$D^i,D^j$分别对应的标签空间$C^i,C^j$关系为$C^i\cap C^j = \emptyset$。值得注意的是，第一个session的$D^0$是相当大的数据集，有充足的训练数据，即base类；对于$D^i \in D,i \neq 0$是少样本数据集，即novel 类。

基于图的少样本类增量节点分类：对于特定的session i，给定图$G=(A ,X)$以及带标签的支持节点$S^i$，模型的任务就是根据相应的querry set $Q^i$预测其标签。 $Q^i$的标签空间包括base set $C^0$，先前session 的novel sets $\{C^1,C^2,...,C^{i-1}\}$以及当前session的novel set $C^i$。对每个session i，少样本类增量节点分类任务$T^i$,对于给定的支持集$S^i$，novel类的数量为N，每个类别中支持节点的数量为K，则该任务称为N-way K-shot增量节点分类任务。$T=\{T^0,T^1,...,T^i,...,T^T\}$。

方法：**HAG-Meta**（a Hierarchical Attention Graph Meta learning framework）

4.1 图伪增量学习（**Graph Pseudo Incremental Learning**）

数据划分：为解决Graph FCL问题，本文将数据集$D$划分为两个部分$D_{base},D_{novel}$。$D_{base}$随机划分为$D_{base/tr},D_{base/val},D_{base/test}$用于预训练；$D_{novel}$划分为$D_{novel/tr},D_{novel/val},D_{novel/test}$。为了模拟Graph FCL问题，目标few-shot数据$\{D^1,...,D^i,...,D^T\}$采样自$D_{novel/test}$，相应的节点和边在预训练和元学习阶段是被masked的。base数据$D^0$由$D_{base}$和$D_{novel/tr}$组成。

预训练：在$D_{base/tr}$上预训练一个基于GNN的encoder $g_{\theta}$，其在预训练之后仍然是可训练的。

4.2 模型

![image-20230102233204743](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230102233204743.png)

本文模型基于原型网络（Prototypical Network，PN）， 将其中的MLP替换为GNN encoder $g_{\theta}$。对于给定的图$G=(A ,X)$，潜层特征表示为：
$$
Z=g_{\theta}(A ,X) \tag{1}
$$
类别原型用该类别节点的平均值表示：
$$
p_k=\frac{1}{|S_k|}\sum_{j \in S_k}z_j \tag{2}
$$
$S_k$代表类别k的支持节点。

对于query 节点$v_q$，（其中d为平方欧氏距离）
$$
p(y=k|q)=\frac{exp(-d(z_q,p_k))}{\sum_{k}exp(-d(z_q,p_k))} \tag{3}
$$
为了应对类别不平衡问题，本文提出了基于层级注意力的图注意力模块，分别为*Task-Level Attention*和*Node-Level Attention*，目标是学习一个强大的规则化器，它可以动态地缩放不同任务中节点的贡献。

**Task-Level Attention：**

为了应对模型可能会过度拟合到基础类或新类上的挑战，本文提出任务级注意（TLA）来估计在不同任务中学习到的类的重要性。即学习一系列损失的比例因子，这些因子应该能够自动降低元训练中简单或无关紧要的任务的贡献，并迅速将模型集中在困难或重要的任务上。

对于FN网络，在特定session i中，支持节点$S^i$的原型矩阵$P^i=\{p_k^i\}$即为query  $Q^i$的分类器，因此本文假设原型即能表示task $T^i$的知识，TLA学习当前任务 $T^i$与所有模型已经训练过的$\{T^1,...,T^i\}$之间的注意力$w^{i,j}$.

由于base类的数量过多，因此先用MLP层将所有任务中的原型投射到相同的size：$u^j=MLP(p^j),\forall j \in [1,i]$

权重w为$w^{i,j}=\frac{exp(u^i,u^j)}{\sum_{j=1}^{i}exp(u^i,u^j)}$

对于同一task中的所有类别，共享相同的权重因子即：$W_C=\frac{\tilde{W}}{|C^i|}$

TLA损失：$L_{TLA}=\sum_{k \in C}w_k*[y_klog(\hat y_k)+(1-y_k)log(1-\hat y_k)]$

最终损失为：$L=L_{CEL}+L_{TLA}=\sum_{k \in C}(1+w_k)*[y_klog(\hat y_k)+(1-y_k)log(1-\hat y_k)]$

 **Node-Level Attention:**

NLA为每个节点保持现有知识和新知识的平衡。NLA学习$\Lambda = \{\lambda _j\},j \in S^i$来调整session i中GNN encoder学习到的原型特征。

在GCN第l层的传播规则为：
$$
h_j^l=\sigma′(R^l(h_j^{l-1}+\sum_{j′ \in N_j}h_j^{l-1}/\sqrt{d_jd_j^′}))
$$
在GCN的最后一层，使用MLP计算$\lambda$: $\lambda _j=MLP(h_j^L)$

然后$\tilde{\lambda _j}=\sigma (log(degree(v_j)+\epsilon )\lambda_j)$

归一化：$\lambda _j=\frac{exp(\tilde{\lambda _j})}{\sum_{j \in S^i}exp(\lambda _j)}$

然后（2）式的原型公式修改为：$p_k=\frac{1}{|S_k|}\sum_{j \in S_k}\lambda _jz_j$，然后就可以使用式（3）来获得最终的标签。

# 20230113

### 25_BOIL: Towards Representation Change for Few-shot Learning_ICLR 2021_有代码

code: https://github.com/jhoon-oh/BOIL

作者：[Jaehoon Oh](https://arxiv.org/search/cs?searchtype=author&query=Oh%2C+J), [Hyungjun Yoo](https://arxiv.org/search/cs?searchtype=author&query=Yoo%2C+H), [ChangHwan Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+C), [Se-Young Yun](https://arxiv.org/search/cs?searchtype=author&query=Yun%2C+S)

贡献：MAML (Model Agnostic Meta-Learning)是最具有代表性的基于梯度的元学习算法之一，最近的工作(ANIL)认为，相比于表征改变(representation change)，表征重用(representation reuse)是通过MAML元初始化模型的主要性能因素。本文针对少样本学习的终极目标——实现域未知任务，研究了表征改变的必要性。提出新的元学习算法——BOIL(Body Only update in Inner Loop),在内部循环中，只更新body（特征提取器），冻结head（分类器），在跨领域任务上，BOIL相较于MAML表现出重大性能提升。

表征重用(representation reuse)：在有效表征上几乎不做改变

表征改变(representation change)：在表征上做较大改变

如图：MAML、ANIL、BOIL在内层循环中分别更新模型不同的部分示意图

![image-20230108095618868](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230108095618868.png)

本文的贡献总结如下：

通过跨领域适应实验，强调元学习中表征变化的必要性；

提出BOIL元学习算法，在大多数基准数据集上提高了性能，本文的这种改进在细粒度数据集或跨域自适应中尤其明显；

解释了BOIL和使用梯度预处理算法的联系；

利用余弦相似度(cosine similarity)和核心对齐（CKA），证明了BOIL算法在body的低/中层上具有表示层重用，在高层上的表现为表征变化；

对于ResNet架构，提出断开连接的技巧（删除最后一次跳跃连接的反向传播路径），增强了body高层的特征变化。

MAML的更新算法：

内层循环（基于任务的）：
$$
\theta _{\tau {_i}}=\theta - \alpha \Delta _{\theta}L_{S_{\tau _i}}(f_\theta) \tag{1}
$$
外层循环：
$$
\theta '= \theta - \beta \Delta _{\theta}L_{meta}(\theta),其中L_{meta}(\theta)=\sum_{i=1}^B L_{Q_{\tau_{i}}} (f_{\theta _{\tau _i}}) \tag{2}
$$
BOIL的更新算法：将元初始化的$\theta$分成两部分：$\theta = \{\theta _b, \theta _h\}$，分别表示body参数与head参数，且内层循环参数更新变为：
$$
\theta {_b,\tau _i}=\theta _b - \alpha _b \Delta _{\theta_b}L_{S_{\tau _i}}(f_\theta) \\  \& \\
\theta {_h,\tau _i}=\theta _h - \alpha _h \Delta _{\theta_h}L_{S_{\tau _i}}(f_\theta)
\tag{3}
$$
学习率设置：

|             |         MAML         |   ANIL   |   BOIL   |
| ----------- | :------------------: | :------: | :------: |
| $\alpha _b$ | $= \alpha _h \neq 0$ |    =0    | $\neq 0$ |
| $\alpha _h$ | $= \alpha _b \neq 0$ | $\neq 0$ |    =0    |

### 26_CAT: LoCalization and IdentificAtion Cascade Detection Transformer for Open-World Object Detection_有代码

code：https://github.com/xiaomabufei/CAT

作者：[Shuailei Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Yuefeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Jiaqi Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+J), [Ying Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+Y), [Thomas H. Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+T+H), [Hongli Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+H), [Fanbing Lv](https://arxiv.org/search/cs?searchtype=author&query=Lv%2C+F)

贡献：

本文指出，当前利用标准目标检测框架+固定的伪标签方式的OWOD检测存在以下问题：1、检测未知对象大幅降低模型对已知对象的检测能力；2、基于已知类训练过程来引导未知类的伪标签生成，没有充分利用（纹理、光线、光流等）先验性息；3、由于伪标签的质量具有不确定性，伪标签的固定选择方式不能保证模型学会在正确的方向上检测未知对象。

由于在面对一个新的场景时，人类总是倾向于先定位到所有的前景对象，然后再分别识别它们，而不是同时定位+识别。受此启发，本文提出新的OWOD模型：CAT，定位与识别串联检测的Transformer。CAT由自适应伪标签机制、共享的transformer decoder以及串联解耦的decoding 架构组成。在unknown recall指标上，CAT(18.3%)比OW-DETR(11.8%)高出6.5%。

网络架构：

![image-20230109173650724](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230109173650724.png)

**Self-Adaptive Pseudo-labelling**：

本文通过结合模型驱动以及输入驱动的打伪标签方式来扩展模型知识的来源。

模型驱动伪标签：注意力驱动伪标签，负责产生伪标签候选框$P^m$和相应的置信度$s_o$

输入驱动伪标签：selective search，负责产生伪标签候选框$P^I$

所生成的伪标签object confidence表示为：
$$
S_i=(norm(s_o))^{W_m}.(\underset{1\leq j\leq |P^I|}{max}(IOU(P^I_j,P^m_i)))^{W_I} \tag{1}
$$
其中$W_m和W_I$为模型、输入驱动的自适应权重，由Measure、Sensor以及Adjuster函数约束；
$$
W^t=Adjuster(W^{t-1},Sensor(Measure(L_m))) \tag{2}
$$
$L_m$为训练过程中实时更新并存储的损失：$L_m=DEQUE(loss_{t-1},loss_{t-2},...loss_{t-n})$，$t$为当前迭代轮次；

考虑到数据质量的不稳定性以及模型的敏感性，使用$Measure$来获得损失的趋势$\Delta l$：
$$
Measure(L_m)=\frac{\sum_{i=1}^{n}\alpha _i.loss_{t-i}}{\sum_{j=n+1}^{N}\beta _j.loss_{t-j}},n < N < T, \tag{4}
$$
其中$\alpha 和 \beta$为加权平均权重，且满足$\sum_{i=1}^{n}\alpha _i = \sum_{j=n+1}^{N}\beta _j=\frac{\alpha _{i} - \alpha _{i-1}}{\alpha _{i+1} - \alpha _{i}}=\frac{\beta _{j} - \beta _{j-1}}{\beta _{j+1} - \beta _{j}}=1$；

通过$Sensor(.)函数获得\Delta w$，$\pi _{nma}和\pi _{pma}$分别为正负动量振幅，
$$
Sensor(\Delta l)=\begin{cases} 
\pi _{nma}.Sigmoid(\Delta l-1), \Delta l > 1, \\ \\
-\pi _{pma}.\Delta l, \Delta l \leq 1,
\end{cases}
\tag {5}
$$
$Adjuster(.)$遵循公式$(6)$通过增量的方式更新自适应权重，
$$
\begin{cases}
W_m^t=W_m^{t-1}+\Delta w *W_m^{t-1}, \\
W_I^t=W_I^{t-1}-\Delta w *W_I^{t-1}, \\
W_m^t, W_I^t=norm(W_m^t, W_I^t),
\end{cases}
\tag{6}
$$
权重更新方式如算法1所示：<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230109201543372.png" alt="image-20230109201543372" style="zoom:80%;" />

**Decoupled Decoding Structure**：

本文探讨了两种解耦decoding 架构的方式：即全解耦(fully decoupled decoding structure)以及串联解耦(cascade decoupled decoding) structure，如下图

**全解耦：**$F_S(·)代表共享decoder，F_e(·)代表encoder，\emptyset (·)代表backbone，P_n和P_m为positional encoding和embedding，\\R为参考点，J为输入图片，Q为相应queries$
$$
\epsilon _{Location}=F_S(F_e(\emptyset (J),P_n),P_m,Q_{Location},R), \tag{7}
$$

$$
\epsilon _{Class}=F_S(F_e(\emptyset (J),P_n),P_m,Q_{Class},R), \tag{8}
$$

**串联解耦：**
$$
\epsilon _{Location}=F_S(F_e(\emptyset (J),P_n),P_m,Q_{Location},R), \tag{9}
$$

$$
\epsilon _{Class}=F_S(F_e(\emptyset (J),P_n),P_m,\epsilon_{Location},R). \tag{10}
$$

![image-20230109202431123](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230109202431123.png)



代码进度

20230106

![image-20221230205937979](https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20221230205937979.png)

<table>
    <tr>
        <td><font size="0.5">Task IDs (→)</font></td>
        <td colspan="3"><font size="1">Task 1</font></td> 
        <td colspan="5"><font size="1">Task 2</font></td>
        <td colspan="5"><font size="1">Task 3</font></td> 
        <td colspan="5"><font size="1">Task 4</font></td> 
    </tr>
    <tr>   
      <td rowspan="2"> </td>
        <td rowspan="2"><font size="0.5">WI(↓)</font></td>
        <td rowspan="2"><font size="0.5">A-OSE(↓)</font></td>
        <td ><font size="1">mAP(↑)</font></td>
        <td rowspan="2"><font size="0.5">WI(↓)</font></td>
        <td rowspan="2"><font size="0.5">A-OSE(↓)</font></td>
        <td colspan="3"><font size="0.5">mAP(↑)</font></td>
        <td rowspan="2"><font size="0.5">WI(↓)</font></td>
        <td rowspan="2"><font size="0.5">A-OSE(↓)</font></td>
        <td colspan="3"><font size="0.5">mAP(↑)</font></td>
        <td colspan="3"><font size="0.5">mAP(↑)</font></td>
    </tr>
    <tr>
        <td><font size="0.5">Ck</font></td>
        <td><font size="0.5">Pk</font></td>
        <td><font size="0.5">Ck</font></td>
        <td><font size="0.5">Both</font></td>
        <td><font size="0.5">Pk</font></td>
        <td><font size="0.5">Ck</font></td>
        <td><font size="0.5">Both</font></td>
        <td><font size="0.5">Pk</font></td>
        <td><font size="0.5">Ck</font></td>
        <td><font size="0.5">Both</font></td>
    </tr>
    <tr>
        <td><font size="0.5">ORE</font></td>
        <td><font size="0.5">0.04988</font></td>
        <td><font size="0.5">6156</font></td>
        <td><font size="0.5">56.25</font></td>
        <td><font size="0.5">0.02985</font></td>
        <td><font size="0.5">7251</font></td>
        <td><font size="0.5">51.65</font></td>
        <td><font size="0.5">25.63</font></td>
        <td><font size="0.5">38.64</font></td>
        <td><font size="0.5">0.0205</font></td>
        <td><font size="0.5">5749</font></td>
        <td><font size="0.5">37.57</font></td>
        <td><font size="0.5">12.55</font></td>
        <td><font size="0.5">29.23</font></td>
        <td><font size="0.5">29.86</font></td>
        <td><font size="0.5">13.23</font></td>
        <td><font size="0.5">25.70</font></td>

CK: Current Known

PK: Previously Known

WI指标，根据论文所述取值均为 measured at a recall level *R* (0.8 in all experiments)，但是都比论文结果要偏高；A-OSE反而偏低。

mark：test4结果是在命令行直接运行（1张卡）跑出来的，nohup（2张卡）貌似会遇到缺图片的error（2次）；（后面nohup跑又成功了）怀疑图片有问题（20230102）。



















biji

# 20230324



### 13_Learning to Decompose Visual Features with Latent Textual Prompts_ICLR 2023_无代码

作者：Feng Wang1 , Manling Li2 , Xudong Lin3 , Hairong Lv1 , Alexander G. Schwing2 & Heng Ji2

​			1Tsinghua University 2University of Illinois at Urbana-Champaign 3Columbia University

> 贡献：

针对 CLIP 类模型存在的两个问题：1、在 zero-shot 范式下，通过检索文本类名进行推断时，准确性和鲁棒性会降低；2、在 linear probing 范式下，会打破 well-established 视觉语言对齐。本文提出 分解特征提示 DeFo（**De**composed **F**eature Pr**o**mpting），基于 CLIP 双模型架构，通过**可学习的嵌入作为文本输入**并添加一个**额外的线性层**来进行分类，DeFo 能够在文本提示的帮助下提取到**分解的视觉特征**  decomposed visual features 。此外，DeFo 支持可变大小的语言输入。

DeFo在使用 ResNet-50 backbone 的 ImageNet 上获得了 73.2% 的测试精度，比 zero-shot CLIP 高15.0%，比 SOTA 的 vision-language prompt tuning 高7.6%。

**hard-target retrieval**：

CLIP 类模型在执行 zero-shot 推理时，直接计算从 image encoder 获得的 vectorial image representation 与从 language encoder 获得的 文本提示表示 之间的距离。与图像的表示向量距离最小的文本提示符对应的目标类构成 zero-shot 推理结果。

使用 hard textual targets 进行推理存在以下两个问题：

* **expressive sensitivity:** text prompt 中的类别名称无法准确地总结图像中的语义信息，这导致推理结果非常受到类别名称选择的影响。（如 "plane" vs "airplane"，"car" vs "automobile"）
* **conceptual sensitivity:** 尽管数以亿计的预训练样本覆盖了大量可能出现在下游数据集中的概念，但 zero-shot 推理仍然难以识别稀有物体。

因此，本文提出 DeFo ，将 CLIP 类模型的 硬目标检索范式 转化为 双模型特征提示：

* DeFo 为 language encoder 提供了一组独立于  hard semantic targets 的可学习的嵌入序列；
* 通过调优一个附加的额外线性层来执行分类。

> 方法

DeFo 致力于使用 language encoder 构建一个映射矩阵，将视觉特征从 CLIP 潜在空间的 $d$-维 映射到 $n$-维的特征空间。

其中只有**线性分类层 **和 **textual queries** 中的参数是可训练的。

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230315192850364.png" alt="image-20230315192850364" style="zoom:80%;" align="left"/> 

如图，DeFo 的构成：

​	1、visual encoder $g_{V} : R^{w×h×3}→R^{d}$ 

​			输入：$w×h×3$ 的图像

​			输出：$f_{I} ∈ R^{d}$

​	2、language encoder $g_{L} : R^{m×d_{e}}→R^{d}$

​			输入： $X_{L}∈R{n×m×d_{e}}$ ,n 个带有 $m$ 个单词的  query sentences，每个单词被嵌入到 $d_{e}$ 维的向量中

​			输出：$f_{T}^{1},f_{T}^{2}, . . . ,f_{T}^{n} ∈R_{d}$

​	3、通过将经ℓ2标准化的 $f_{I}$ 和每个经ℓ2标准化的 $f_{T}^{i}$ 做点乘，得到 $n-$ 维向量，第 $i$ 个元素即表示该图与第 $i$ 个 text query 的相似度。

​	4、通过一个线性层将 $n-$ 维投射到 $k-$ 维，并对$k-$ 维向量进行 softmax 计算 probabilities。$p_{i} = \frac{exp(⟨f_{I},f_{T}^{i}⟩)/τ}{\sum_{j=1}^{k}exp(⟨f_{I},f_{T}^{i}⟩)/τ}$



### 14_Learning Object-Language Alignments for Open-Vocabulary Object Detection_ICLR 2023_有代码

作者：Chuang Lin,Peize Sun,Yi Jiang,Ping Luo,Lizhen Qu,Gholamreza Haffari,Zehuan Yuan,Jianfei Cai

代码：https://github.com/clin1223/VLDet

>  贡献：

本文提出直接从图像-文本对 ( image-text pair ) 中学习 fine-grained 对象-语言 ( object-language ) （又称 region-word ）对齐，将其看作集合匹配任务，使用匈牙利匹配算法，训练一个端到端的 Open-Vocabulary 目标检测器。在 Open-vocabulary LVIS and Open-vocabulary COCO 数据集上获得了 SOTA 性能。

**Open-vocabulary Object Detection：**

开放词汇表目标检测致力于构建这样一个目标检测器：通过在**带有 base-class 边框标注信息的数据集** $C^{base}$ 和 **包含大量词汇的 image-caption 对的数据集** $C^{open}$ 上训练，使模型在测试阶段有能力检测 novel classes $C^{novel}$ 。

注： $C^{base}$ + $C^{novel}$ 有可能会、有可能不会和 $C^{open}$ 存在交叉。

> 框架

<img src="https://raw.githubusercontent.com/yuki1ssad/typora_images/main/image-20230316190316363.png" alt="image-20230316190316363" style="zoom:80%;" />

**通过 Bipartite Matching 学习 Object-Language 对齐：**

注：object <--> region  language <--> word

二分图匹配描述的是 **X** 个 workers 和 **Y** 个 jobs 之间的分配问题

* 本文中，来自图像 $I$ 的 regions $r_{i}$ 作为 jobs，来自描述 $C$ 的 words $w_{j}$ 作为 workers。

* 给定一张图像 $I$ ，来自 image encoder 的候选区域特征 $R = [r_{1}, r_{2}, ... , r_{m}]$，$m$ 为  候选 regions 个数；

* 给定一个描述 $C$ ，从中选取所有**名词**，并用 language encoder 编码成 word embedding $W = [w_{1}, w_{2}, ... , w_{|W|}]$，$|W|$ 为描述 $C$ 中名词的数量； 通常 $m > |W|$
* regions 和 words 之间的对齐分数为 $S = WR^{T} \quad \quad \quad |W|*m$
* 匹配操作后，通过以下交叉熵损失对分类头进行优化：

$$
L_{region−word} = \sum_{i=1}^{|W|} - [log\sigma(s_{ik}) + \sum_{j \in W′} log(1 - \sigma(s_{ik}))] \tag{2}
$$

​	其中 $\sigma$ 为 sigmoid 激活函数，$s_{ik}$ 为第 $i$ 个 word embedding 和 第 $k$ 个 region feature 之间的对齐分数；$W′$ 表示同一 batch 中 其它 Caption 中的名词。

进一步，本文还将图像-文本对视为特殊的区域-词对。通过将整个图像作为一个特殊区域，将文本编码器中的整个 caption 特征作为一个特殊单词，从而提取图像的 RoI 特征。对于一个图像，将其 caption 视为阳性样本，而同一 batch 中的其他 caption 作为阴性样本。用类似于公式 (2) $L_{image−text}$ 损失作为约束。





















