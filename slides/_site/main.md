name: inverse
class: center, middle, inverse
layout: true
title: YouTube-8M Review

---
class: titlepage, no-number

# Review: Google Cloud & YouTube-8M Video Understanding Challenge
## .gray.author[Seil Na, Youngjae Yu, Sangho Lee, Jisung Kim]
### .gray.small[Vision & Learning Lab, SNU & Video Tech Lab, SK Telecom]
### .gray.small[July 5, 2017]
### .x-small[https://seilna.github.io/youtube8m-review]
.sklogobg[ ![Sklogo](images/sk-logo.png) ]
.snulogobg[ ![snulogo](images/snu-logo.png) ]

---
layout: false

## About

- **YouTube-8M**: Video Multilabel Classification Challenge 소개
- Video Understanding Tutorial: (i)Bag of Visual Words, (ii)Fisher Vector, (iii)VLAD, (iv)RNN
- Solution for our team: SNUVL X SKT
- Solutions for 1st team
- Further observation, Conclusion
- 그래서 Video Understanding을 잘 하려면...?
---

template: inverse

# Introduction to YouTube-8M

---
# YouTube-8M

### **.blue[Video]** Multi-label Classification

- input으로 최대 300초 길이의 Video가 들어온다
- number of samples: (Train / Validation / Test) : (4M / 1.4M / 1.4M)
- 전체 비디오 길이는 대략 450,000 시간
- raw video 데이터가 주어지는 것이 아닌, Inception Network로 뽑은 feature 형태로만 데이터 사용가능
- 1초 간격으로 raw video에서 2,048-D inception feature를 뽑고, PCA를 거쳐 1,024-D 로 차원을 줄임
- 추가적으로, 1초 간격으로 128-D audio feature 를 뽑아서(via VGG Network), video feature에 concat함
- 즉, 300초 길이의 비디오라면, 300 x 1,152 matrix가 input이 됨
- 시각적 정보만으로 classification 할 수 있는 비디오만 수집. 즉, audio feature는 도움은 되지만 반드시 필요한 것은 아님.

---
# YouTube-8M

### Video **.blue[Multi-label Classification]**

- 주어진 video에 대하여, 미리 정의된 4,716개 class로 classification 을 해야 함
- 단, 2개 이상의 class가 정답이 될 수 있는, multi-label classification 문제임
- 평균적으로, video 하나 당 3.4 개의 ground-truth label이 붙어있음
- **label imbalance**: 각 class에 속한 example 수가 많이 차이가 나서, 이미 biased된 데이터셋에 classifier를 fit시키므로 generalize 하기 어렵다
- **correlations between labels**: (Mario Cart, Mario Cart3, Mario Cart 64) 등 label 자체의 속성만으로 이미 연관되어 있는 label들이 있고, 이들은 특정 비디오에 같이 나타날 확률이 높음
---
# YouTube-8M
### correlations between labels
.img-100[![](images/mario.png)]
.footnote[https://research.google.com/youtube8m/explore.html]
---
# YouTube-8M

### Evaluation
* Kaggle submission page에서 Test dataset을 대상으로 진행
* model이 예측한 4,716개 class score 중, Top20 개의 score만 추출
* Top20 개의 score와 ground-truth 간의 [GAP(Google Average Precision)](https://www.kaggle.com/c/youtube8m#evaluation) 을 측정
* competition 도중에는 1.4M 개의 test sample 중 50% 의 data만을 대상으로 public score 를 표시해주고, competition이 종료되면 나머지 50% 의 data로 final score 및 순위 공지
* test example 수가 많아 public score와 final score의 차이가 거의 없음
---
# YouTube-8M

### Final result
.img-90[![](images/result.png)]
.footnote[https://www.kaggle.com/c/youtube8m/leaderboard]
---
template: inverse

# Video Understanding Tutorial
---
## Focus
- (데이터셋 특성상)raw video(+audio)를 직접 handling할 수 없으므로, 이미 추출된 feature들을 powerful하게 **pooling** 해보자
- 즉, T x 1,152 input matrix(T=비디오 길이) 를 compact 한 vector로 표현하는 과정
- 이 과정에서, video domain의 특성을 잘 살리면서 dimension reduction을 해야 하는데, **i)distribution modeling**과 **ii)temporal modeling** 을 고려해야 함
- **distribution modeling**: T개의 frame vector들이 전체적으로 어떤 **분포**를 띠는지에 포커싱
- **temporal modeling**: T개의 frame vector들이 배열되어 있는 **순서**를 capturing 하는 데에 포커싱
---
## Mean Pool
- **distribution modeling에 초점**
- 가장 간단한 pooling 방법
- 각 frame vector의 distribution을 blur하는 효과가 있어, 전체 프레임 중 일부분에 특히 집중해야 하는 경우를 커버하기 어려움(비디오의 일부분이 key factor가 될 때)
---
## Bag of Visual Words
- **distribution modeling에 초점**
- 본래, Image representation 을 만들기 위하여 쓰였던 것을 Video domain으로 확장(유의미한 vector로 표현되는 data는 모두 표현 가능함)
- 전체 Training data를 모두 모아서, 각 이미지에서 local descriptor(SIFT 등)을 뽑음
- 모든 data의 local descriptor들을 모아서, K-means Clustering을 함. 이 때, 각 Cluster(Visual word) 는 특정 의미를 가지는 일종의 **대표값** 역할을 함
- K개의 cluster(대표값)을 구성하고 나면, 임의의 이미지에서 똑같이 local descriptor를 추출, 각 descriptor마다 가장 가까운 cluster에 assign
- 결과적으로, 임의의 이미지는 K개 cluster에 대한 histogram 으로 표현됨
---
## Bag of Visual Words
.img-80[![](images/bow1.png)]
.footnote[Li Fei-Fei, Rob Fergues, Antonio Torralba, "Recognizing and Learning Object Categories", ICCV 2005 short course]
---
## Bag of Visual Words
.img-50[![](images/bow2.png)]
.footnote[Li Fei-Fei, Rob Fergues, Antonio Torralba, "Recognizing and Learning Object Categories", ICCV 2005 short course]

---
## Bag of Visual Words
Video로 확장하면?
* local descriptor -> frame feature vector
* frame feature vector 전체를 모아 clustering
* 임의의 video는 cluster에 대한 histogram으로 표현됨
---
## Bag of Visual Words
**.green[장점]**
- mean pool 보다 더욱 powerful한 pooling으로 볼 수 있음(다양한 cluster들의 조합으로 표현되므로)

**.red[단점]**
- Fully differentiable 하지 않음 -> End-to-End training 불가
- 데이터가 많아지면 clustering 하는 데에 오래 걸림
- ** frame vector들의 order를 capture하지 못함 ** -> video representation 측면에서 치명적(?)
---
## VLAD: Vector of Locally Aggregated Descriptors
- **distribution modeling에 초점**
- BoW와 비슷하게, frame vector들로 clustering을 하여 임의의 video를 cluster 정보로 표현
- 그러나, cluster에 대한 histogram 형태가 아닌, 각 frame vector와 cluster center와의 **difference**로 표현
- cluster 개수를 $K$, local descriptor dimension을 $d$, 각 cluster center를 $c_k$, 표현하려는 이미지의 VLAD representation을 $v \in \mathbb{R}^{K \times d}$ 라고 했을 때,

- $v_k = \sum_i^N \mathbb{1}(x_i \in c_k) \cdot (x_i - c_k)$ 
- 즉, VLAD representation $v$를 보면, K개 cluster에서 각각 $d$-dimensional vector형태의 정보가 나오게 됨
---
## VLAD: Vector of Locally Aggregated Descriptors
.img-50[![](images/vlad1.png)]
.footnote[https://ryanlei.wordpress.com/2011/03/09/]
---
## VLAD: Vector of Locally Aggregated Descriptors
**.green[장점]**
- 각 cluster에 대한 count 정보만 저장하고 있는 BoW보다 메모리를 더 많이 먹지만, 각 cluster로부터의 difference(residual)을 vector 형태로 저장하므로 좀 더 powerful함
- (실험적으로) representation이 sparse하고, 각 cluster에 속한 descriptor들의 residual이 한쪽으로 쏠리는 경향(매우 크거나 or 매우 작거나)이 있어 PCA와 궁합이 잘 맞음

**.red[단점]**
- clustering 하는 데에 오래 걸림
- distribution modeling 에 초점을 맞추고 있어 **frame vector의 order를 capture하지 못함**
---
## VLAD: Vector of Locally Aggregated Descriptors
.img-90[![](images/vlad2.png)]
* sparse representation(few values have a significant energy)
* most high descriptor values are located in same cluster
.footnote[Jégou, Hervé, et al. "Aggregating local descriptors into a compact image representation.", CVPR 2010]
---
## Fisher Vector
- **distribution modeling에 초점**
- BoW, VLAD와 비슷하게 모든 training data를 설명하는 Gaussian Mixture Model $p(x|\theta), \theta = (\mu_k, \Sigma_k, \phi_k), k=1...K$ 를 정의함
- model $p(x|\theta)$ 의 parameter에 대한 gradient $\nabla_\theta p(x|\theta)$를 representation으로 사용 
- GMM의 $K$개 mode 마다 gradient vector $\nabla_{\mu_k} p(x|\theta)$ 가 representation vector로 사용이 됨
- $\nabla_{\sigma_k} p(x|\theta)$ 가 같이 사용되는 경우도 있음

.img-100[![](images/fisher1.png)]
.footnote[https://www.slideshare.net/anhtuan68/a-survey-about-object-retrieval]
---
## RNN: Recurrent Neural Network
* temporal modeling 에 초점

**.green[장점]**
* 매 time step의 input에 frame vector가 순서대로 입력되고, 이전 time step의 state도 입력에 포함되므로, frame sequence의 order를 반영할 수 있음

**.red[단점]**
* 상대적으로 각 frame vector들의 전체적인 distribution을 capture하기에는 어려움
---
template: inverse

# Solution for our team: SNUVL X SKT
(8th ranked model, accepted at CVPR'17 Workshop on YouTube-8M Large-Scale Video Understanding as Oral Presentation)

paper : [link](https://arxiv.org/abs/1706.07960)

code: [link](https://github.com/seilna/youtube8m)
---
# Solution for our team: SNUVL X SKT
.img-33[![](images/overall.png)]
---
# Solution for our team: SNUVL X SKT
model을 다음과 같이 4개의 component 로 나눔 
* i)Frame Encoder 
* ii)Classification Layer
* iii)Label Processing Layer
* iv)Loss Function

각 component 에서 YouTube-8M 을 풀기 위하여 다음의 3가지 이슈를 정하고, 이를 커버하기 위한 method들을 제안
* i)frame feature pooling
* ii)label imbalance problem
* iii)correlation between labels
---
## frame feature pooling
.img-80[![](images/snu1.png)]
---
## frame feature pooling
### variants of LSTM

* distribution + temporal modeling
* 매 time step마다 frame vector가 input으로 들어가는 전형적인 LSTM
* pooling feature로써 (LSTM의 마지막 hidden state + 매 step output의 average + 매 step input의 average) 를 사용
* LSTM의 각 layer에 Layer Normalization 적용
* output $\in \mathbb{R}^{4 \times d}$ (d=size of LSTM cell)

.img-30[![](images/snu2.png)]
---
## frame feature pooling
### CNN
* distribution + temporal modeling
* Text CNN style의 convoltuion operation을 사용
* $T \times 1152$ matrix에 $c_h \times c_v \times 1 \times d$ kernel로 conv ($c_h=5, c_v=1152, d=256 $) 후에
* time 축으로 max-pooling 적용
* output $\in \mathbb{R}^{d}$ ($d$=size of output channel)

.img-33[![](images/snu3.png)]
---
## frame feature pooling
### Position Encoding
* End-to-End Memory Networks 에서 쓰였던 Position Encoding Matrix를 elemtwise-multiplication

* $\mathbf{L}_{ij} = (1 - i / T)-(j / 1152)(1 - 2 \times i/ T), \mathbf{L} \in \mathbb{R}^{T \times 1152}$

* frame vector에 $\mathbf{L}$ 을 곱해준 후, time 축으로 average
* output $\in \mathbb{R}^{1152}$

.img-33[![](images/snu4.png)]
---
## frame feature pooling
### Indirect Clustering
* Fisher Vector, VLAD 와 비슷한 distribution modeling 방법
* 각 frame vector를 원소로 하는 cluster를 구성하고, 그 중 크기가 가장 큰 cluster에 더 많이 집중(해당 cluster에 포함된 frame들이 main scene이 될 것이다)
* 직접 cluster를 만들기엔 오래 걸리므로, frame vector들 간의 attention으로 간접 모델링
* $\mbox{p}_t = \mbox{softmax}(\sum_i^T \mathbf{I}_t \cdot \mathbf{I}_i)  $
* frame vector와 attention의 weighted summation

.img-25[![](images/snu5.png)]
---
## frame feature pooling
### Adaptive Gaussian Noise

* label imbalance problem 을 cover 하기 위한 방법
* example 수가 더 적은 class에 속한 frame vector에 더 많은 noise를 주자

* $ \mbox{I}_t \gets \mbox{I}_t + \gamma \cdot \mbox{Z}$, where $\mbox{Z} \sim  \mathcal{N}(0, I)$, $\gamma = \frac{1}{n} \sum_i^n \frac{1}{S(y_i)}$ 

.img-30[![](images/snu6.png)]
---
## classification layer

.img-100[![](images/classification1.png)]
---
## classification layer
### Multi-Layer MoE

* Regression model 을 동시에 여러 개 training 하고, 서로 다른 regression model 마다 trainable한 weight를 주는 방법
* 각 regression model의 activation(gate)은 $g_i = \mbox{softmax}(w_g^T O_g + b_g)$ 로 정의되고,
* 각 regression model 에 해당하는 weight는 $e_i = \sigma(w_e^T O_g + b_e)$ 로 정의
* gate activation 을 구하는 layer를 multi-layer로 구성

.center.img-15[![](images/classification2.png)]
---
## classification layer
### N-Layer MLP
* Fully connected layer 3개 + Layer Normalization

.center.img-20[![](images/classification3.png)]
---
## classification layer
### Many-to-Many
* 매 time step마다 frame vector가 input으로 들어가는 LSTM에서, time step 마다 output으로 class score를 구함
* 최종 score는 매 time step 에서 구해진 score의 average

.center.img-20[![](images/classification4.png)]
---
## label processing layer
.img-90[![](images/label1.png)]
---
## label processing layer
* training과 독립적으로, label correlation matrix $\mbox{M}_c$ 계산
* training data 에서, 한 example 내에서 같이 등장하는 label들을 서로 count 후 L2-Normalize
* $\mbox{M}_c$ 가 가지고 있는 correlation 정보를 score 를 update 하는 데에 사용
* $\mbox{O}_c = \alpha \cdot \mbox{O}_h + \beta \cdot \mbox{O}_h \mbox{M}_c + \gamma \cdot \mbox{O}_h \mbox{M}_c^\prime$ 
* $\alpha, \beta, \gamma$ 는 hyperparameter, $\mbox{M}_c$ 는 fix 된 matrix, $\mbox{M}_c^\prime$ 은 trainable 한 matrix(초기화는 $\mbox{M}_c$ 값으로)

.center.img-33[![](images/label2.png)]
---
## loss function
### center loss

* video 의 embedding을 구성할 때, 다른 class에 속하는 example의 embedding이 서로 떨어지도록 loss에 constraints를 줌
* 또한, 같은 class에 속하는 example의 embedding은 더 가깝게 함
* $\mathcal{L}_c = \frac{1}{N} \sum_i^N \parallel e_i - c_k \parallel^2, k=y_i$
* $\mathcal{L} = \lambda \cdot \mathcal{L}c + \mathcal{L}_s$

.center.img-40[![](images/centerloss1.png)]
---
## loss function
### Huber loss
* L2 loss와 L1 loss의 combination
* noise instance나, sample이 아주 적은 class에 속한 instance에 대한 효율적인 training
* differntiable form을 위해 Pseudo-Huber loss function을 다음과 같이 사용
* $\mathcal{L} = \delta^2(\sqrt{1 + (\mathcal{L}_{CE}/\delta)^2}-1) $ 
---
## Training
* Adam Optimizer with learning rate=0.0006 $\beta_1=0.9, \beta_2=0.999, \epsilon=1e-8$ 
* batch size = 128
* learning rate decay with 0.95 for every 1.5M step
* training data와 validation data를 모두 training에 활용하고, 5 epoch동안 training
---
## Result - frame encoding(pooling method)
.center.img-50[![](images/result1.png)]
* LSTM 이 pooling method 중에는 가장 좋은 성능 $\rightarrow$ temporal modeling이 distribution modeling보다 중요한가?
* LSTM 안의 더 많은 정보 $(\mbox{M, O})$ 를 활용할 수록 성능이 더 좋아짐 $\rightarrow$ distribution 정보를 추가로 포함시키면 성능이 올라간다
* Layer Normalization은 20~30배 큰 learning rate에도 안정적이고 빠르게 학습을 하게 해주지만, final performance는 더 낮아졌음 $\rightarrow$ Large-Scale Dataset 에서도 LSTM-LN이 과연 효과적인가?
---
## Result - frame encoding(pooling method)
.center.img-50[![](images/result1.png)]
* CNN 은 기대와 달리 매우 낮은 성능을 보여줌
* 그러나 채널 개수를 늘릴수록(64$\rightarrow$256) 가파른 성능 향상을 보임
* 1024 채널부터는 GPU 메모리 한계(12GB) 때문에 모델 로드 불가능
* 더 많은 채널에서 CNN 이 LSTM 보다 좋은 성능을 보여줄 수 있을까? $\rightarrow$ convolution operation이 fully parallelizable 하므로, 분산 환경에서의 활용 가능성
---
## Result - frame encoding(pooling method)
.center.img-50[![](images/result1.png)]
* Position Encoding 은 성능 향상이 거의 없었음
* PE는 frame 의 order가 반영되는 정도가 약하다(trainable parameter가 없기 때문에)
---
## Result - frame encoding(pooling method)
.center.img-50[![](images/result1.png)]
* Indirect Clustering도 매우 조금 성능 향상 $\rightarrow$ distribution modeling이 도움은 되지만, 좀 더 섬세한 방법이 필요
* Adaptive Noise는 거의 성능 향상 없음 $\rightarrow$ i)label imbalance modeling이 너무 naive 하거나, ii)performance metric 만 중요하다면, label imbalance problem 을 다루는 것이 오히려 손해?
---
## Result - classification layer
.center.img-60[![](images/result2.png)]

* Many-to-Many 모델은 LSTM 구조를 사용함에도 불구하고 높은 성능을 내는 데에 실패 $\rightarrow$ why?
* Multi-layer MoE 모델은 overfitting이 심하게 발생
* LSTM과 달리, fully-connected layer 에서는 Layer Normalization이 잘 동작하였다

---
## Result - label processing layer
.center.img-60[![](images/result3.png)]

* correlation 정보를 활용한 모델 모두 성능 변화가 크게 없었음 $\rightarrow$ 이 부분이 없다면, 모델은 4716개 각 class에 대하여 서로 완전히 독립적인 classifier를 훈련시키고 있는 것이기 때문에, correlation을 더 잘 반영하는 method가 있다면 높은 확률로 성능 향상을 가져올 수 있을 것으로 예상
---
## Result - loss function
.center.img-60[![](images/result4.png)]

* center loss는 feature 로 input이 주어지는 상황에서 사용이 제한적이고, 성능 향상이 미미했음
* Huber loss는 꽤 많은 성능 향상을 보여 YouTube annotation System$^1$의 noise에 어느정도 robust하게 만들어 주는 역할을 함
.footnote[1. https://www.youtube.com/watch?v=wf_77z1H-vQ]
---
## Result - Ensemble
* 이러한 모델들을 모아 총 40개 가량 모델을 ensemble 하여 test performance 0.839를 달성
* 기존 ensemble에 모델을 추가할 때, 추가할 모델 중 single 로 best-performance 를 내는 모델이 꼭 ensemble 했을 때 가장 좋은 성능을 내지는 않았음
* ensemble의 의도가 서로 다른 experts 들의 score를 어느정도 blur해줌으로써 성능 향상이 있는 것이기 때문에, 비슷한 모델을 ensemble하는 경우 performance 향상이 높지 않음
* 따라서, ensemble의 성능을 높이려면 합치려는 모델을 고를 때 최대한 다른 mechanism으로 동작하는 모델을 선택하면 좋음 
---
template: inverse

# Solution for 1st team: WILLOW
Learnable pooling with Context Gating for video classification, Antoine Miech et al. [link](https://arxiv.org/abs/1706.06905)
---
# Solution for 1st team: WILLOW
.center.img-100[![](images/willow1.png)]
* 전체 pipeline을 2단계로 나눔 $\rightarrow$ i)features pooling, ii)classification layer
* Context Gating 이라는 새로운 method를 도입
---
# features pooling
* T 개의 frame vector들을 pooling 하여 compact 한 vector(또는 matrix) 로 표현
* pooling을 위하여 다음 4가지 모델을 사용
* i)Variant of NetVLAD
* ii)Soft BoW
* iii)NetFV(Net Fisher Vector)
* iv)LSTM/GRU
---
# features pooling
## NetVLAD
.img-30[![](images/netvlad1.png)]

* 기존 VLAD representation은 task에 관계없이 오직 feature들로부터 구성되는 cluster를 통해 embedding을 만듬
* NetVLAD는 task에 dependant하게(좀 더 fit 되도록) cluster를 update
* End-to-End 방식으로 cluster 까지 update할 수 있음
.footnote[Arandjelovic, Relja, et al. NetVLAD: CNN architecture for weakly supervised place recognition. CVPR 2016]
---
# features pooling
## NetVLAD
.img-30[![](images/netvlad1.png)]

* VLAD representation: local descriptor(여기에서는 frame vector) 들을 k개 cluster에 대한 residual로 표현
* $V(k) = \sum_i^N a_k(x_i)(x_i - c_k)$ 
* cluster k에 대한 representation은 해당 cluster에 속한 frame vector들과 cluster center와의 sum of redisual로써 표현됨
* 여기에서, $a_k(x_i)$ 는 $x_i$ 가 cluster $k$ 에 속할 때 1, 아니면 0의 값을 가지는 데, 이 같은 hard assign operation 때문에 미분이 불가능함

---
# features pooling
## NetVLAD
.img-30[![](images/netvlad1.png)]

* hard assign을 미분가능하도록 바꾸고, 이전에 clustering해서 얻은 $c_k$ 들을 task에 fit되도록 update하자는 것이 핵심
* 따라서, 먼저 assign을 soft한 형태로 제안하고, $\bar{a}_k(x_i) = \frac{e^{-\alpha \parallel x_i - c_k \parallel^2}}{\sum_j^K e^{-\alpha \parallel x_i - c_j \parallel^2}}$ 
* trainable 한 parameters $\mathbf{w}_k = 2 \alpha c_k, \mathbf{b}_k = -\alpha \parallel c_k\parallel^2$ 를 도입하여 soft assign 을 수정: $\bar{a}_k(x_i) = \frac{e^{\mathbf{w}_k^T x_i + \mathbf{b}_k}}{\sum_j^K e^{\mathbf{w}_j^T x_i + \mathbf{b}_j}}$

---
# features pooling
## NetVLAD
.img-30[![](images/netvlad1.png)]

* 결과적으로, VLAD Representation은 다음과 같이 나타내어지며, task 에 맞도록 update됨
* $V(k) = \sum_i^N \bar{a}_k(x_i)(x_i - c_k)$
---
# features pooling
## NetVLAD $\rightarrow$ NetRVLAD

* WILLOW team은 기존의 NetVLAD 를 그대로 쓰지 않고 다음과 같은 minor한 변형인 NetRVLAD를 제안함
* $c_k$ 를 k-means clustering 으로 초기화하지 않고 random으로 initialization $\rightarrow$ **why?** pretrained된 cluster center로 초기화를 해도 성능 향상이 없었음
* frame vector $x_i$와 cluster center $c_k$ 사이의 residual를 summation 하는 대신, $x_i$ 정보만 누적함 $\rightarrow c_k$ 를 사용하지 않고 $w_k, b_k$ 만 활용함으로써 parameter 수를 줄임
* .red[distribution modeling] 방법으로써 frame order는 고려되지 않음
---
# features pooling
## BoW $\rightarrow$ Soft-DBoW

* NetVLAD 에서의 soft assign operation 을 BoW에 적용
* .red[distribution modeling] 방법으로써 frame order는 고려되지 않음
---
# features pooling
## NetFV

* NetVLAD 방식으로 cluster를 형성 후 residual이 아닌 gradient로 Fisher Vector를 얻음
* .red[distribution modeling] 방법으로써 frame order는 고려되지 않음
---
# features pooling
## LSTM / GRU
* hidden size=1024 의 RNN을 stack (2-layer)
* final state를 pooled feature로 활용
* .blue[temporal modeling] (frame order) 가 고려됨
---
# Context Gating
.center.img-80[![](images/willow1.png)]

* input vector 를 $x \in \mathbb{R}^{d}$ 라고 했을 때, Context Gating은 다음과 같의 정의된다
* $\mbox{Y} = \sigma(Wx + b) \circ x$ where $\sigma$ is element-wise sigmoid activation, $\circ$ is element-wise multiplication
* trainable 한 gate activation 을 구성하여 input $x$를 재조정
.footnote[Learnable pooling with Context Gating for video classification, Antoine Miech et al. CVPR 2017]
---
# Context Gating
.center.img-80[![](images/willow1.png)]

* 이러한 Context Gating 을 두 번 적용함 
* i)pooling한 frame feature들에 FC Layer 뒤에, ii)MoE로 class score 를 낸 뒤에
---
# Context Gating
## 어떤 효과?

* i)pooling 한 frame feature 뒤에 붙였을 경우
* feature 가 가지고 있는 여러 정보들의 가중치를 조정
* 예를 들어, label이 ski로 붙어있는 video의 embedding에는 눈, 나무, 스키 등 비디오에 등장한 여러 정보가 섞여있을 것
* feature 뒤에 CG를 붙이게 되면, 이러한 정보들 중 labeling에 불필요한 눈, 나무 같은 정보는 약화시키고, 스키 등의 정보는 강화시킴
* discussion. 이러한 선택적 정보 재조정 기능을 증명할 실험이 나와있지 않아 관련 실험이 필요할 듯
---
# Context Gating
## 어떤 효과?

* ii)score(output probability) 뒤에 붙였을 경우
* correlations between labels 정보를 capture 할 수 있음
* Gate 가 score의 linear transformation으로 구성되므로, 직관적인 레벨에서는 reasonable함
---
# Result - WILLOW
.center.img-40[![](images/willow2.png)]

* NetFV, NetVLAD 는 temporal order를 전혀 고려하지 않는 method 임에도 불구하고, LSTM / GRU 보다 더 높은 성능을 나타냄 
* 적어도 YouTube-8M(Video Classification) 에서는, temporal modeling 못지 않게 distribution modeling이 중요함을 시사
---
# Result - WILLOW
.center.img-40[![](images/willow2.png)]

* 사용한 모델 종류와 관계없이, Context Gating은 높은 폭의 성능 향상을 일관되게 가져옴

---
# Result - WILLOW
.center.img-50[![](images/willow3.png)]

* GLU는 CG의 original version 으로써, $Y = \sigma(W_1 x + b_1) \circ (W_2 x + b_2)$ 으로 정의됨(CG는 GLU의 simplified version)
* GLU 보다 CG 가 더 높은 성능을 나타내었으며, pooling feature 뒤에 붙인 경우와 class score 뒤에 붙인 경우 모두 성능 향상을 가져옴
---
# Result - WILLOW
.center.img-60[![](images/willow4.png)]

* audio feature와 video feature를 언제 합치는(fusion) 것이 좋은가?
* distribution modeling approach에서는 video, audio 따로 representation 을 만든 후 합치는 게 성능이 좋았음
* RNN 에서는 video + audio를 합친 형태를 input으로 넣어주는 게 성능이 좋았음
---
# Result - WILLOW
.center.img-40[![](images/willow5.png)]

* 이러한 모델들을 ensemble 할 때, greedy 한 방식을 취함
* 첫번째로, best performance 모델을 선택하고, 나머지 모델 중 best model과 앙상블했을 때 가장 성능이 높은 모델을 두번째 모델로 선택, 이후 반복
* 모델 7개 앙상블만으로 kaggle기준 1위 성능을 달성하였고, 이후 모델을 25개까지 늘려 성능을 더 높임
---
template: inverse
# Conclusion
---
# Conclusion
* label imbalance problem을 cover해서 성능 향상을 본 팀은 없었음
* correlations between label은 거의 모든 팀에서 다루었으나, 전부 성공적이지는 않았음
* video feature와 audio feature의 fusion method는 심도있게 다뤄지지 않음
* distribution 만을 modeling한 approach가 가장 좋은 성능을 내었음
* temporal encoding method 도 많은 성능 향상을 가져옴
* 즉, ** frame vector의 distribution 과 temporal order를 모두(한번에) 모델링 할 수 있는 simple한 모델이 필요** 해 보임
---
# Video Understanding 을 잘 하려면?
* YouTube-8M Video Classification 은 Video Understanding 의 부분집합이기 때문에...
* YouTube-8M 에서 다뤄진 높은 성능의 pooling method들이 다른 video understanding task(QA, Captioning, retrieval 등)에도 좋은 성능을 낼 것이라는 보장은 없음
* 결론은 task 에 맞게 가장 좋은 모델을...
---
name: last-page
class: center, middle, no-number
## Thank You!


<div style="position:absolute; left:0; bottom:20px; padding: 25px;">
  <p class="left" style="margin:0; font-size: 13pt;">
  <b>Special Thanks to</b>: Jongwook Choi</p>
</div>

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]




<!-- vim: set ft=markdown: -->
