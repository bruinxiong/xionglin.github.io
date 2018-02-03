# Bio of Xiong Lin
![](XiongLin_NIPS2017.jpeg)

I'm a research engineer of Learning & Vision, Core Technology Group, Panasonic R&D Center Singapore (PRDCSG). I received Ph.D degree of pattern recognition & intelligent system from school of electronic engineering, Xidian University. My Master's superviser is [Prof. ZHANG li](https://www.researchgate.net/profile/Li_Zhang80/info) and Ph.D superviser is [Prof. JIAO licheng](http://web.xidian.edu.cn/lchjiao/indexen.html). I am working on developing deep neural network models and algorithm for face detection, face recognition, image generation and instance segmentation.

## Research Interests:

Unconstrained/large-scale face recognition, deep learning architecture engineering, person re-identification, transfer learning, Riemannian manifold optimization, sparse and low-rank matrix factorization.

## News:

**Jan 2018:** As a Chinese Tech-oriented media, AI Era gave a [full interview](https://mp.weixin.qq.com/s/s9H_OXX-CCakrTAQUFDm8g) for our NIST IJB-A Face Challenge. Our latest performance of IJB-A will be updated in the next version of our [arXiv](https://arxiv.org/abs/1704.00438) paper. **(coming soon)** 

**Jul 2017:** We attended [MS-Celeb-1M Large-Scale Face Recognition](http://www.msceleb.org/) with our proposed face recognition [algorithm](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w27/Xu_High_Performance_Large_ICCV_2017_paper.pdf) and archieved No.1 place on Challenge-1 [Random Set and Hard Set](http://www.msceleb.org/leaderboard/iccvworkshop-c1). AI Era gave an [interview](http://www.sohu.com/a/160503484_473283) for this challenge.

**May 2017:** Panasonic released a [news](http://news.panasonic.com/jp/press/data/2017/05/jn170510-5/jn170510-5.html) to introduce our project and report our archievent on IJB-A Face Verification and Identification Challenge. Moreover, [CNET Japan](https://japan.cnet.com/article/35100942/) also picked up the story.

**Apr 2017:** We proposed Transferred Deep Feature Fusion (TDFF) for face recogntion and obtained No.1 place on all tracks of National Institute of Standards and Technology (NIST) IARPA Janus Benchmark A (IJB-A) Unconstrained Face Verification and Identification Challenge. Official reports can be found in here: [Identification](https://github.com/bruinxiong/xionglin.github.io/blob/master/IJBA_1N_report.pdf) and [Verification](https://github.com/bruinxiong/xionglin.github.io/blob/master/IJBA_11_report.pdf) 

## Selected Publication:
* Jian Zhao, **Lin Xiong**, Karlekar Jayashree, Jianshu Li, Fang Zhao, Zhecan Wang, Sugiri Pranata, Shengmei Shen, Shuicheng Yan, Jiashi Feng, **"Dual-Agent GANs for Photorealistic and Identity Preserving Profile Face Synthesis"**, in Proceedings of the 31st Conference on Neural Information Processing Systems **(NIPS)**, Long Beach, CA, USA, 2017. **Acceptance rate is 20.9%**. [PDF](http://papers.nips.cc/paper/6612-dual-agent-gans-for-photorealistic-and-identity-preserving-profile-face-synthesis.pdf), [Full version](https://www.researchgate.net/publication/320223416_Dual-Agent_GANs_for_Photorealistic_and_Identity_Preserving_Profile_Face_Synthesis).

+ **Abstract:** Synthesizing realistic profile faces is promising for more efficiently training deep pose-invariant models for large-scale unconstrained face recognition, by populating samples with extreme poses and avoiding tedious annotations. However, learning from synthetic faces may not achieve the desired performance due to the discrepancy between distributions of the synthetic and real face images. To narrow this gap, we propose a Dual-Agent Generative Adversarial Network (DA-GAN) model, which can improve the realism of a face simulator's output using unlabeled real faces, while preserving the identity information during the realism refinement. The dual agents are specifically designed for distinguishing real v.s. fake and identities simultaneously. In particular, we employ an off-the-shelf $3$D face model as a simulator to generate profile face images with varying poses. DA-GAN leverages a fully convolutional network as the generator to generate high-resolution images and an auto-encoder as the discriminator with the dual agents. Besides the novel architecture, we make several key modifications to the standard GAN to preserve pose and texture, preserve identity and stabilize training process: (i) a pose perception loss; (ii) an identity perception loss; (iii) an adversarial loss with a boundary equilibrium regularization term. Experimental results show that DA-GAN not only presents compelling perceptual results but also significantly outperforms state-of-the-arts on the large-scale and challenging NIST IJB-A unconstrained face recognition benchmark. In addition, the proposed DA-GAN is also promising as a new approach for solving generic transfer learning problems more effectively. DA-GAN is the foundation of our submissions to NIST IJB-A 2017 face recognition competitions, where we won the 1st places on the tracks of verification and identification.

* **Lin Xiong**, Jayashree Karlekar, Jian Zhao, Jiashi Feng, Sugiri Pranata, Shengmei Shen, **"A Good Practice Towards Top Performance of Face Recognition: Transferred Deep Feature Fusion"**, [arXiv](https://arxiv.org/pdf/1704.00438.pdf) **Keep the Top 1 performance on IJB-A, the new version is coming soon.**

+ **Abstract:** Unconstrained face recognition performance evaluations have traditionally focused on Labeled Faces in the Wild (LFW) dataset for imagery and the YouTubeFaces (YTF) dataset for videos in the last couple of years. Spectacular progress in this field has resulted in a saturation on verification and identification accuracies for those benchmark datasets. In this paper, we propose a unified learning framework named transferred deep feature fusion targeting at the new IARPA Janus Bechmark A (IJB-A) face recognition dataset released by NIST face challenge. The IJB-A dataset includes real-world unconstrained faces from 500 subjects with full pose and illumination variations which are much harder than the LFW and YTF datasets. Inspired by transfer learning, we train two advanced deep convolutional neural networks (DCNN) with two different large datasets in source domain, respectively. By exploring the complementarity of two distinct DCNNs, deep feature fusion is utilized after feature extraction in target domain. Then, template specific linear SVMs is adopted to enhance the discrimination of framework. Finally, multiple matching scores corresponding different templates are merged as the final results. This simple unified framework outperforms the state-of-the-art by a wide margin on IJB-A dataset. Based on the proposed approach, we have submitted our IJB-A results to National Institute of Standards and Technology (NIST) for official evaluation.

* Shasha Mao, **Lin Xiong<sup>*</sup>**, Licheng Jiao, Tian Feng, Sai-Kit Yeung, **"A Novel Riemannian Metric Based on Riemannian Structure and Scaling Information for Fixed Low-Rank Matrix Completion"**, IEEE Transactions on Cybernetics **(TCYB)**, vol. 47, no. 5, pp. 1299–1312, 2017. **IF 7.384 (<sup>*</sup> Corresponding author)**. [PDF](https://www.researchgate.net/publication/305670183_A_Novel_Riemannian_Metric_Based_on_Riemannian_Structure_and_Scaling_Information_for_Fixed_Low-Rank_Matrix_Completion)

+ **Abstract:** Riemannian optimization has been widely used to deal with the fixed low-rank matrix completion problem, and Riemannian metric is a crucial factor of obtaining the search direction in Riemannian optimization. This paper proposes a new Riemannian metric via simultaneously considering the Riemannian geometry structure and the scaling information, which is smoothly varying and invariant along the equivalence class. The proposed metric can make a tradeoff between the Riemannian geometry structure and the scaling information effectively. Essentially, it can be viewed as a generalization of some existing metrics. Based on the proposed Riemanian metric, we also design a Riemannian nonlinear conjugate gradient algorithm, which can efficiently solve the fixed low-rank matrix completion problem. By experimenting on the fixed low-rank matrix completion, collaborative filtering, and image and video recovery, it illustrates that the proposed method is superior to the state-of-the-art methods on the convergence efficiency and the numerical performance.

* Shasha Mao<sup>+</sup>, Licheng Jiao, **Lin Xiong<sup>+</sup>**, Shuiping Gou, Bo Chen, Sai-Kit Yeung. **“Weighted classifier ensemble based on quadratic form”**. Pattern Recognition **(PR)**, vol.48 (5), pp.1688-1706, 2015.  **IF 4.582 (<sup>+</sup> Equal contribution with first author)**. [PDF](https://www.researchgate.net/publication/272402925_Weighted_classifier_ensemble_based_on_quadratic_form)

+ **Abstract:** Diversity and accuracy are the two key factors that decide the ensemble generalization error. Constructing a good ensemble method by balancing these two factors is difficult, because increasing diversity is at the cost of reducing accuracy normally. In order to improve the performance of an ensemble while avoiding the difficulty derived of balancing diversity and accuracy, we propose a novel method that weights each classifier in the ensemble by maximizing three different quadratic forms. In this paper, the optimal weight of individual classifiers is obtained by minimizing the ensemble error, rather than analyzing diversity and accuracy. Since it is difficult to minimize the general form of the ensemble error directly, we approximate the error in an objective function subject to two constraints. Particularly, we introduce an error term with a weight vector w0, and subtract this error with the quadratic form to obtain our approximated error. This subtraction makes minimizing the approximation form equivalent to maximizing the original quadratic form. Theoretical analysis finds that when the value of the quadratic form is maximized, the error of an ensemble system with the corresponding optimal weight w* will be smallest, especially compared with the ensemble with w0. Finally, we demonstrate improved classification performance from the experimental results of an artificial dataset, UCI datasets and PolSAR image data.

Full publication can be found in my [researchgate](https://www.researchgate.net/profile/Lin_Xiong4/contributions) and [googlescholar](https://scholar.google.com.sg/citations?user=Auze-lcAAAAJ&hl=en).

## Work Experience:

* 09/2015 - Present: Research Engineer of Panasonic R&D Center Singapore, Singapore.

* 05/2015 - 09/2015: 2012 Labs, MV OSS Technology Development Department in HuaWei Technologies Co., LTD.

## Awards:

* 

* 

## Hobbies:

Photography, Travel, Figure collection (Saint Seiya, Mazinger and so on), LEGO and Cooking.

## Contact:

**Email:** lin.xiong@sg.panasonic.com, bruinxiong@me.com, bruinxiongmac@gmail.com

**Phone:** +65 83752875

**WeChat:** bruinxiongmac

**Address:** 202 Bedok South Avenue 1 #02-11, Panasonic R&D Center Singapore, Singapore, 469332 
