### 面向多天气退化图像恢复的自注意力扩散模型

摘要：复杂天气下的图像恢复对于后续的高级计算机视觉任务具有重要意义。然而，多数现有的图像恢复算法仅能去除单一天气退化，鲜有针对多天气退化图像恢复的同一模型。对此，本文结合去噪扩散概率模型 (Denoising Diffusion Probability Model, DDPM) 和视觉自注意力网络首次提出一种用于多天气退化图像恢复的自注意力扩散模型。首先，利用天气退化图像作为条件来引导扩散模型反向采样生成去除退化的干净背景图像。其次，提出次空间转置自注意力噪声估计网络 (Subspace Transposed Transformer for Noise Estimation, NE-STT) ，其利用退化图像和噪化状态来估计噪声分布，包括次空间转置自注意力机制 (Subspace Transposed Self-Attention, STSA) 和双分组门控前馈网络 (Dual Grouped Gated Feed-Forward Network, DGGFFN)。STSA利用次空间变换系数实现有效学习特征全局性长距离依赖的同时显著降低计算负担，DGGFFN利用双分组门控机制来增强前馈网络的非线性表征能力。实验结果表明，在5个天气退化图像数据集上，相比近来的同类算法All-in-One和TransWeather，本文算法所得恢复图像的平均峰值信噪比分别提高3.68 dB和3.08 dB，平均结构相似性分别提高2.93%和3.13%，并且噪声估计网络的单步估计用时相比目前的DDPM减少38.50%。

github: [https://github.com/chdwyb/AIR-TDM](https://github.com/chdwyb/AIR-TDM)

![扩散过程示意](images/扩散过程示意.png)