# Deep learning in survival analysis

This repository contains all of the code used to write my masters thesis on usage of deep learning methods for survival analysis. It contains 2 models, a continous-time model based on work by Katzman et al.[1] and a discrete-time model based on Lee et al.[2]

## Evaluation
The models were evaluated on 6 survival analysis datasets, and for the [MEATABRIC](https://www.cbioportal.org/study/summary?id=brca_metabric) dataset they got evaluated on both clinical data and clinical data with mRna z scores 

[1] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger. Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.  _BMC Medical Research Methodology_, 18(1), 2018. [[paper](https://doi.org/10.1186/s12874-018-0482-1)]

[2] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning approach to survival analysis with competing risks.  _In Thirty-Second AAAI Conference on Artificial Intelligence_, 2018. [[paper](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit)]
