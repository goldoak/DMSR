# DMSR

This repository contains code and resources related to the paper [RGB-based Category-level Object Pose Estimation via Decoupled Metric Scale Recovery](https://arxiv.org/abs/2309.10255) by Jiain Wei.

## Abstract

While showing promising results, recent RGB-D camera-based category-level object pose estimation methods have restricted applications due to the heavy reliance on depth sensors. RGB-only methods provide an alternative to this problem yet suffer from inherent scale ambiguity stemming from monocular observations. In this paper, we propose a novel pipeline that decouples the 6D pose and size estimation to mitigate the influence of imperfect scales on rigid transformations. Specifically, we leverage a pre-trained monocular estimator to extract local geometric information, mainly facilitating the search for inlier 2D-3D correspondence. Meanwhile, a separate branch is designed to directly recover the metric scale of the object based on category-level statistics. Finally, we advocate using the RANSAC-P$n$P algorithm to robustly solve for 6D object pose. Extensive experiments have been conducted on both synthetic and real datasets, demonstrating the superior performance of our method over previous state-of-the-art RGB-based approaches, especially in terms of rotation accuracy.


## Citation

If you find our work useful in your research, please cite our paper:

```
@misc{wei2023rgbbased,
      title={RGB-based Category-level Object Pose Estimation via Decoupled Metric Scale Recovery}, 
      author={Jiaxin Wei and Xibin Song and Weizhe Liu and Laurent Kneip and Hongdong Li and Pan Ji},
      year={2023},
      eprint={2309.10255},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```






