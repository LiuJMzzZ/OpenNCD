# Open-world Semi-supervised Novel Class Discovery
[IJCAI 2023] Official code for [Open-world Semi-supervised Novel Class Discovery](https://arxiv.org/abs/2305.13095) (OpenNCD)

## Introduction

Open-world scenario: 

- Unseen novel classes mixed in unlabeled data in semi-supervised learning

Our tasks:

1. Recognize the known classes
2. Discover the novel classes
3. Estimate the number of the novel classes



## Running

### Requirements

Please refer to requirements.txt.

### Pretrain Models

We use the unsupervised SimCLR for pretraining. The pretrained resnet-18 models can be found [here](https://drive.google.com/drive/folders/1brOsw-09BKJLu0W6aTDsjYtYNDDFDnct?usp=share_link) (same as [orca](https://github.com/snap-stanford/orca)). Please unzip them to './pretrained'.



### Scripts

- If the number of novel classes is **pre-known**, spectral clustering will be used for prototype grouping. 
- To train on CIFAR-10 with 10\% labeled data in known class data, run
```bash
python main.py --dataset cifar10 --labeled_num 5 --labeled_ratio 0.1  --save_log
```


- If the number of the novel class is **unknown**, ['propagation', 'connected', 'louvain'] can be used for prototype grouping and class number estimation, where ['louvain'] performs best in our experiments. 
- To train on CIFAR-10 without a pre-defined number of classes, run
```bash
python main.py --dataset cifar10 --labeled_num 5 --labeled_ratio 0.1 --group_method louvain  --unknown_n_cls --save_log
```

## Update
**2023.8.25** Fix the bug when training with the known number of class

**2023.8.10** Update main.py in the class number estimation part (--unknown_n_cls)
- Implement an adaptive range for EPS searching to facilitate progressive grouping
- Set the value of l_cls to 0 at the grouping stage to avoid over-fitting


## Acknowledgements
Thanks for the excellent repositories [orca](https://github.com/snap-stanford/orca) and [GCD](https://github.com/sgvaze/generalized-category-discovery).

## Citation

If you find our code useful, please consider citing:

```

@inproceedings{ijcai2023p445,
  title     = {Open-world Semi-supervised Novel Class Discovery},
  author    = {Liu, Jiaming and Wang, Yangqiming and Zhang, Tongze and Fan, Yulu and Yang, Qinli and Shao, Junming},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {4002--4010},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/445},
}
