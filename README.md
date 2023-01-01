# Synthesizing Counterfactual Samples for Effective Image-Text Matching

Official PyTorch implementation of the paper [Synthesizing Counterfactual Samples for Effective Image-Text Matching](https://dl.acm.org/doi/abs/10.1145/3503161.3547814) (MM 2022 Oral).

Please use the following bib entry to cite this paper if you are using any resources from the repo.

```
@inproceedings{wei2022synthesizing,
  title={Synthesizing Counterfactual Samples for Effective Image-Text Matching},
  author={Wei, Hao and Wang, Shuhui and Han, Xinzhe and Xue, Zhe and Ma, Bin and Wei, Xiaoming and Wei, Xiaolin},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={4355--4364},
  year={2022}
}
```

We referred to the implementation of [VSE_infty](https://github.com/woodfrog/vse_infty) to build up our codebase.

## Preparation

### Environment

We trained and evaluated our models with the following key dependencies:

- Python 3.8.12

- Pytorch 1.10.2

- Transformers 4.14.1

Run `pip install -r requirements.txt` to install the exactly same dependencies as our experiments. 

### Data

We organize all data used in the experiments in the following manner:

```
data
├── coco
│   └── precomp  # pre-computed BUTD region features for COCO, provided by SCAN 
│
├── f30k
│   └── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)
```

The download links for precomputed BUTD features, and corresponding vocabularies are from the offical repo of [SCAN](https://github.com/kuanghuei/SCAN#download-data). The `precomp` folders contain pre-computed BUTD region features, and `vocab` folder contains corresponding vocabularies.

## Training

Assuming the data root is `/tmp/data`, we provide example training scripts for BUTD Region feature for the image feature, BERT-base for the text feature. See `train_coco.sh` and `train_f30k.sh`.

## Evaluation

Run `eval.py` to evaluate specified models on either COCO and Flickr30K. For evaluting pre-trained models on COCO, use the following command (assuming the local data path is `/tmp/data` and the model name is `coco_butd_region_bert`):

```
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset coco --data_path /tmp/data/coco --model coco_butd_region_bert
```

For evaluting pre-trained models on Flickr30K, use the command: 

```
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset f30k --data_path /tmp/data/f30k --model f30k_butd_region_bert
```

## Results

|           | R1   | R5   | R1   | R5   | Link                                                         |
| --------- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| COCO 1K   | 80.6 | 96.8 | 65.0 | 91.4 | [Google drive](https://drive.google.com/drive/folders/1oRRbXo6CfNY5zc7MBFEpC1ZuCW_bCVjm?usp=sharing) |
| COCO 5K   | 59.5 | 86.1 | 42.7 | 73.1 |                                                              |
| Flickr30K | 82.7 | 95.5 | 62.6 | 86.9 | [Google drive](https://drive.google.com/drive/folders/1J7A--rKfZkXGI5NRR3JCODIkSiYP2Kke?usp=sharing) |

