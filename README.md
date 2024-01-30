## Description
This repository contains the latest source codes of PANE proposed in the conference paper titled "Scaling Attributed Network Embedding to Massive Graphs" and its extended version to graphs with large attribute sets.

## Env Requirements
- Linux
- Python 2.7

## Data preprocessing
Collect datasets from [here](https://renchi.ac.cn/datasets/).
```
$ cd data/
$ python split_train_test.py cora/edgelist.txt 0.3   # split edges into training set (70%) and positive test set (30%)
$ python gen_neg_egdes.py cora/edgelist.txt 0.3      # generate negative test set (30%)

$ python gen_test_neg_attr.py cora/attrs.pkl 0.3     # split attributes into training set (70%) and test set (30%)
```

## Generate embeddings using PANE
```
$ mkdir emb
$ mkdir emb/mask/
$ python2.7 paneplus.py --d 128 --full 1 --t 5 --data flickr    # generate embeddings for node classification
$ python2.7 paneplus.py --d 128 --full 0 --t 5 --data flickr    # generate embeddings for link prediction
$ python2.7 paneplus.py --d 128 --full 1 --t 5 --data flickr --mask 0.7   # generate embeddings for attribute inference
```

## Generate embeddings using PANE++
```
$ mkdir emb
$ mkdir emb/mask/
$ python2.7 paneplus.py --d 128 --full 1 --t 5 --data flickr --kappa 1024    # generate embeddings for node classification
$ python2.7 paneplus.py --d 128 --full 0 --t 5 --data flickr --kappa 1024    # generate embeddings for link prediction
$ python2.7 paneplus.py --d 128 --full 1 --t 5 --data flickr --kappa 1024 --mask 0.7   # generate embeddings for attribute inference
```

## Evaluation
```
$ python node_class.py --algo pane --data cora --d 128    # node classification
$ python link_pred_plus.py --algo pane --data cora --d 128     # link prediction
$ python attr_infer_plus.py --algo pane --data cora --d 128 --ratio 0.7   # attribute inference
```

## Citation
```
@article{yang2020scaling,
  title={Scaling Attributed Network Embedding to Massive Graphs},
  author={Yang, Renchi and Shi, Jieming and Xiao, Xiaokui and Yang, Yin and Liu, Juncheng and Bhowmick, Sourav S},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={1},
  pages={37--49},
  year={2021},
  publisher={VLDB Endowment}
}

@article{yang2023pane,
  title={PANE: scalable and effective attributed network embedding},
  author={Yang, Renchi and Shi, Jieming and Xiao, Xiaokui and Yang, Yin and Bhowmick, Sourav S and Liu, Juncheng},
  journal={The VLDB Journal},
  pages={1--26},
  year={2023},
  publisher={Springer}
}
```


