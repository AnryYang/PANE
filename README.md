## Env Requirements
- Linux
- Python 2.7

## Datasets
Download from [here](https://renchi.ac.cn/#datasets)


## Data preprocessing
```
$ cd data/
$ python split_train_test.py cora/edgelist.txt 0.3   # split edges into training set (70%) and positive test set (30%)
$ python gen_neg_egdes.py cora/edgelist.txt 0.3      # generate negative test set (30%)
```

## Generate embeddings
```
$ mkdir emb
$ mkdir emb/mask/
$ python main.py --data cora --d 128 --t 5 --full 1    # generate embeddings for node classification
$ python main.py --data cora --d 128 --t 5 --full 0    # generate embeddings for link prediction
```

## Evaluation
```
$ python node_class.py --algo pane --data cora --d 128    # node classification
$ python link_pred.py --algo pane --data cora --d 128     # link prediction
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
```
