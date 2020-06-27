## Combining structured and unstructured data for predictive models: a deep learning approach

This repository contains source code for paper *Combining structured and unstructured data for predictive models: a deep learning approach*. In this paper, we proposed 2 frameworks, namely Fusion-CNN and Fusion-LSTM, to combine sequential clinical notes and temporal signals for patient outcome prediction. Experiments of in-hospital mortality prediction, long length of stay prediction, and 30-day readmission prediction on MIMIC-III datasets empirically shows the effectiveness of proposed models. Combining structured and unstructured data leads to a significant performance improvement.

### Framework

![Fusion-CNN](https://imgur.com/nKhAOrM.png)

> Fusion-CNN is based on document embeddings, convolutional layers, max-pooling layers. The final patient representation is the concatenation of the latent representation of sequential clinical notes, temporal signals, and the static information vector. Then the final patient representation is passed to output layers to make predictions.

![Fusion-LSTM](https://imgur.com/AgrIkl6.png)

> Fusion-LSTM is based on document embeddings, LSTM layers, max-pooling layers. The final patient representation is the concatenation of the latent representation of sequential clinical notes, temporal signals, and the static information vector. Then the final patient representation is passed to output layers to make predictions.

### Requirements

#### Dataset

MIMIC-III database analyzed in the study is available on [PhysioNet](https://mimic.physionet.org/about/mimic) repository. Here are some steps to prepare for the dataset:

- To request access to MIMIC-III, please follow https://mimic.physionet.org/gettingstarted/access/. Make sure to place `.csv` files under `data/mimic`.
- With access to MIMIC-III, to build the MIMIC-III dataset locally using Postgres, follow the instructions at https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres.
- Run SQL queries to generate necessary views, please follow https://github.com/onlyzdd/clinical-fusion/tree/master/query.

#### Software

- Python 3.6.10
- Gensim 3.8.0
- NLTK: 3.4.5
- Numpy: 1.14.2
- Pandas: 0.25.3
- Scikit-learn: 0.20.1
- Tqdm: 4.42.1
- PyTorch: 1.4.0

### Preprocessing

```sh
$ python 00_define_cohort.py # define patient cohort and collect labels
$ python 01_get_signals.py # extract temporal signals (vital signs and laboratory tests)
$ python 02_extract_notes.py --firstday # extract first day clinical notes
$ python 03_merge_ids.py # merge admission IDs
$ python 04_statistics.py # run statistics
$ python 05_preprocess.py # run preprocessing
$ python 06_doc2vec.py --phase train # train doc2vec model
$ python 06_doc2vec.py --phase infer # infer doc2vec vectors
```

### Run

#### Baselines

Baselines (i.e., logistic regression, and random forest) are implemented using scikit-learn. To run:

```sh
$ python baselines.py --model [model] --task [task] --inputs [inputs]
```

#### Deep models

Fusion-CNN and Fusion-LSTM are implemented using PyTorch. To run:

```sh
$ python main.py --model [model] --task [task] --inputs [input] # train Fusion-CNN or Fusion-LSTM
$ python main.py --model [model] --task [task] --inputs [input] --phase test --resume # evaluate
```
