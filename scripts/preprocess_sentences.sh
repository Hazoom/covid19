#!/usr/bin/env bash
python src/preprocessing/preprocess.py -m ../../workspace/kaggle/covid19/data/metadata.csv -d ../../workspace/kaggle/covid19/data/biorxiv_medrxiv/biorxiv_medrxiv/ ../../workspace/kaggle/covid19/data/noncomm_use_subset/noncomm_use_subset/ ../../workspace/kaggle/covid19/data/comm_use_subset/comm_use_subset/ ../../workspace/kaggle/covid19/data/custom_license/custom_license/ -o ../../workspace/kaggle/covid19/data/clean_articles.csv --sentences