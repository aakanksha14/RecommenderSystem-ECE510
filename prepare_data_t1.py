# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import argparse
import pickle
import json
import pdb
import cv2

from os import listdir
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from widedeep.utils.data_utils import label_encode

from widedeep.models.wdtypes import *
pd.options.mode.chained_assignment = None

def prepare_deep(df:pd.DataFrame, embeddings_cols:List[Union[str, Tuple[str,int]]],
    continuous_cols:List[str], standardize_cols:List[str], scale:bool=True, def_dim:int=8):
    """
    Highly customised function to prepare the features that will be passed
    through the "Deep-Dense" model.

    Parameters:
    ----------
    df: pd.Dataframe
    embeddings_cols: List
        List containing just the name of the columns that will be represented
        with embeddings or a Tuple with the name and the embedding dimension.
        e.g.:  [('education',32), ('relationship',16)
    continuous_cols: List
        List with the name of the so called continuous cols
    standardize_cols: List
        List with the name of the continuous cols that will be Standarised.
        Only included because the Airbnb dataset includes Longitude and
        Latitude and does not make sense to normalise that
    scale: bool
        whether or not to scale/Standarise continuous cols. Should almost
        always be True.
    def_dim: int
        Default dimension for the embeddings used in the Deep-Dense model

    Returns:
    df_deep.values: np.ndarray
        array with the prepare input data for the Deep-Dense model
    embeddings_input: List of Tuples
        List containing Tuples with the name of embedding col, number of unique values
        and embedding dimension. e.g. : [(education, 11, 32), ...]
    embeddings_encoding_dict: Dict
        Dict containing the encoding mappings that will be required to recover the
        embeddings once the model has trained
    deep_column_idx: Dict
        Dict containing the index of the embedding columns that will be required to
        slice the tensors when training the model
    """
    # If embeddings_cols does not include the embeddings dimensions it will be
    # set as def_dim (8)
    if type(embeddings_cols[0]) is tuple:
        emb_dim = dict(embeddings_cols)
        embeddings_coln = [emb[0] for emb in embeddings_cols]
    else:
        emb_dim = {e:def_dim for e in embeddings_cols}
        embeddings_coln = embeddings_cols
    deep_cols = embeddings_coln+continuous_cols

    # copy the df so it does not change internally
    df_deep = df.copy()[deep_cols]

    # Extract the categorical column names that will be label_encoded
    categorical_columns = list(df_deep.select_dtypes(include=['object']).columns)
    categorical_columns+= list(set([c for c in df_deep.columns if 'catg' in c]))

    # Encode the dataframe and get the encoding dictionary
    df_deep, encoding_dict = label_encode(df_deep, cols=categorical_columns)
    embeddings_encoding_dict = {k:encoding_dict[k] for k in encoding_dict if k in deep_cols}
    embeddings_input = []
    for k,v in embeddings_encoding_dict.items():
        embeddings_input.append((k, len(v), emb_dim[k]))

    # select the deep_cols and get the column index that will be use later
    # to slice the tensors
    deep_column_idx = {k:v for v,k in enumerate(df_deep.columns)}

    # The continous columns will be concatenated with the embeddings, so you
    # probably want to normalize them
    if scale:
        scaler = StandardScaler()
        for cc in standardize_cols:
            df_deep[cc]  = scaler.fit_transform(df_deep[cc].values.reshape(-1,1).astype(float))

    return df_deep.values, embeddings_input, embeddings_encoding_dict, deep_column_idx


def prepare_wide(df:pd.DataFrame, target:str, wide_cols:List[str],
    crossed_cols:List[Tuple[str,str]], already_dummies:Optional[List[str]]=None):
    """
    Highly customised function to prepare the features that will be passed
    through the "Wide" model.

    Parameters:
    ----------
    df: pd.Dataframe
    target: str
    wide_cols: List
        List with the name of the columns that will be one-hot encoded and
        pass through the Wide model
    crossed_cols: List
        List of Tuples with the name of the columns that will be "crossed"
        and then one-hot encoded. e.g. (['education', 'occupation'], ...)
    already_dummies: List
        List of columns that are already dummies/one-hot encoded

    Returns:
    df_wide.values: np.ndarray
        values that will be passed through the Wide Model
    y: np.ndarray
        target
    """
    y = np.array(df[target])
    df_wide = df.copy()[wide_cols]

    crossed_columns = []
    for cols in crossed_cols:
        colname = '_'.join(cols)
        df_wide[colname] = df_wide[cols].apply(lambda x: '-'.join(x), axis = 1)
        crossed_columns.append(colname)

    if already_dummies:
        dummy_cols = [c for c in wide_cols+crossed_columns if c not in already_dummies]
    else:
        dummy_cols = wide_cols+crossed_columns
    df_wide = pd.get_dummies(df_wide, columns=dummy_cols)

    return df_wide.values, y

def prepare_data_adult(df, wide_cols, crossed_cols, embeddings_cols, continuous_cols,
    standardize_cols, target, out_dir, scale=True, def_dim=8, seed=1, save=True):

    dfc = df.copy()

    X_deep, cat_embed_inp, cat_embed_encoding_dict, deep_column_idx = \
        prepare_deep(dfc, embeddings_cols, continuous_cols, standardize_cols, scale=True,
            def_dim=8)

    X_wide, y = prepare_wide(dfc, target, wide_cols, crossed_cols)

    # train/valid/test split
    X_tr_wide, X_val_wide = train_test_split(X_wide, test_size=0.4, random_state=seed)
    X_tr_deep, X_val_deep = train_test_split(X_deep, test_size=0.4, random_state=seed)
    y_tr, y_val = train_test_split(y, test_size=0.4, random_state=seed)

    X_val_wide, X_te_wide = train_test_split(X_val_wide, test_size=0.5, random_state=seed)
    X_val_deep, X_te_deep = train_test_split(X_val_deep, test_size=0.5, random_state=seed)
    y_val, y_te = train_test_split(y_val, test_size=0.5, random_state=seed)

    wd_dataset = dict(
        train = dict(
            wide = X_tr_wide.astype('float32'),
            deep_dense = X_tr_deep,
            target = y_tr,
            ),
        valid = dict(
            wide = X_val_wide.astype('float32'),
            deep_dense = X_val_deep,
            target = y_val,
            ),
        test = dict(
            wide = X_te_wide.astype('float32'),
            deep_dense = X_te_deep,
            target = y_te,
            ),
        cat_embeddings_input = cat_embed_inp,
        cat_embeddings_encoding_dict = cat_embed_encoding_dict,
        continuous_cols = continuous_cols,
        deep_column_idx = deep_column_idx
        )
    if save: pickle.dump(wd_dataset, open('data/IMDB/wd_dataset.p', 'wb'))
    print('Wide and Deep adult data preparation completed.')
    return wd_dataset