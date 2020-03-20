# -*- coding: utf-8 -*-
#
#----------------------------------------------------------
# script name: 
#----------------------------------------------------------
# creator: zhidong.lu
# create date: 2019-05-30
# update date: 2019-05-30
# version: 1.0
#----------------------------------------------------------
#
#       
#----------------------------------------------------------


import os
import re
import sys
import csv
import json
import time
import pytz
import datetime
from collections import OrderedDict
from itertools import product
import pickle

import gc
import multiprocessing

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import random


pd.pandas.set_option("display.max_columns", None)
pd.pandas.set_option("display.max_rows", 300)

plt.style.use({"figure.figsize": [s0*3 for s0 in (4, 2)]})
sns.set(style="whitegrid", rc={"figure.figsize": [s0*3 for s0 in (4, 2)]})



###########################################################################
# 统计dataframe的数据描述
def func_dataframe_describe(in_df, var_names=None, drop_labels=None):
    describe_info = (in_df if var_names==None else in_df[var_names]).drop(labels=([] if drop_labels==None else drop_labels), axis=1) \
        .groupby(axis=1, level=0, sort=False).apply(
        lambda s0: OrderedDict({
            "data_type": ("Numerical" if re.search("(float|int)", s0.dtypes[0].name)!=None else "Categorical"),
            "count": s0.shape[0],
            "count_missing": s0[s0.iloc[:, 0].isna()].shape[0],
            "count_nomissing": s0[-s0.iloc[:, 0].isna()].shape[0],
            "pct_missing": s0[s0.iloc[:, 0].isna()].shape[0]/s0.shape[0],
            "pct_nomissing": s0[-s0.iloc[:, 0].isna()].shape[0]/s0.shape[0],
            "unique_count": s0.iloc[:, 0].unique().shape[0],
            "unique_pct": s0.iloc[:, 0].unique().shape[0]/s0.shape[0],
            
#             "min": (s0.iloc[:, 0].dropna().min() if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "min": (s0.iloc[:, 0].dropna().min() if (s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "mean": (s0.iloc[:, 0].dropna().mean() if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
#             "max": (s0.iloc[:, 0].dropna().max() if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "max": (s0.iloc[:, 0].dropna().max() if (s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            
            "percentile_05": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.05) if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "percentile_25": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.25) if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "percentile_50": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.50) if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "percentile_75": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.75) if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "percentile_95": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.95) if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),

            "top5_value": str(dict(s0.iloc[:, 0].dropna().value_counts().reset_index().values[:5, :])),

            "entropy": np.sum([-t*np.log2(t+1e-20) for t in s0.iloc[:, 0].value_counts()/s0.shape[0]]),
            "entropy_ratio": np.sum([-t*np.log2(t+1e-20) for t in s0.iloc[:, 0].value_counts()/s0.shape[0]])/(np.log(s0.iloc[:, 0].unique().shape[0]+1e-20)+1e-20),
        })
    )
    df_describe_info = pd.DataFrame(describe_info.values.tolist(), index=describe_info.index.values) \
        .reset_index().rename(columns={"index": "column_name"}) \
        .set_index(keys=["column_name"])
    return df_describe_info


###########################################################################
# 统计dataframe对应字段的值频数分布
def func_freqency_stat(in_df, var_names=None, drop_labels=None):
    _tmp = [[
        _col,
        _df.dtypes[0].name,
        _df.shape[0],
        _df.iloc[:, 0].value_counts(dropna=False),
    ] for _col, _df in (in_df if var_names==None else in_df[var_names]).drop(labels=([] if drop_labels==None else drop_labels), axis=1) \
            .groupby(axis=1, level=0, sort=False)
    ]
    df_freq_table = pd.concat([
        pd.DataFrame([OrderedDict({
            "column_name": _col,
            "data_type": ("Numerical" if re.search("(float|int)", _dtype_name)!=None else "Categorical"),
            "value": ("NaN" if pd.isna(_value) else _value),
            "count": _count,
            "count_cum": _count_cum,
            "pct": _count/_df_size,
            "pct_cum": _count_cum/_df_size,
        }) for _value, _count, _count_cum in zip(_data.index.tolist(), _data.values, _data.cumsum())])
        for _col, _dtype_name, _df_size, _data in _tmp
    ], ignore_index=True)
    
    return df_freq_table


###########################################################################
# 计算关联表匹配率（主键匹配率、外键匹配率）
def func_table_match_rate(df_primary, var_foreign,
                          df_foreign, var_f_primary):
    ######################################################
    _tmp = df_primary[var_foreign].merge(
        right=df_foreign[var_f_primary].reset_index(),
        how="left", left_on=var_foreign, right_on=var_f_primary,
    )
    primary_match_rate = _tmp[_tmp["index"].notna()].shape[0]/_tmp.shape[0]
    
    ######################################################
    _tmp = df_foreign[var_f_primary].merge(
        right=df_primary[var_foreign].drop_duplicates().reset_index(),
        how="left", left_on=var_f_primary, right_on=var_foreign,
    )
    foreign_match_rate = _tmp[_tmp["index"].notna()].shape[0]/_tmp.shape[0]
    
    rt = {
        "primary_match_rate": primary_match_rate,
        "foreign_match_rate": foreign_match_rate,
    }
    return rt


###########################################################################
# 分箱（连续型变量）
def func_binning_continuous_v1(in_data, bins, fillna_value=-9999, right_border=True, include_lowest=False, reverse_label=False):
    bins = pd.Series(bins).sort_values().unique().tolist()
    bins_cnt = len(bins)-1
    x = pd.Series(in_data).fillna(value=fillna_value)
    rt = pd.cut(
        x=x,
        bins=bins, right=right_border, include_lowest=include_lowest,
        labels=[s0 for s0 in range(bins_cnt)], 
    ).astype(int).apply(lambda s0: "{:0{}d}_{}{:.4f}, {:.4f}{}".format((bins_cnt-1-s0 if reverse_label else s0)+1, int(np.log10(len(bins)))+1, ("(" if right_border else "["), bins[s0], bins[s0+1], ("]" if right_border else ")")))
    return rt

def func_binning_continuous_v2(in_df, var_name, bins, fillna_value=-9999, right_border=True, include_lowest=False, reverse_label=False):
    rt = func_binning_continuous_v1(
        in_data=in_df[var_name], bins=bins, 
        fillna_value=fillna_value, right_border=right_border, include_lowest=include_lowest, reverse_label=reverse_label,
    )
    return rt


###########################################################################
# 合箱（离散型变量）
def func_combining_discrete_v1(in_data, mapping_gb_class, fillna_value="NaN", cvt_fillna_value=1, reverse_label=False):
    mapping_gb_class[fillna_value] = cvt_fillna_value
    if reverse_label:
        _value_max = np.max(list(mapping_gb_class.values()))
        mapping_gb_class = dict([(_k, _value_max-_v+1) for _k, _v in mapping_gb_class.items()])
    in_data = pd.Series([(s0 if s0 in set(mapping_gb_class.keys()) else "NaN") for s0 in pd.Series(in_data).fillna("NaN")])
    rt = in_data.apply(lambda s0: mapping_gb_class.get(s0))
    return rt

def func_combining_discrete_v2(in_df, var_name, fillna_value="NaN", cvt_fillna_value=1, reverse_label=False):
    rt = func_binning_continuous_v1(
        in_data=in_df[var_name], mapping_gb_class=mapping_gb_class,
        fillna_value=fillna_value, cvt_fillna_value=cvt_fillna_value, reverse_label=reverse_label,
    )
    return rt


###########################################################################
# 自动分箱（连续型变量）
def func_auto_binning_continuous_v1(in_var, in_target, min_pct=0.05, max_bins_cnt=None, fillna_value=-9999, right_border=True, include_lowest=False, method="01_decision_tree"):
    max_bins_cnt = (int(1/min_pct)+1 if max_bins_cnt==None else max_bins_cnt)
    data = pd.Series(in_var).fillna(value=fillna_value)
    data_size = data.shape[0]
    _target_labels = in_target.unique().tolist()
    
    #####################################################################################
    if method=="01_decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(
            criterion='entropy',
            max_leaf_nodes=max_bins_cnt,
            min_samples_leaf=min_pct,
        )
        clf.fit(X=data.values.reshape(-1, 1), y=in_target)

        boundary = [clf.tree_.threshold[idx] for idx in range(clf.tree_.node_count) if clf.tree_.children_left[idx]!=clf.tree_.children_right[idx]]
        boundary.sort()
        boundary = [-np.inf]+boundary+[np.inf]
    
    #####################################################################################
    elif method=="02_best_ks":
        crosstab = pd.crosstab(index=data, columns=in_target)
        crosstab.index.name = "index"
        crosstab.columns.name = ""
        crosstab = crosstab.reset_index().rename(columns={"index": "value"})

        def _func_ks_stat(in_crosstab):
            if in_crosstab.shape[0]>0:
                crosstab = in_crosstab.reset_index(drop=True)
                crosstab[["{}_%".format(s0) for s0 in _target_labels]] = (crosstab[_target_labels]/crosstab[_target_labels].sum()).fillna(0)
                crosstab[["{}_%_cum".format(s0) for s0 in _target_labels]] = crosstab[["{}_%".format(s0) for s0 in _target_labels]].cumsum()
                crosstab["diff_abs"] = crosstab[["{}_%_cum".format(s0) for s0 in _target_labels]].apply(lambda s0: abs(s0[0]-s0[1]), axis=1)
                ks_value = crosstab["diff_abs"].max()
                return crosstab, ks_value
            else:
                return None, None

        def _func_cut_ks_value(in_crosstab, ks_value):
            crosstab = in_crosstab.reset_index(drop=True)
            cut_stats = crosstab[crosstab["diff_abs"]==ks_value]
            cut_value = cut_stats["value"].values[0]
            crosstab_left_eq, ks_value_left_eq = _func_ks_stat(in_crosstab=crosstab.query("value<={}".format(cut_value))[["value"]+_target_labels])
            crosstab_right, ks_value_right = _func_ks_stat(crosstab.query("value>{}".format(cut_value))[["value"]+_target_labels])
            return cut_value, crosstab_left_eq, crosstab_right, ks_value_left_eq, ks_value_right

        crosstab, ks_value = _func_ks_stat(in_crosstab=crosstab)
        cut_value, _, _, _, _ = _func_cut_ks_value(in_crosstab=crosstab, ks_value=ks_value)
        _crosstab_pending = [{"ks_value": ks_value, "crosstab": crosstab, "size": crosstab[_target_labels].sum().sum()}]
        boundary = [cut_value]
        while 1:
            _max_ks = max([s0["ks_value"] for s0 in _crosstab_pending])
            _t1 = [s0 for s0 in _crosstab_pending if s0["ks_value"]==_max_ks]
            _t2 = [s0 for s0 in _crosstab_pending if s0["ks_value"]!=_max_ks]
#             _max_size = max([s0["size"] for s0 in _crosstab_pending])
#             _t1 = [s0 for s0 in _crosstab_pending if s0["size"]==_max_size]
#             _t2 = [s0 for s0 in _crosstab_pending if s0["size"]!=_max_size]
            _crosstab = _t1[0]["crosstab"]
            _crosstab_pending = _t1[1:]+_t2

            if _crosstab[_target_labels].sum().sum()/data_size>=min_pct and _crosstab.shape[0]>1:
                _crosstab, ks_value = _func_ks_stat(in_crosstab=_crosstab)
                cut_value, crosstab_left_eq, crosstab_right, ks_value_left_eq, ks_value_right = _func_cut_ks_value(in_crosstab=_crosstab, ks_value=ks_value)

                if ((ks_value_left_eq!=None and ks_value_right!=None) and \
                    (crosstab_left_eq[_target_labels].sum().sum()/data_size>=min_pct and crosstab_right[_target_labels].sum().sum()/data_size>=min_pct)):
                    boundary.append(cut_value)
                    boundary = list(set(boundary))
                    _crosstab_pending = _crosstab_pending+[{"ks_value": ks_value_left_eq, "crosstab": crosstab_left_eq, "size": crosstab_left_eq[_target_labels].sum().sum()},
                                                           {"ks_value": ks_value_right, "crosstab": crosstab_right, "size": crosstab_right[_target_labels].sum().sum()}]

            if len(boundary)+1>=max_bins_cnt or len(_crosstab_pending)==0:
                break
        boundary.sort()
        boundary = [-np.inf]+boundary+[np.inf]
    #####################################################################################
    
    data_converted = func_binning_continuous_v1(
        in_data=data, bins=boundary, fillna_value=fillna_value,
        right_border=right_border, include_lowest=include_lowest, reverse_label=False,
    )
    crosstab_converted = func_woe_report_v1(in_var=data_converted, in_target=in_target, with_total=True)
    
    return data_converted, crosstab_converted, boundary

def func_auto_binning_continuous_v2(in_df, var_name, target_label, min_pct=0.05, max_bins_cnt=None, fillna_value=-9999, right_border=True, include_lowest=False, method="01_decision_tree"):
    data_converted, crosstab_converted, boundary = \
        func_auto_binning_continuous_v1(
            in_var=in_df[var_name], in_target=in_df[target_label],
            min_pct=min_pct, max_bins_cnt=max_bins_cnt, fillna_value=fillna_value, right_border=right_border, include_lowest=include_lowest, method=method,
        )
    return data_converted, crosstab_converted, boundary


###########################################################################
# 自动合箱（离散型变量）
def func_auto_combining_discrete_v1(in_var, in_target, min_pct=0.05, max_bins_cnt=None, method="01_equal_width"):
    max_bins_cnt = (np.ceil(1/min_pct) if max_bins_cnt==None else max_bins_cnt)
    max_pct = (min_pct if 1/max_bins_cnt<min_pct else 1/max_bins_cnt)
    data = pd.Series(in_var).fillna("NaN")
    data_size = data.shape[0]
    _target_labels = in_target.unique().tolist()
    
    crosstab = func_woe_report_v1(in_var=data, in_target=in_target, with_total=False)
    crosstab = crosstab.rename(columns=dict([("{}_#".format(t), t) for t in _target_labels]))
    crosstab = crosstab.sort_values(by=["bad_rate", "total_pct"], ascending=[False, False])
    crosstab.index.name = "index"
    crosstab = crosstab.reset_index().rename(columns={"index": "value"}).reset_index().rename(columns={"index": "idx"})
    
    #####################################################################################
    if method=="01_equal_width":
        gb_class = 1
        total_pct_cum = 0
        total_pct_remain = 1
        _mapping_data = []
        for idx, good_cnt, bad_cnt, total_pct in crosstab[["value"]+_target_labels+["total_pct"]].values[:]:
            if good_cnt==0 or bad_cnt==0:
                total_pct_cum = total_pct_cum+total_pct
            else:
                if total_pct_cum<=max_pct:
                    total_pct_cum = total_pct_cum+total_pct
                else:
                    if gb_class+1<=max_bins_cnt:
                        if total_pct_remain>=min_pct:
                            gb_class = gb_class+1
                        total_pct_cum = total_pct
                    else:
                        total_pct_cum = total_pct_cum+total_pct
                        gb_class = max_bins_cnt

            total_pct_remain = total_pct_remain-total_pct
            _mapping_data.append([idx, gb_class])
            # print(idx, gb_class, total_pct, total_pct_cum, total_pct_remain)
    
    #####################################################################################
    elif method=="02_best_ks":
        def _func_ks_stat(in_crosstab):
            if in_crosstab.shape[0]>0:
                crosstab = in_crosstab.reset_index(drop=True)
                crosstab[["{}_%".format(s0) for s0 in _target_labels]] = (crosstab[_target_labels]/crosstab[_target_labels].sum()).fillna(0)
                crosstab[["{}_%_cum".format(s0) for s0 in _target_labels]] = crosstab[["{}_%".format(s0) for s0 in _target_labels]].cumsum()
                crosstab["diff_abs"] = crosstab[["{}_%_cum".format(s0) for s0 in _target_labels]].apply(lambda s0: abs(s0[0]-s0[1]), axis=1)
                ks_value = crosstab["diff_abs"].max()
                return crosstab, ks_value
            else:
                return None, None

        def _func_cut_ks_value(in_crosstab, ks_value):
            crosstab = in_crosstab.reset_index(drop=True)
            cut_stats = crosstab[crosstab["diff_abs"]==ks_value]
            cut_idx = cut_stats["idx"].values[0]
            crosstab_left_eq, ks_value_left_eq = _func_ks_stat(in_crosstab=crosstab.query("idx<={}".format(cut_idx))[["idx", "value"]+_target_labels])
            crosstab_right, ks_value_right = _func_ks_stat(crosstab.query("idx>{}".format(cut_idx))[["idx", "value"]+_target_labels])
            return cut_idx, crosstab_left_eq, crosstab_right, ks_value_left_eq, ks_value_right

        crosstab, ks_value = _func_ks_stat(in_crosstab=crosstab)
        cut_idx, _, _, _, _ = _func_cut_ks_value(in_crosstab=crosstab, ks_value=ks_value)
        _crosstab_pending = [{"ks_value": ks_value, "crosstab": crosstab, "size": crosstab[_target_labels].sum().sum()}]
        boundary_idx = [cut_idx]
        while 1:
            _max_ks = max([s0["ks_value"] for s0 in _crosstab_pending])
            _t1 = [s0 for s0 in _crosstab_pending if s0["ks_value"]==_max_ks]
            _t2 = [s0 for s0 in _crosstab_pending if s0["ks_value"]!=_max_ks]
#             _max_size = max([s0["size"] for s0 in _crosstab_pending])
#             _t1 = [s0 for s0 in _crosstab_pending if s0["size"]==_max_size]
#             _t2 = [s0 for s0 in _crosstab_pending if s0["size"]!=_max_size]
            _crosstab = _t1[0]["crosstab"]
            _crosstab_pending = _t1[1:]+_t2

            if _crosstab[_target_labels].sum().sum()/data_size>=min_pct and _crosstab.shape[0]>1:
                _crosstab, ks_value = _func_ks_stat(in_crosstab=_crosstab)
                cut_idx, crosstab_left_eq, crosstab_right, ks_value_left_eq, ks_value_right = _func_cut_ks_value(in_crosstab=_crosstab, ks_value=ks_value)

                if ((ks_value_left_eq!=None and ks_value_right!=None) and \
                    (crosstab_left_eq[_target_labels].sum().sum()/data_size>=min_pct and crosstab_right[_target_labels].sum().sum()/data_size>=min_pct)):
                    boundary_idx.append(cut_idx)
                    boundary_idx = list(set(boundary_idx))
                    _crosstab_pending = _crosstab_pending+[{"ks_value": ks_value_left_eq, "crosstab": crosstab_left_eq, "size": crosstab_left_eq[_target_labels].sum().sum()},
                                                           {"ks_value": ks_value_right, "crosstab": crosstab_right, "size": crosstab_right[_target_labels].sum().sum()}]

            if len(boundary_idx)+1>=max_bins_cnt or len(_crosstab_pending)==0:
                break
        boundary_idx.sort()

        crosstab["retain_flag"] = crosstab["idx"].apply(lambda s0: s0 in boundary_idx)
        _mapping_data = []
        gb = 1
        for _value, _retain_flag in zip(crosstab["value"], crosstab["retain_flag"]):
            _mapping_data.append([_value, gb])
            if _retain_flag==True:
                gb = gb+1
    
    #####################################################################################
    mapping_gb_class = dict(_mapping_data)
    data_converted = data.apply(lambda s0: mapping_gb_class.get(s0))
    crosstab_converted = func_woe_report_v1(in_var=data_converted, in_target=in_target, with_total=True)
    return data_converted, crosstab_converted, mapping_gb_class

def func_auto_combining_discrete_v2(in_df, var_name, target_label, min_pct=0.05, max_bins_cnt=None, method="01_equal_width"):
    data_converted, crosstab_converted, mapping_gb_class = \
        func_auto_combining_discrete_v1(
            in_var=in_df[var_name], in_target=in_df[target_label],
            min_pct=min_pct, max_bins_cnt=max_bins_cnt, method=method,
        )
    return data_converted, crosstab_converted, mapping_gb_class


###########################################################################
# 计算变量WOE报告
def func_woe_report_v1(in_var, in_target, with_total=True):
    in_var = pd.Series(in_var).fillna(value="NaN")
    
    crosstab = pd.crosstab(index=in_var, columns=in_target)
    _rename = columns=dict(zip(crosstab.columns.tolist(), ["{}_#".format(s0) for s0 in crosstab.columns.tolist()]))
    crosstab[["{}_%".format(s0) for s0 in crosstab.columns.tolist()]] = crosstab[crosstab.columns.tolist()]/crosstab.sum(axis=0)
    crosstab = crosstab.rename(columns=_rename)

    crosstab["WOE"] = crosstab[[s0 for s0 in crosstab.columns.tolist() if ("%" in s0)]].apply(lambda s0: np.log(s0[0]+1e-20)-np.log(s0[1]+1e-20), axis=1)
    crosstab["IV"] = crosstab[[s0 for s0 in crosstab.columns.tolist() if ("%" in s0)]+["WOE"]].apply(lambda s0: (s0[0]-s0[1])*s0["WOE"], axis=1)
    crosstab["total"] = crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]].apply(lambda s0: s0.sum(), axis=1).astype(int)
    crosstab["total_pct"] = crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]].sum(axis=1)/crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]].values.sum()
    
    if with_total:
        _total = pd.DataFrame(crosstab.sum(axis=0), columns=["total"]).T
        _total["WOE"] = np.NaN
        crosstab = crosstab.append(_total)
        crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]+["total"]] = crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]+["total"]].astype(int)
    
    crosstab["bad_rate"] = crosstab.iloc[:, 1]/crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]].sum(axis=1)
    crosstab.columns.name = ""
    return crosstab

def func_woe_report_v2(in_df, var_name, target_label, with_total=True):
    crosstab = func_woe_report_v1(in_var=in_df[var_name], in_target=in_df[target_label], with_total=with_total)
    return crosstab


###########################################################################
# 计算KS
def func_calc_ks_cross(y_labels, y_pred, plot=False):
    crossfreq = pd.crosstab(index=y_pred, columns=y_labels)
    crossfreq.index.name = "predict_prob"
    crossfreq.columns.name = ""
    
    crossdens = crossfreq.cumsum(axis=0)/crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0]-crossdens[1])
    ks = crossdens[crossdens['gap']==crossdens['gap'].max()]
    if plot:
        _data = pd.DataFrame(np.concatenate([
            y_labels.reshape([-1, 1]),
            y_pred.reshape([-1, 1]),
        ], axis=1), columns=["TARGET_label_bad", "PRED_bad_prob"]).astype(dtype={
            "TARGET_label_bad": np.int,
            "PRED_bad_prob": np.float,
        })
        sns.violinplot(data=_data, y="PRED_bad_prob", x="TARGET_label_bad", hue="TARGET_label_bad")
        
        crossdens.plot(kind="line")
        plt.show()
    return ks, crossdens


###########################################################################
# 计算AUC_ROC
def func_calc_auc_roc(y_labels, y_pred, plot=False):
    from sklearn import metrics
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_labels, y_pred)
    auc = metrics.auc(false_positive_rate, true_positive_rate)
    if plot:
        plt.style.use({"figure.figsize": [s0*3 for s0 in (2, 2)]})
        plt.title('ROC')
        plt.plot(false_positive_rate, true_positive_rate, color='b', label='AUC = {:0.4f}'.format(auc))
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.show()
        plt.style.use({"figure.figsize": [s0*3 for s0 in (4, 2)]})
        plt.show()
    return auc


###########################################################################
# 计算LIFT
def func_calc_lift(y_labels, y_pred, bucket_cnt=20, bad_label="bad", plot=False):
    crosstab = pd.crosstab(index=y_pred, columns=y_labels)
    crosstab.index.name = "predict_prob"
    crosstab.columns.name = ""
    crosstab = crosstab.reset_index().sort_values(by=["predict_prob"], ascending=[False]).reset_index(drop=True)

    mapping_bucket_idx = dict([(idx, min(int(idx/(crosstab.shape[0]//bucket_cnt))+1, 20)) for idx in crosstab.index.tolist()])
    crosstab = crosstab.reset_index().rename(columns={"index": "bucket_idx"})
    crosstab["bucket_idx"] = crosstab["bucket_idx"].apply(lambda s0: mapping_bucket_idx.get(s0))

    crosstab_bucket = pd.DataFrame([OrderedDict({
        "bucket_idx": _bucket_idx,
        "bucket_cum_pct": int(_bucket_idx*(100/bucket_cnt)),
        "predict_prob_gte": _df["predict_prob"].min(),
        "obs_cnt": _df.shape[0],
        "bad_cnt": _df[bad_label].sum(),
    }) for _bucket_idx, _df in crosstab.groupby(by=["bucket_idx"])])
    crosstab_bucket["bad_pct"] = crosstab_bucket["bad_cnt"]/crosstab_bucket["bad_cnt"].sum()
    crosstab_bucket["obs_pct"] = crosstab_bucket["obs_cnt"]/crosstab_bucket["obs_cnt"].sum()
    crosstab_bucket["lift"] = crosstab_bucket["bad_pct"]/crosstab_bucket["obs_pct"]

    if plot:
        crosstab_bucket.set_index(keys=["bucket_cum_pct"])[["lift"]].plot(kind="line")
        crosstab_bucket.set_index(keys=["bucket_cum_pct"])[["bad_pct", "obs_pct"]].plot(kind="bar")
        plt.show()
    return crosstab_bucket


###########################################################################
# 计算PSI

# 计算离散变量PSI
def func_calc_psi_discrete_v1(in_data_actual, in_data_expected, plot=False):
    in_data_actual = pd.Series(in_data_actual).fillna("NaN")
    in_data_expected = pd.Series(in_data_expected).fillna("NaN")
    
    psi_table = pd.merge(
        left=pd.DataFrame([{
            "data_label": _data_label,
            "actual_cnt": _cnt,
        } for _data_label, _cnt in in_data_actual.value_counts().sort_index().reset_index().values]),
        right=pd.DataFrame([{
            "data_label": _data_label,
            "expected_cnt": _cnt,
        } for _data_label, _cnt in in_data_expected.value_counts().sort_index().reset_index().values]),
        how="outer", on="data_label",
    )[["data_label", "actual_cnt", "expected_cnt"]]
    psi_table[["actual_pct", "expected_pct"]] = psi_table[["actual_cnt", "expected_cnt"]].apply(lambda s0: s0/s0.sum())
    psi_table["minus_act_exp"] = psi_table["actual_pct"] - psi_table["expected_pct"]
    psi_table["ln_act_exp"] = np.log((psi_table["actual_pct"]+1e-20)/(psi_table["expected_pct"]+1e-20))
    psi_table["Index"] = psi_table["minus_act_exp"]*psi_table["ln_act_exp"]

    psi = psi_table["Index"].sum()
    if plot:
        psi_table.set_index(keys=["data_label"])[["actual_pct", "expected_pct"]].plot(kind="bar")
        plt.show()
    return psi, psi_table

def func_calc_psi_discrete_v2(in_df, actual_label, expected_label, plot=False):
    psi, psi_table = func_calc_psi_discrete_v1(
        in_data_actual=in_df[actual_label],
        in_data_expected=in_df[expected_label],
        plot=plot)
    return psi, psi_table
    

# 计算连续变量PSI，等距分箱操作
def func_calc_psi_continuous_v1(in_data_actual, in_data_expected, bins_cnt=10, plot=False):
    in_data_actual = pd.Series(in_data_actual)
    in_data_expected = pd.Series(in_data_expected)
    
    _cut_left = in_data_expected.min()
    _cut_right = in_data_expected.max()
    cut_bins = [-np.inf]+[_cut_left+((_cut_right-_cut_left)/bins_cnt)*s0 for s0 in range(1, bins_cnt)]+[np.inf]
    in_data_actual_binning = func_binning_continuous_v1(
        in_data=in_data_actual,
        bins=cut_bins,
        fillna_value=-9999, right_border=True, include_lowest=True, reverse_label=False)
    in_data_expected_binning = func_binning_continuous_v1(
        in_data=in_data_expected,
        bins=cut_bins,
        fillna_value=-9999, right_border=True, include_lowest=True, reverse_label=False)
    
    psi, psi_table = func_calc_psi_discrete_v1(
        in_data_actual=in_data_actual_binning, in_data_expected=in_data_expected_binning, plot=plot)
    return psi, psi_table

def func_calc_psi_continuous_v2(in_df, actual_label, expected_label, bins_cnt=10, plot=False):
    psi, psi_table = func_calc_psi_continuous_v1(
        in_data_actual=in_df[actual_label],
        in_data_expected=in_df[expected_label],
        bins_cnt=bins_cnt,
        plot=plot)
    return psi, psi_table


# 按月度频率计算离散变量PSI
def func_calc_psi_discrete_features_monthly(in_data_features, in_data_YM, base_last_months=4, verbose=0):
    in_data_features = in_data_features.reset_index(drop=True)
    in_data_YM = pd.DataFrame(pd.Series(in_data_YM).values, columns=["data_dt"])
    _data = pd.merge(left=in_data_YM, right=in_data_features,
                     how="left", left_index=True, right_index=True)
    feature_names = in_data_features.columns.tolist()
    data_dt = in_data_YM["data_dt"].drop_duplicates().sort_values().tolist()
    mapping_data_dt_for_calc = dict(
        [(s0, [t for t in data_dt if t<s0][-base_last_months:]) for s0 in data_dt[1:]]
    )
    mapping_gb_data_dt = dict([(_dt, _df[feature_names]) for _dt, _df in _data.groupby(by=["data_dt"])])
    
    _features_psi = []
    if verbose:
        print("========================================")
        print("{} features pending...".format(len(feature_names)))
    for _idx, _feature_name in enumerate(feature_names):
        _features_psi.append(OrderedDict(
            [("feature_name", _feature_name)]+
            [(_dt, OrderedDict([(s0,
                     func_calc_psi_discrete_v1(
                         in_data_actual=mapping_gb_data_dt.get(_dt)[_feature_name],
                         in_data_expected=mapping_gb_data_dt.get(s0)[_feature_name],
                         plot=False)[0]
                     ) for s0 in _base_dts]))
             for _dt, _base_dts in mapping_data_dt_for_calc.items()][:]
        ))
        if verbose:
            print("[done {}] {}".format(_idx+1, _feature_name))
    df_features_psi = pd.DataFrame(_features_psi).set_index(keys=["feature_name"])
    df_features_psi_summary = df_features_psi.applymap(lambda s0: pd.Series(s0).mean())
    return df_features_psi_summary


if __name__=="__main__":
    pass

