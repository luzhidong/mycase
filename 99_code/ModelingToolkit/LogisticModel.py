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

import numpy as np
import pandas as pd
from collections import OrderedDict

import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')


#############################################################################################
# colinearity
def func_colinearity_rt_col(in_df_features, corr_threshold=0.6, corr_method="pearson"):
    corr_kendall_table = in_df_features.corr(method=corr_method)
    corr_over_threshold = corr_kendall_table.applymap(lambda s0: np.abs(s0)>corr_threshold)
    col = corr_over_threshold.columns.tolist()
    i = 0
    while i<len(col)-1:
        exclude_var = corr_over_threshold.iloc[:,i][i+1:].index[np.where(corr_over_threshold.iloc[:,i][i+1:])].tolist()
        for var in exclude_var:
            col.remove(var)
        corr_over_threshold = corr_over_threshold.loc[col, col]
        i = i+1
    return col, corr_kendall_table


#############################################################################################
# multicolinearity
# VIF
def func_multicolinearity_vif(in_df_features, with_constant=True):
    df = in_df_features.reset_index(drop=True)
    if with_constant:
        df["constant"] = 1
    vif_table = pd.DataFrame([OrderedDict({
        "feature_name": _col,
        "VIF": variance_inflation_factor(exog=df.values, exog_idx=_idx),
    }) for _idx, _col in enumerate(df.columns.tolist())])
    return vif_table


############################################################################
def func_logistic_model(df_X, y, maxiter=100, with_intercept=True, disp=False):
    # warnings.filterwarnings('ignore')
    _exog_X = df_X.reset_index(drop=True)
    if with_intercept:
        _exog_X["Intercept"] = 1
        _exog_X = _exog_X[["Intercept"]+df_X.columns.tolist()]
    _endog_y = y
    
    logistic_model = sm.Logit(
        endog=_endog_y,
        exog=_exog_X,
    )
    logistic_model_res = logistic_model.fit(
        method="newton",
        maxiter=maxiter,
        disp=disp,
    )
    # warnings.filterwarnings('default')
    return logistic_model_res

############################################################################
# stepwise
def func_logistic_model_stepwise(df_X, y,
                                 initial_cols=None, include_cols=None, exclude_cols=None,
                                 sle=0.01, sls=0.05, with_intercept=True, verbose=1):
    initial_cols = (initial_cols if initial_cols!=None else [])
    include_cols = (include_cols if include_cols!=None else [])
    exclude_cols = (exclude_cols if exclude_cols!=None else [])
    
    ############################################################################
    include_cols = [s0 for s0 in include_cols if (s0 not in exclude_cols)]
    initial_cols = pd.Series(include_cols+initial_cols).unique().tolist()
    model_cols = [s0 for s0 in initial_cols if ((s0 in include_cols) or (s0 not in exclude_cols))]
    remain_cols = [s0 for s0 in df_X.columns.tolist() if (s0 not in model_cols and s0 not in exclude_cols)]
    
    ############################################################################
    selection_step = []
    curr_step = 1
    while len(remain_cols)>0:
        # analysis of eligible for entey
        eligible_for_entey = []
        for _col in remain_cols[:]:
            logistic_model_res = func_logistic_model(
                df_X=df_X[model_cols+[_col]], y=y,
                maxiter=100, with_intercept=with_intercept, disp=False,
            )
            eligible_for_entey.append(
                OrderedDict(zip(["col_name", "score_chi2", "pvalue"], [_col]+logistic_model_res.wald_test_terms().summary_frame().loc[_col, ["chi2", "P>chi2"]].tolist()))
            )
        df_eligible_for_entey = pd.DataFrame(eligible_for_entey).query("pvalue<={}".format(sle)).reset_index(drop=True)

        # entry
        entry_col = None
        if df_eligible_for_entey.shape[0]>0:
            entry = df_eligible_for_entey.query("pvalue=={}".format(df_eligible_for_entey["pvalue"].min())) \
                .sort_values(by=["score_chi2"], ascending=[False])
            entry_col , score_chi2, pvalue = entry.values[0]
            model_cols.append(entry_col)

            if verbose:
                print("==================================================================")
                print("current step {}:".format(curr_step))
                print("    variable [ {} ] entered.".format(entry_col))

            selection_step.append(OrderedDict({
                "step": curr_step,
                "entered_variable": (entry_col if entry_col!=None else ""),
                "removed_variable": "",
                "score_chi2": score_chi2,
                "wald_chi2": None,
                "pvalue": pvalue,
                "_model_variables": ",".join(model_cols),
            }))
            curr_step = curr_step+1
        else:
            print("selection step done.")
            break

        # stay
        if len(model_cols)>0:
            logistic_model_res = func_logistic_model(
                df_X=df_X[model_cols], y=y,
                maxiter=100, with_intercept=with_intercept, disp=False,
            )
            _df_wald_test_terms = logistic_model_res.wald_test_terms().summary_frame() \
                .rename(columns={"chi2": "wald_chi2", "P>chi2": "pvalue"}).loc[model_cols, ["wald_chi2", "pvalue"]] \
                .reset_index().rename(columns={"index": "col_name"})
            df_for_remove = _df_wald_test_terms[-_df_wald_test_terms["col_name"].isin(include_cols)] \
                .query("pvalue>{}".format(sls)).reset_index(drop=True)
            df_for_stay = _df_wald_test_terms[-_df_wald_test_terms["col_name"].isin(df_for_remove["col_name"].tolist())].reset_index(drop=True)
            model_cols = df_for_stay["col_name"].tolist()

            if len(model_cols)==0:
                print("no variable enter model !!")
                break

            remain_cols = [s0 for s0 in df_X.columns.tolist() if (s0 not in model_cols and s0 not in exclude_cols)]
            removed_cols = df_for_remove["col_name"].tolist()

            if verbose:
                if len(removed_cols)>0:
                    print("    variable(s): [ {} ] removed.".format(" / ".join(removed_cols)))
                else:
                    print("    no variable removed.")
                print()

            if len(removed_cols)>0:
                # 判断是否出现死循环
                _model_cols_sorted_string = ",".join(pd.Series(model_cols).sort_values().tolist())
                _model_cols_sorted_string_hist_selection = [",".join(pd.Series(s0.get("_model_variables").split(",")).sort_values().tolist()) for s0 in selection_step]
                if _model_cols_sorted_string in _model_cols_sorted_string_hist_selection:
                    print("selection step done.")
                    break
                else:
                    for _col, wald_chi2, pvalue in df_for_remove.values:
                        selection_step.append(OrderedDict({
                            "step": curr_step,
                            "entered_variable": "",
                            "removed_variable": _col,
                            "score_chi2": None,
                            "wald_chi2": wald_chi2,
                            "pvalue": pvalue,
                            "_model_variables": ",".join(model_cols),
                        }))
                        curr_step = curr_step+1

    df_result_selection_step =  pd.DataFrame(selection_step) \
        [["step", "entered_variable", "removed_variable", "score_chi2", "wald_chi2", "pvalue"]]
    logistic_model_res = func_logistic_model(
        df_X=df_X[model_cols], y=y,
        maxiter=100, with_intercept=with_intercept, disp=False,
    )
    return logistic_model_res, model_cols, df_result_selection_step


############################################################################
# forward
def func_logistic_model_forward(df_X, y,
                                initial_cols=None, exclude_cols=None,
                                sle=0.05, with_intercept=True, verbose=1):
    initial_cols = (initial_cols if initial_cols!=None else [])
    exclude_cols = (exclude_cols if exclude_cols!=None else [])
    
    ############################################################################
    model_cols = [s0 for s0 in initial_cols if (s0 not in exclude_cols)]
    remain_cols = [s0 for s0 in df_X.columns.tolist() if (s0 not in model_cols and s0 not in exclude_cols)]
    
    ############################################################################
    selection_step = []
    curr_step = 1
    while len(remain_cols)>0:
        # analysis of eligible for entey
        eligible_for_entey = []
        for _col in remain_cols[:]:
            logistic_model_res = func_logistic_model(
                df_X=df_X[model_cols+[_col]], y=y,
                maxiter=100, with_intercept=with_intercept, disp=False,
            )
            eligible_for_entey.append(
                OrderedDict(zip(["col_name", "score_chi2", "pvalue"], [_col]+logistic_model_res.wald_test_terms().summary_frame().loc[_col, ["chi2", "P>chi2"]].tolist()))
            )
        df_eligible_for_entey = pd.DataFrame(eligible_for_entey).query("pvalue<={}".format(sle)).reset_index(drop=True)

        # entry
        entry_col = None
        if df_eligible_for_entey.shape[0]>0:
            entry = df_eligible_for_entey.query("pvalue=={}".format(df_eligible_for_entey["pvalue"].min())) \
                .sort_values(by=["score_chi2"], ascending=[False])
            entry_col , score_chi2, pvalue = entry.values[0]
            model_cols.append(entry_col)
            remain_cols = [s0 for s0 in df_X.columns.tolist() if (s0 not in model_cols and s0 not in exclude_cols)]
            
            if verbose:
                print("==================================================================")
                print("current step {}:".format(curr_step))
                print("    variable [ {} ] entered.".format(entry_col))

            selection_step.append(OrderedDict({
                "step": curr_step,
                "entered_variable": (entry_col if entry_col!=None else ""),
                "removed_variable": "",
                "score_chi2": score_chi2,
                "pvalue": pvalue,
                "_model_variables": ",".join(model_cols),
            }))
            curr_step = curr_step+1
        else:
            print("selection step done.")
            break

    df_result_selection_step =  pd.DataFrame(selection_step) \
        [["step", "entered_variable", "removed_variable", "score_chi2",
          "pvalue"]]
    logistic_model_res = func_logistic_model(
        df_X=df_X[model_cols], y=y,
        maxiter=100, with_intercept=with_intercept, disp=False,
    )
    return logistic_model_res, model_cols, df_result_selection_step


############################################################################
# backward
def func_logistic_model_backward(df_X, y,
                                 initial_cols=None, include_cols=None,
                                 sls=0.05, with_intercept=True, verbose=1):
    initial_cols = (initial_cols if initial_cols!=None else df_X.columns.tolist())
    include_cols = (include_cols if include_cols!=None else [])
    
    ############################################################################
    initial_cols = pd.Series(include_cols+initial_cols).unique().tolist()
    model_cols = initial_cols
    
    ############################################################################
    selection_step = []
    curr_step = 1
    while len(model_cols)>0:
        # stay
        logistic_model_res = func_logistic_model(
            df_X=df_X[model_cols], y=y,
            maxiter=100, with_intercept=with_intercept, disp=False,
        )
        _df_wald_test_terms = logistic_model_res.wald_test_terms().summary_frame() \
            .rename(columns={"chi2": "wald_chi2", "P>chi2": "pvalue"}).loc[model_cols, ["wald_chi2", "pvalue"]] \
            .reset_index().rename(columns={"index": "col_name"})
        df_for_remove = _df_wald_test_terms[-_df_wald_test_terms["col_name"].isin(include_cols)] \
            .query("pvalue>{}".format(sls)).reset_index(drop=True)
        if df_for_remove.shape[0]==0:
            print("selection step done.")
            break
        # 与stepwise的剔除规则有出入，此处为一次只剔除一个pvalue最大的变量，再校验拟合模型进行剔除。
        df_for_remove = df_for_remove.query("pvalue=={}".format(df_for_remove["pvalue"].max())).reset_index(drop=True)
        df_for_stay = _df_wald_test_terms[-_df_wald_test_terms["col_name"].isin(df_for_remove["col_name"].tolist())].reset_index(drop=True)
        model_cols = df_for_stay["col_name"].tolist()

        if len(model_cols)==0:
            print("no variable enter model !!")
            break

        removed_cols = df_for_remove["col_name"].tolist()
        
        if verbose:
            print("==================================================================")
            print("current step {}:".format(curr_step))
            if len(removed_cols)>0:
                print("    variable(s): [ {} ] removed.".format(" / ".join(removed_cols)))
            else:
                print("    no variable removed.")
            print()

        for _col, wald_chi2, pvalue in df_for_remove.values:
            selection_step.append(OrderedDict({
                "step": curr_step,
                "entered_variable": "",
                "removed_variable": _col,
                "wald_chi2": wald_chi2,
                "pvalue": pvalue,
                "_model_variables": ",".join(model_cols),
            }))
            curr_step = curr_step+1
    
    logistic_model_res = func_logistic_model(
        df_X=df_X[model_cols], y=y,
        maxiter=100, with_intercept=with_intercept, disp=False,
    )
    if len(model_cols)>0:
        df_result_selection_step =  pd.DataFrame(selection_step) \
            [["step", "entered_variable", "removed_variable",
              "wald_chi2", "pvalue"]]
        return logistic_model_res, model_cols, df_result_selection_step
    else:
        return logistic_model_res, [], None


if __name__=="__main__":
    pass
    
    # ############################################################################
    # logistic_model_res, model_cols, df_result_selection_step = \
    #     func_logistic_model_stepwise(df_X=X_train, y=y_train.iloc[:, 1],
    #                                  initial_cols=None, include_cols=None, exclude_cols=None,
    #                                  sle=0.01, sls=0.05, with_intercept=True, verbose=1)
    
    # ############################################################################
    # logistic_model_res, model_cols, df_result_selection_step = \
    #     func_logistic_model_forward(df_X=X_train, y=y_train.iloc[:, 1],
    #                                 initial_cols=None, exclude_cols=None,
    #                                 sle=0.05, with_intercept=True, verbose=1)
    
    # ############################################################################
    # logistic_model_res, model_cols, df_result_selection_step = \
    #     func_logistic_model_backward(df_X=X_train, y=y_train.iloc[:, 1],
    #                                  initial_cols=None, include_cols=None,
    #                                  sls=0.05, with_intercept=True, verbose=1)
