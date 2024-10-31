""" Example for biomarker identification
"""
import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    data_folder = 'ROSMAP'
    exp_path = "exp"
    view_list = [1]
    num_trails = 2
    adj_parameter = 4
    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5

    for t in range(num_trails):
        featimp_list_list = []
        featimp_list = cal_feat_imp(data_folder, f"{exp_path}/{data_folder}_v1_t{num_trails}_a{adj_parameter}/trail{t}", view_list, num_class, adj_parameter)
        featimp_list_list.append(copy.deepcopy(featimp_list))
    
    feat_names, df_featimp_top = summarize_imp_feat(featimp_list_list, topn=30)
    
    print(feat_names)
    #print(featimp_list)

    