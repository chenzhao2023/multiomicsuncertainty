""" Example for MOGONET classification
"""
import os
import numpy as np
import pandas as pd
import argparse
from collections import Counter
from train_test import train_test
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="KIPAN")
    parser.add_argument("--num_epoch", type=int, default=2500)
    parser.add_argument("--num_epoch_pretrain", type=int, default=500)
    parser.add_argument("--num_trials", type=int, default=2)
    parser.add_argument("--view_list", type=str, default="1,2")
    parser.add_argument("--lr_e_pretrain", type=float, default=1e-3)
    parser.add_argument("--lr_e", type=float, default=5e-4)
    parser.add_argument("--lr_c", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument("--num_adj", type=int, default=2)
    parser.add_argument("--exp", type=str, default="exp")

    args = parser.parse_args()

    # 
    data_folder = args.data_folder
    num_epoch = args.num_epoch
    num_epoch_pretrain = args.num_epoch_pretrain
    num_trials = args.num_trials
    view_list = args.view_list.split(",")
    lr_e_pretrain = args.lr_e_pretrain
    lr_e = args.lr_e
    lr_c = args.lr_c
    num_adj = args.num_adj

    exp_path = f"{args.exp}/{data_folder}_v{args.view_list}_t{num_trials}_a{num_adj}"

    save_path = f"{args.save_path}/{data_folder}/{','.join(map(str, view_list))}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if not os.path.isfile(f"{save_path}/n_trials{num_trials}_adj{num_adj}.csv"):
        result_df = open(f"{save_path}/n_trials{num_trials}_adj{num_adj}.csv", "w")
        trial_columns = [f'trial{i+1}' for i in range(num_trials)]
        columns = ['patient_id'] + trial_columns + ['std'] + ['label'] + ['pred']
        result_df.write(','.join(map(str, columns)))
        result_df.write("\n")

        probs = []
        preds = []
        for t in range(num_trials):
            if data_folder in ['ROSMAP', 'LGG']:
                num_class = 2
            if data_folder in ['BRCA', 'KIPAN']:
                num_class = 5
            
            # if not os.path.isdir(f"{exp_path}/trail_{t}"):
            #     os.makedirs(f"{exp_path}/trail_{t}")

            acc, best_1, best_2, prob, label, pred = train_test(f"{exp_path}/trail{t}", 
                                                                data_folder, view_list, num_class, lr_e_pretrain, lr_e, lr_c, 
                                                                num_epoch_pretrain, num_epoch, num_adj)
            preds.append(list(pred))
            probs.append(prob)

        probs = np.array(probs).T

        final_predictions = []
        for instance_preds in zip(*preds):
            # Count the occurrences of each prediction
            vote_count = Counter(instance_preds)
            # Select the prediction with the highest count (majority vote)
            final_predictions.append(vote_count.most_common(1)[0][0])
        acc = accuracy_score(label, final_predictions)

        for patient_id in range(probs.shape[0]):
            new_row = []
            new_row.append(patient_id)
            for n in range(num_trials):
                new_row.append(probs[patient_id][n])
            new_row.append(np.std(probs[patient_id]))
            new_row.append(label[patient_id])
            new_row.append(final_predictions[patient_id])
            result_df.write(','.join(map(str, new_row)))
            result_df.write("\n")
        
        result_df.flush()
        result_df.close()
            