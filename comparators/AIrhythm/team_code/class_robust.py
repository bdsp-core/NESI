#!/usr/bin/env python

# @author: Morteza Zabihi (morteza.zabihi@gmail.com) 
# Copyright (C) 2023 Morteza Zabihi  
# =============================================================================
# By accessing the code through the Physionet webpage and/or by installing, 
# copying, or otherwise using this software, you acknowledge and agree to be 
# bound by the terms and conditions of the attached "LICENSE.md" file. If you 
# do not agree to these terms and conditions, do not install, copy or use the 
# software.
# 
# Please note that we make no representation or warranty regarding the 
# suitability or fitness for any particular purpose of this licensed deliverable.
# The software is provided "as is" without any express or implied warranty 
# of any kind.
# 
# Furthermore, any use of the licensed deliverables must include the above
# disclaimer. For more detailed information on the terms and conditions 
# governing the use of this software, please refer to the "readme" file and 
# "LICENSE.md" file located on our GitHub repository.
# 
# Please review the terms and conditions carefully before using the software.
# If you have any questions or concerns about the terms and conditions, 
# please contact Morteza Zabihi (morteza.zabihi@gmail.com)
# =============================================================================

import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, f1_score

from class_model import *
from evaluate_model import *


class class_robust:
    def __init__(self):
        self.seed = 0
        random.seed(42)
    # #########################################################################
    def feature_selecrtion_module(self, feature_importances):
        all_features = []
        for iter1 in range(len(feature_importances)):
            temp = feature_importances[iter1]
            if iter1 == 0:
                lf = len(temp)
            all_features.append(temp)
        
        all_features1 = np.array(all_features)
        all_features1 = all_features1.flatten()
        n = np.quantile(all_features1, 0.15)#int(0.90*lf) #lf

        indicess = []
        for iter1 in range(len(all_features)):
            arr = all_features[iter1]
            indices = np.where(arr > n)[0]
            # indices = np.argpartition(arr, -n)[-n:]
            if len(indices) >0:
                indicess.append(indices)
        
        common_values = indicess[0]  # Start with the first array
        for i in range(1, len(indicess)):
            common_values = np.intersect1d(common_values, indicess[i])
        return common_values
    # #########################################################################
    def calculate_mean_std(self, chscore_kfold, sensitivity_kfold, 
                           specificity_kfold, auroc_kfold, aurpoc_kfold,
                           fpr_kfold, tnr_kfold, f1_kfold, CM_kfold):
        
        chscore_mean = np.mean(chscore_kfold)
        chscore_std = np.std(chscore_kfold)
        
        sensitivity_mean = np.mean(sensitivity_kfold)
        sensitivity_std = np.std(sensitivity_kfold)

        specificity_mean = np.mean(specificity_kfold)
        specificity_std = np.std(specificity_kfold)

        auroc_mean = np.mean(auroc_kfold)
        auroc_std = np.std(auroc_kfold)

        aurpoc_mean = np.mean(aurpoc_kfold)
        aurpoc_std = np.std(aurpoc_kfold)

        fpr_mean = np.mean(fpr_kfold)
        fpr_std = np.std(fpr_kfold)

        tnr_mean = np.mean(tnr_kfold)
        tnr_std = np.std(tnr_kfold)
        
        f1_mean = np.mean(f1_kfold)
        f1_std = np.std(f1_kfold)
        
        
        for iter1 in range(len(CM_kfold)):
            if iter1 == 0:
                cmavg = CM_kfold[iter1]
            else:
                cmavg += CM_kfold[iter1]
        
        l = len(CM_kfold)
        l = float(l)
        cmavg1 = cmavg/l
        # print(f"Mean challenge score: {chscore_mean:.3f}")
        print(f"Mean challenge score: {chscore_mean:.3f} ± {chscore_std:.3f}")
        print(f"Mean Sensitivity: {sensitivity_mean:.2f} ± {sensitivity_std:.2f}")
        print(f"Mean Specificity: {specificity_mean:.2f} ± {specificity_std:.2f}")
        print(f"Mean AUROC: {auroc_mean:.2f} ± {auroc_std:.2f}")
        print(f"Mean AURPOC: {aurpoc_mean:.2f} ± {aurpoc_std:.2f}")
        print(f"Mean FPR: {fpr_mean:.2f} ± {fpr_std:.2f}")
        print(f"Mean TNR: {tnr_mean:.2f} ± {tnr_std:.2f}")
        print(f"Mean F1: {f1_mean:.2f} ± {f1_std:.2f}")
        print("CM: ", cmavg1)
    # #########################################################################  
    def calculate_metrics(self, labels, predictions, hosp):
        
        predictionsb = np.copy(predictions)
        predictionsb[predictionsb >= 0.5] = 1
        predictionsb[predictionsb < 0.5] = 0
        tn, fp, fn, tp = confusion_matrix(labels, predictionsb).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fpr = fp / (fp + tn)
        tnr = tn / (tn + fp)
        auroc = roc_auc_score(labels, predictions)
        precision, recall, _ = precision_recall_curve(labels, predictions)
        aurpoc = np.trapz(precision[::-1], recall[::-1])
        f1 = f1_score(labels, predictionsb, average="weighted")
        cm1 = np.array([[tn, fp], [fn, tp]])
        challenge_score = compute_challenge_score(labels, predictions, hosp)
        
        return challenge_score, sensitivity, auroc, aurpoc, specificity, fpr, tnr, f1, cm1
    # #########################################################################
    def cv(self, features, timestamps, labels, cpc_labels, pids, confounders, h, n_folds=5, clfop="xgb", ind_feature=None, prepop="mean", challenge=True):
        # --------------------------------------------------------------------- 
        hospitals = confounders[:, 1]
        # Define the cross-validation strategy
        cmobj = class_model()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
        models = []
        models_cpc = []
        medians = []
        
        chscore_kfold = []
        sensitivity_kfold = []
        specificity_kfold = []
        auroc_kfold = []
        aurpoc_kfold = []
        fpr_kfold = []
        tnr_kfold = []
        f1_kfold = []
        CM_kfold = []
        ppps = []
        pppsl = []
        feature_importances = []
        print("[INFO:] 5-fold cross-validation starts! ======================")
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(pids)):
            
            if set(train_idx).intersection(set(val_idx)):
                raise Exception("[Error:] The two lists have at least one common element. normal cv!")
            # -----------------------------------------------------------------
            features_train = [features[i] for i in train_idx]
            timestamps_train = [timestamps[i] for i in train_idx]            
            clf, cpc_clf, medians_nan, feature_importance = cmobj.train_model_wrapper(features_train, timestamps_train, labels[train_idx], cpc_labels[train_idx], pids[train_idx], clfop, h, ind_feature, prepop)
            # -----------------------------------------------------------------
            if not challenge:
                features_test = [features[i] for i in val_idx]
                timestamps_test = [timestamps[i] for i in val_idx]      
                predictions, cpc_predictions, ppp = cmobj.inference(pids[val_idx], features_test, timestamps_test, clf, cpc_clf, medians_nan, h, ind_feature, prepop)
                # predictions, cpc_predictions, ppp = cmobj.lstm_inference(pids[val_idx], features_test, timestamps_test, clf, cpc_clf, medians_nan, h)

                ppps.append(ppp)
                # -------------------------------------------------------------
                # evaluation
                predictions = predictions[:, 1]
                chscore, sensitivity, auroc, aurpoc, specificity, fpr, tnr, f1, cm =\
                    self.calculate_metrics(labels[val_idx], predictions, hospitals[val_idx])
                pppsl.append(labels[val_idx])
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                chscore_kfold.append(chscore)
                sensitivity_kfold.append(sensitivity)
                specificity_kfold.append(specificity)
                auroc_kfold.append(auroc)
                aurpoc_kfold.append(aurpoc)
                fpr_kfold.append(fpr)
                tnr_kfold.append(tnr)
                f1_kfold.append(f1)
                CM_kfold.append(cm)
                print(chscore_kfold)
            # -----------------------------------------------------------------
            models.append(clf)
            models_cpc.append(cpc_clf)
            medians.append(medians_nan)
            feature_importances.append(feature_importance)
            # -----------------------------------------------------------------
        if not challenge:
            self.calculate_mean_std(chscore_kfold, sensitivity_kfold, specificity_kfold,
                               auroc_kfold, aurpoc_kfold, fpr_kfold, tnr_kfold, f1_kfold, CM_kfold)
        # ---------------------------------------------------------------------
        if clfop == "stacking" or clfop == "stacking1" or clfop == "stacking2" or clfop == "lda" or clfop == "svm" or clfop == "mlp" or clfop == "lgb":
            common_features = []
            
        else:
            common_features = self.feature_selecrtion_module(feature_importances)
        # ---------------------------------------------------------------------   
        return models, models_cpc, medians, common_features, ppps, pppsl
    # #########################################################################
    def cv_cluster(self, features, timestamps, labels, cpc_labels, pids, confounders, h, optimal_k=3, modecluster="kmeans", clfop="xgb", ind_feature=None, prepop="mean", challenge=True):
        hospitals = confounders[:, 1]
        # ---------------------------------------------------------------------
        cmobj = class_model()        
        # ---------------------------------------------------------------------
        if modecluster == "kmeans":
            confounders_patient_wise, _ = cmobj.replace_nan_with_median(confounders, mode="train")
            # -----------------------------------------------------------------
            min_vals = np.min(confounders_patient_wise, axis=0)
            max_vals = np.max(confounders_patient_wise, axis=0)
            max_vals[max_vals == min_vals] = max_vals[max_vals == min_vals] + 1
            normalized_matrix = (confounders_patient_wise - min_vals) / (max_vals - min_vals)
            # -----------------------------------------------------------------
            kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(normalized_matrix)
        # ---------------------------------------------------------------------
        # elif modecluster == "kmedoids":
        #     kmeans = KModes(n_clusters=optimal_k, init='Huang', n_init=5, verbose=1)
        #     clusters = kmeans.fit_predict(features)
        # ---------------------------------------------------------------------
        cluster_labels = kmeans.labels_
        # ---------------------------------------------------------------------
        split_indices = []
        for i in range(optimal_k):
            split_indices.append(np.where(cluster_labels == i)[0])
        # --------------------------------------------------------------------- 
        models = []
        models_cpc = []
        medians = []
        
        chscore_kfold = []
        sensitivity_kfold = []
        specificity_kfold = []
        auroc_kfold = []
        aurpoc_kfold = []
        fpr_kfold = []
        tnr_kfold = []
        f1_kfold = []
        CM_kfold = []
        ppps = []
        pppsl = []
        feature_importances = []
        print("[INFO:] Cluster cross-validation starts! =====================")
        for i in range(optimal_k):
            train_indices = split_indices[:i] + split_indices[i+1:]
            all_train_indices = []
            for indices in train_indices:
                all_train_indices.extend(indices)
            del train_indices
            train_indices = all_train_indices
            del all_train_indices, indices
        
            test_indices = split_indices[i]
            
            if set(test_indices).intersection(set(train_indices)):
                raise Exception("[Error:] The two lists have at least one common element. cluster cv!")
            # -----------------------------------------------------------------
            features_train = [features[i] for i in train_indices]
            timestamps_train = [timestamps[i] for i in train_indices] 
            clf, cpc_clf, medians_nan, feature_importance = cmobj.train_model_wrapper(features_train, timestamps_train, labels[train_indices], cpc_labels[train_indices], pids[train_indices], clfop, h, ind_feature, prepop)
            # -----------------------------------------------------------------
            if not challenge:
                features_test = [features[i] for i in test_indices]
                timestamps_test = [timestamps[i] for i in test_indices]
                predictions, cpc_predictions, ppp = cmobj.inference(pids[test_indices], features_test, timestamps_test, clf, cpc_clf, medians_nan, h, ind_feature, prepop)
                # predictions, cpc_predictions, ppp = cmobj.lstm_inference(pids[test_indices], features_test, timestamps_test, clf, cpc_clf, medians_nan, h)

                ppps.append(ppp)
                # -------------------------------------------------------------
                # evaluation
                predictions = predictions[:, 1]
                chscore, sensitivity, auroc, aurpoc, specificity, fpr, tnr, f1, cm =\
                    self.calculate_metrics(labels[test_indices], predictions, hospitals[test_indices])
                pppsl.append(labels[test_indices])
                # -------------------------------------------------------------
                chscore_kfold.append(chscore)
                sensitivity_kfold.append(sensitivity)
                specificity_kfold.append(specificity)
                auroc_kfold.append(auroc)
                aurpoc_kfold.append(aurpoc)
                fpr_kfold.append(fpr)
                tnr_kfold.append(tnr)
                f1_kfold.append(f1)
                CM_kfold.append(cm)
                print(chscore_kfold)
            # -----------------------------------------------------------------                
            models.append(clf)
            models_cpc.append(cpc_clf)
            medians.append(medians_nan)
            feature_importances.append(feature_importance)
        # ---------------------------------------------------------------------
        if not challenge:
            self.calculate_mean_std(chscore_kfold, sensitivity_kfold, specificity_kfold,
                               auroc_kfold, aurpoc_kfold, fpr_kfold, tnr_kfold, f1_kfold, CM_kfold)
        # ---------------------------------------------------------------------
        if clfop == "stacking" or clfop == "stacking1" or clfop == "stacking2" or clfop == "lda" or clfop == "svm" or clfop == "mlp"  or clfop == "lgb":
            common_features = []
            
        else:
            common_features = self.feature_selecrtion_module(feature_importances)
        # ---------------------------------------------------------------------
        return models, models_cpc, medians, common_features, ppps, pppsl
    # #########################################################################
    def cv_ps(self, features, timestamps, labels, cpc_labels, pids, confounders, h, clfop="xgb",  ind_feature=None, prepop="mean", challenge=True):
        hospitals = confounders[:, 1]
        # ---------------------------------------------------------------------
        cmobj = class_model()
        # ---------------------------------------------------------------------
        confounders_patient_wise, _ = cmobj.replace_nan_with_median(confounders, mode="train")        
        # ---------------------------------------------------------------------
        ps_groups = self.ps_grouping(confounders_patient_wise, labels)
        # ---------------------------------------------------------------------
        models = []
        models_cpc = []
        medians = []
        
        chscore_kfold = []
        sensitivity_kfold = []
        specificity_kfold = []
        auroc_kfold = []
        aurpoc_kfold = []
        fpr_kfold = []
        tnr_kfold = []
        f1_kfold = []
        CM_kfold = []
        feature_importances = []
        ppps = []
        pppsl = []
        print("[INFO:] PS cross-validation starts! ==========================")
        for iter1 in range(len(np.unique(ps_groups))):
            
            test_indices = np.where(ps_groups == iter1)[0]
            # -----------------------------------------------------------------
            train_indices = []
            for iter2 in range(len(np.unique(ps_groups))):
                if iter2 != iter1:
                    train_indices.extend(np.where(ps_groups == iter2)[0])
            del iter2
            # -----------------------------------------------------------------
            if set(test_indices).intersection(set(train_indices)):
                raise Exception("[Error:] The two lists have at least one common element. ps cv!")
            # -----------------------------------------------------------------
            features_train = [features[i] for i in train_indices]
            timestamps_train = [timestamps[i] for i in train_indices] 
            
            clf, cpc_clf, medians_nan, feature_importance = cmobj.train_model_wrapper(features_train, timestamps_train, labels[train_indices], cpc_labels[train_indices], pids[train_indices], clfop, h, ind_feature, prepop)
            # -----------------------------------------------------------------
            if not challenge:
                features_test = [features[i] for i in test_indices]
                timestamps_test = [timestamps[i] for i in test_indices] 
                
                predictions, cpc_predictions, ppp = cmobj.inference(pids[test_indices], features_test, timestamps_test, clf, cpc_clf, medians_nan, h, ind_feature, prepop)
                # predictions, cpc_predictions, ppp = cmobj.lstm_inference(pids[test_indices], features_test, timestamps_test, clf, cpc_clf, medians_nan, h)

                ppps.append(ppp)
                # -------------------------------------------------------------
                # evaluation
                predictions = predictions[:, 1]
                chscore, sensitivity, auroc, aurpoc, specificity, fpr, tnr, f1, cm =\
                    self.calculate_metrics(labels[test_indices], predictions, hospitals[test_indices])
                pppsl.append(labels[test_indices])
                # -------------------------------------------------------------
                chscore_kfold.append(chscore)
                sensitivity_kfold.append(sensitivity)
                specificity_kfold.append(specificity)
                auroc_kfold.append(auroc)
                aurpoc_kfold.append(aurpoc)
                fpr_kfold.append(fpr)
                tnr_kfold.append(tnr)
                f1_kfold.append(f1)
                CM_kfold.append(cm)
                print(chscore_kfold)
            # -----------------------------------------------------------------
            models.append(clf)
            models_cpc.append(cpc_clf)
            medians.append(medians_nan)
            feature_importances.append(feature_importance)
        # ---------------------------------------------------------------------
        if not challenge:
            self.calculate_mean_std(chscore_kfold, sensitivity_kfold, specificity_kfold,
                               auroc_kfold, aurpoc_kfold, fpr_kfold, tnr_kfold, f1_kfold, CM_kfold)
        # ---------------------------------------------------------------------
        if clfop == "stacking" or clfop == "stacking1" or clfop == "stacking2" or clfop == "lda" or clfop == "svm" or clfop == "mlp"  or clfop == "lgb":
            common_features = []
            
        else:
            common_features = self.feature_selecrtion_module(feature_importances)
        # ---------------------------------------------------------------------   
        return models, models_cpc, medians, common_features, ppps, pppsl
    # #########################################################################
    def ps_grouping(self, confounder_matrix, labelsx, n_folds=5):
        
        cm = class_model()
        # ---------------------------------------------------------------------
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        indicesps = []
        ps = []
        # ---------------------------------------------------------------------
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(confounder_matrix)):
            
            if set(train_idx).intersection(set(val_idx)):
                raise Exception("[Error:] The two lists have at least one common element. ps grouping!")
            
            # Split the data into training and validation sets ----------------
            X_train = confounder_matrix[train_idx, :]
            y_train = labelsx[train_idx]
            X_val = confounder_matrix[val_idx, :]
            
            # Find the minimum and maximum values for each column -------------
            min_vals = np.min(X_train, axis=0)
            max_vals = np.max(X_train, axis=0)
        
            # Avoid dividing by zero when a column has all the same values ----
            max_vals[max_vals == min_vals] = max_vals[max_vals == min_vals] + 1
        
            # Normalize each column between 0 and 1
            norm_matrix = (X_train - min_vals) / (max_vals - min_vals)
            balanced_featuresz, balanced_labelsz, _ = cm.balance_classes(norm_matrix, y_train)
            clf = LogisticRegression().fit(balanced_featuresz, balanced_labelsz)
            
            # predict labels
            norm_matrix = (X_val - min_vals) / (max_vals - min_vals)
            pred_labels = clf.predict_proba(norm_matrix)
            
            indicesps.extend(val_idx)
            ps.extend(pred_labels[:, 1])
            
        
        # zip the two lists together
        zipped = zip(indicesps, ps)
        # sort the zipped list based on the values in list1
        sorted_pairs = sorted(zipped)
        # unzip the sorted list back into separate lists
        indicesps, ps = zip(*sorted_pairs)
        # ---------------------------------------------------------------------
        n = confounder_matrix.shape[0]
        groups = np.zeros(n, dtype=int)
        thresholds = np.percentile(ps, [25, 50, 75])
        for i in range(n):
            score = ps[i]
            if score <= thresholds[0]:
                groups[i] = 0
            elif score > thresholds[0] and score <= thresholds[1]:
                groups[i] = 1
            elif score > thresholds[1] and score < thresholds[2]:
                groups[i] = 2
            else:
                groups[i] = 3
        # ---------------------------------------------------------------------
        return groups
        