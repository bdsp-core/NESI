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

import numpy as np, os
import joblib
import random
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import skew


class class_model:
    def __init__(self):
        self.seed = 0
        random.seed(42)
    # #########################################################################
    def regression_features_fit(self, data, labels_cpc, labels):
        num_folds = 5
        predictions = []

        # Initialize the KFold cross-validator
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        k1 = 0
        regmodels = []
        # Loop through the folds
        for train_idx, test_idx in kf.split(data):
            train_data, test_data = data[train_idx, :], data[test_idx, :]
            train_labels, test_labels = labels_cpc[train_idx], labels_cpc[test_idx]
            labels_fold = labels[test_idx]
            if len(labels_fold.shape) == 1:
                labels_fold = np.expand_dims(labels_fold, axis=1)
            # # Initialize and train the CatBoost regressor
            # model = CatBoostRegressor(iterations=1500, 
            #                           learning_rate=0.003, 
            #                           loss_function='RMSE',
            #                           logging_level='Silent',
            #                           max_depth=6,
            #                           od_type="Iter",
            #                           od_wait=160,
            #                           allow_writing_files=False,
            #                           )
            
            # model.fit(train_data, train_labels, verbose=100)
            model = XGBRegressor()
            model.fit(train_data, train_labels, verbose=False)
            
            regmodels.append(model)
            # Make predictions on the test set
            fold_predictions = model.predict(test_data)

            # Store the fold predictions
            predictions.append(fold_predictions)

            # Append the test data and predictions to the new_data array
            fold_new_data = np.hstack((test_data, fold_predictions.reshape(-1, 1)))
            
            if k1 == 0:
                new_labels = labels_fold
                new_data = fold_new_data
            else:
                new_labels = np.vstack((new_labels, labels_fold))
                new_data = np.vstack((new_data, fold_new_data))
            
            k1 += 1

        return regmodels, new_data, new_labels
    # #########################################################################
    def regression_features_pred(self, data, regmodels):
        
        fold_predictions = np.zeros((data.shape[0], len(regmodels)))
        for iter1 in range(len(regmodels)):
            model = regmodels[iter1]
            fold_predictions[:, iter1] = model.predict(data)
        
        fold_predictions = np.nanmean(fold_predictions, axis=1, keepdims=True)
        new_data = np.hstack((data, fold_predictions))

        return new_data
    # #########################################################################
    def nan_mean_harmonic(self, numbers):
        valid_numbers = np.array(numbers)[~np.isnan(numbers)]
        if len(valid_numbers) == 0:
            return np.nan
    
        return len(valid_numbers) / np.sum(1 / valid_numbers)
    # #########################################################################
    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''
    
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
        
        
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        return position_enc
    # #########################################################################
    def save_challenge_model(self, model_folder, outcome_model, cpc_model, medians, configs):
        d = {'outcome_model': outcome_model, 'cpc_model': cpc_model, 'medians': medians, "configs": configs}
        filename = os.path.join(model_folder, 'models.sav')
        joblib.dump(d, filename, protocol=0)
    # #########################################################################    
    def load_challenge_models(self, model_folder):
        filename = os.path.join(model_folder, 'models.sav')
        return joblib.load(filename)
    # #########################################################################
    def replace_nan_with_median(self, array, mode="train", medians_nan=[]): #&
        if mode == "train":
            medians_nan = np.zeros((array.shape[1], 1))
            for i in range(array.shape[1]):
                column = array[:, i]
                nan_indices = np.isnan(column)
                inf_indices = np.isinf(column)
                if np.any(inf_indices):
                    inf_indices = np.isinf(column)
                    column_without_inf = column[~inf_indices]
                    if len(column_without_inf)!=0:
                        median = np.nanmedian(column_without_inf)
                    else:
                        median = 0
                else:
                    median = np.nanmedian(column)
                    
                medians_nan[i, 0] = median
                array[nan_indices, i] = median
                array[inf_indices, i] = median
                
            return array, medians_nan
        
        if mode == "test":
            for i in range(array.shape[1]):
                column = array[:, i]
                nan_indices = np.isnan(column)
                inf_indices = np.isinf(column)
                if np.any(nan_indices):
                    array[nan_indices, i] = medians_nan[i, 0]
                
                if np.any(inf_indices):
                    array[inf_indices, i] = medians_nan[i, 0]
            return array
    # #########################################################################    
    def create_temporal_datasets_daily(self, features, timestamps, labels, cpc_labels,
                                 h=6, prepop="mean", window_size=3, day=1): #& 6 12 72
        
        """
        features :
        labels : 
        cpc_labels :
        h : TYPE, lookback period, in hours. The default is 6.
        window_size : size of rolling window for feature extraction in minutes.
                      The default is 3.
        """
        # ---------------------------------------------------------------------
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        # ---------------------------------------------------------------------  
        if features.shape[0]>1:
            
            timestamps = (timestamps * window_size) / 60 # in hours
            timestamps = timestamps - timestamps[0] #****zero the start
            
            temps = []
            temps1 = []
            temps2 = []
            
            start = 0
            
            end = h
            end1 = 12
            end2 = 18
            
            starts = []
            while (True):
                inds = np.where((timestamps>=start) & (timestamps<end))[0]
                inds1 = np.where((timestamps>=start) & (timestamps<end1))[0]
                inds2 = np.where((timestamps>=start) & (timestamps<end2))[0]
                # -------------------------------------------------------------
                if len(inds) > 0:
                    temp0 = features[inds, :]
                    temp1 = features[inds1, :]
                    temp2 = features[inds2, :]
                    # ---------------------------------------------------------
                    if len(temp0.shape) == 1:
                        temp0 = np.expand_dims(temp0, axis=0)
                    
                    if len(temp1.shape) == 1:
                        temp1 = np.expand_dims(temp1, axis=0)
                        
                    if len(temp2.shape) == 1:
                        temp2 = np.expand_dims(temp2, axis=0)
                    # ---------------------------------------------------------   
                    temps.append(self.prepare_data(temp0, prepop)) #&
                    temps1.append(self.prepare_data(temp1, prepop)) #&
                    temps2.append(self.prepare_data(temp2, prepop)) #&
                    
                    starts.append(start)
                    # ---------------------------------------------------------
                    if inds[-1] == len(timestamps) - 1:
                        break
                    else:
                        pass
                    # ---------------------------------------------------------
                # -------------------------------------------------------------
                else:
                    pass
                # -------------------------------------------------------------
                start = end
                end += h
                end1 += 12
                end2 += 18
                
                if end > 72:
                    break
            # -----------------------------------------------------------------
            if len(temps) > 0:
                
                features_temporal = np.array(temps)
                if len(features_temporal.shape) == 3:
                    features_temporal = np.squeeze(features_temporal)
                    
                if len(features_temporal.shape) == 1:
                    features_temporal = np.expand_dims(features_temporal, axis=0)
                    
                features_temporal1 = np.array(temps1)
                if len(features_temporal1.shape) == 3:
                    features_temporal1 = np.squeeze(features_temporal1)
                    
                if len(features_temporal1.shape) == 1:
                    features_temporal1 = np.expand_dims(features_temporal1, axis=0)
                    
                features_temporal2 = np.array(temps2)
                if len(features_temporal2.shape) == 3:
                    features_temporal2 = np.squeeze(features_temporal2)
                    
                if len(features_temporal2.shape) == 1:
                    features_temporal2 = np.expand_dims(features_temporal2, axis=0)
                
                labels_temporal = labels*np.ones((features_temporal.shape[0],))
                cpc_labels_temporal = cpc_labels*np.ones((features_temporal.shape[0],))
                # -------------------------------------------------------------
                starts = np.squeeze(np.array(starts))
                # -------------------------------------------------------------
                return features_temporal, features_temporal1, features_temporal2, labels_temporal, cpc_labels_temporal, starts  
            else:
                return [], [], [], [], [], []
        else:
            return self.prepare_data(features, prepop), self.prepare_data(features, prepop), self.prepare_data(features, prepop), labels, cpc_labels, []
    # #########################################################################
    def create_temporal_datasets(self, features, timestamps, labels, cpc_labels,
                                 h=6, prepop="mean", window_size=3): #& 6 12 72
        
        """
        features :
        labels : 
        cpc_labels :
        h : TYPE, lookback period, in hours. The default is 6.
        window_size : size of rolling window for feature extraction in minutes.
                      The default is 3.
        """
        # ---------------------------------------------------------------------
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        # ---------------------------------------------------------------------
        if features.shape[0]>1:
            
            timestamps = (timestamps * window_size) / 60 # in hours
            timestamps = timestamps - timestamps[0] #****zero the start
            temps = []
            start = 0
            end = h
            
            while (True):
                inds = np.where((timestamps>=start) & (timestamps<end))[0]
                # -------------------------------------------------------------
                if len(inds) > 0:
                    temp1 = features[inds, :]
                    # ---------------------------------------------------------
                    if len(temp1.shape) == 1:
                        temp1 = np.expand_dims(temp1, axis=0)
                        
                    temps.append(self.prepare_data(temp1, prepop)) #&
                    # ---------------------------------------------------------
                    if inds[-1] == len(timestamps) - 1:
                        break
                    else:
                        pass
                    # ---------------------------------------------------------
                # -------------------------------------------------------------
                else:
                    pass
                # -------------------------------------------------------------
                start = end
                end += h
                
                if end > 72:
                    break
            # ----------------------------------------------------------------- 
            features_temporal = np.array(temps)
            if len(features_temporal.shape) == 3:
                features_temporal = np.squeeze(features_temporal)
                
            if len(features_temporal.shape) == 1:
                features_temporal = np.expand_dims(features_temporal, axis=0)
            
            labels_temporal = labels*np.ones((features_temporal.shape[0],))
            cpc_labels_temporal = cpc_labels*np.ones((features_temporal.shape[0],))
            # -----------------------------------------------------------------
            # -----------------------------------------------------------------
            return features_temporal, labels_temporal, cpc_labels_temporal
        # ---------------------------------------------------------------------
        else:
            return self.prepare_data(features, prepop), labels, cpc_labels     
    # #########################################################################
    def prepare_data(self, features, mode="mean"): #&
        
        if mode == "mean":
            temp = np.nanmean(features, axis=0, keepdims=True)
        if mode == "q88":
            temp = np.quantile(features, 0.88, axis=0, keepdims=True)
        if mode == "q89":
            temp = np.quantile(features, 0.89, axis=0, keepdims=True)
        if mode == "q91":
            temp = np.quantile(features, 0.91, axis=0, keepdims=True)
        if mode == "diff_mean":
            temp1 = np.nanmean(features, axis=0, keepdims=True)
            temp2 = np.nanstd(features, axis=0, keepdims=True) + np.finfo(np.float32).eps
            temp = np.divide(temp1, temp2)
        if mode == "combine":
            if len(features.shape) == 1:
                features = np.expand_dims(features, axis=0)
            # -----------------------------------------------------------------
            if features.shape[0]>1:
                temp1 = skew(features, axis=0, nan_policy="omit", keepdims=True)
            else:
                temp1 = np.zeros((1, features.shape[1]))
                
            temp23 = np.quantile(features, 0.89, axis=0, keepdims=True)
            if len(temp23.shape) == 1:
                temp23 = np.expand_dims(temp23, axis=0)
            
            temp = np.hstack((temp1, temp23))
            
        if mode == "combine1":
            if len(features.shape) == 1:
                features = np.expand_dims(features, axis=0)
            # -----------------------------------------------------------------
            if features.shape[0]>1:
                temp1 = skew(features, axis=0, nan_policy="omit", keepdims=True)
                temp1 = np.nan_to_num(temp1)
            else:
                temp1 = np.zeros((1, features.shape[1]))
                
            temp23 = np.quantile(features, 0.89, axis=0, keepdims=True)
            if len(temp23.shape) == 1:
                temp23 = np.expand_dims(temp23, axis=0)
            
            temp = np.hstack((temp1, temp23))   
            

        
        if len(temp.shape) == 1:
            temp = np.expand_dims(temp, axis=0)
        return temp
    # #########################################################################
    def balance_classes(self, featuresz, labelsz):
        
        classes, class_counts = np.unique(labelsz, return_counts=True)
        minority_class = classes[np.argmin(class_counts)]
        majority_class = classes[np.argmax(class_counts)]
        # ---------------------------------------------------------------------
        majority_indices = np.where(labelsz == majority_class)[0]
        minority_indices = np.where(labelsz == minority_class)[0]
        # ---------------------------------------------------------------------
        majority_sample_indices = np.random.choice(majority_indices,
                                                   size=len(minority_indices),
                                                   replace=False)
    
        indices = np.concatenate((minority_indices, majority_sample_indices))
    
        np.random.shuffle(indices)
        # # ---------------------------------------------------------------------
        # minority_sample_indices = np.random.choice(minority_indices,
        #                                            size=len(majority_indices),
        #                                            replace=True)
    
        # indices1 = np.concatenate((minority_sample_indices, majority_indices))
    
        # np.random.shuffle(indices1)
        # # ---------------------------------------------------------------------
        # indices = np.hstack((indices, indices1))
        # ---------------------------------------------------------------------
        balanced_features = featuresz[indices, :]
        if labelsz.ndim == 2:
            balanced_labels = labelsz[indices, :]
        else:
            balanced_labels = labelsz[indices]
        return balanced_features, balanced_labels, indices
    # ######################################################################### 
    def flip_labels(self, binary_array, flip_percentage=0.05):
        # Calculate the number of labels to flip
        num_labels = binary_array.size
        num_flips = int(num_labels * flip_percentage)

        # Generate random indices for flipping
        flip_indices = np.random.choice(num_labels, num_flips, replace=False)

        # Flip the selected labels
        flipped_array = binary_array.copy()
        flipped_array[flip_indices] = 1 - flipped_array[flip_indices]

        return flipped_array
    # #########################################################################
    def stacking_ensemble(self, train_data0, train_data1, train_data2, train_labels, 
                          learners=[], mode="train"):
        
        if mode == "train":
            def perform_cross_validation(data, labels, n_splits=5, random_state=None):
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                fold_indices = []
                for train_idx, test_idx in kf.split(data):
                    # train_data, test_data = data[train_idx], data[test_idx]
                    # train_labels, test_labels = labels[train_idx], labels[test_idx]
                    fold_indices.append((train_idx, test_idx))
                return fold_indices

            fold_indices = perform_cross_validation(train_data0, train_data1)

            learners = []
            for i in range(len(fold_indices)):
                
                train_idx, test_idx = fold_indices[i]
                # -------------------------------------------------------------------------
                base_1 = CatBoostClassifier(iterations=1500, learning_rate=0.003,
                                          verbose=False, max_depth=6, od_type="Iter",
                                          od_wait=160, allow_writing_files=False)
                
                base_2 = CatBoostClassifier(iterations=1500, learning_rate=0.003,
                                          verbose=False, max_depth=6, od_type="Iter",
                                          od_wait=160, allow_writing_files=False)
                
                base_3 = CatBoostClassifier(iterations=1500, learning_rate=0.003,
                                          verbose=False, max_depth=6, od_type="Iter",
                                          od_wait=160, allow_writing_files=False)
                
                base_1.fit(train_data0[train_idx, :], train_labels[train_idx])
                base_2.fit(train_data1[train_idx, :], train_labels[train_idx])
                base_3.fit(train_data2[train_idx, :], train_labels[train_idx])
                
                base_predictions = np.zeros((len(test_idx), 6))
                base_predictions[:, 0:2] = base_1.predict_proba(train_data0[test_idx, :])
                base_predictions[:, 2:4] = base_2.predict_proba(train_data1[test_idx, :])
                base_predictions[:, 4:6] = base_3.predict_proba(train_data2[test_idx, :])
                
                meta_model = LogisticRegression()
                meta_model.fit(base_predictions, train_labels[test_idx])
                
                # =============================================================
                model = {'base1': base_1, "base2": base_2, "base3": base_3, 
                         "meta": meta_model}
                learners.append(model)
                # -------------------------------------------------------------------------

            return learners
        
        elif mode == "test":
            all_preds = []
            
            for iter1 in range(len(learners)):
                
                learner = learners[iter1]
                base_1 = learner["base1"]
                base_2 = learner["base2"]
                base_3 = learner["base3"]
                
                meta_model = learner["meta"]
                
                # Train base models
                base_predictions = np.zeros((train_data0.shape[0], 6))
                
                base_predictions[:, 0:2] = base_1.predict_proba(train_data0)
                base_predictions[:, 2:4] = base_2.predict_proba(train_data1)
                base_predictions[:, 4:6] = base_3.predict_proba(train_data2)
                
                predictions = meta_model.predict_proba(base_predictions)
                if iter1 == 0:
                    all_preds = predictions
                else:
                    all_preds += predictions
                
            all_preds = all_preds / len(learners) 
            return all_preds
    # #########################################################################
    def train_model(self, X_train, labels_train, cpc_labels_train, clfop="xgb", ind_features=None):
        
        if clfop=="stacking2":
            temp_features0 = X_train["temp_features0"]
            temp_features1 = X_train["temp_features1"]
            temp_features2 = X_train["temp_features2"]
            
            temp_features0, balanced_labels, indbal = self.balance_classes(temp_features0, labels_train)
            temp_features1 = temp_features1[indbal, :]
            temp_features2 = temp_features2[indbal, :]
            
            
            clf = self.stacking_ensemble(temp_features0, temp_features1, temp_features2, balanced_labels, 
                                  learners=[], mode="train")
            
            feature_importance = []
            
        if clfop !="stacking2":
            balanced_features, balanced_labels, indbal = self.balance_classes(X_train, labels_train)
            # balanced_labels = self.flip_labels(balanced_labels, flip_percentage=0.05)#**
        
            if ind_features is not None:
                balanced_features = balanced_features[:, ind_features]
        # ---------------------------------------------------------------------
        # regmodels, balanced_features, balanced_labels = self.regression_features_fit(balanced_features, cpc_labels_train[indbal], balanced_labels)
        # ---------------------------------------------------------------------
        if clfop == "xgb":
            clf = XGBClassifier(n_estimators=700, eval_metric='mlogloss')
            clf.fit(balanced_features, balanced_labels, verbose=False)
            feature_importance = clf.feature_importances_
            
            
        elif clfop == "cat":
            clf = CatBoostClassifier(iterations=1500, learning_rate=0.003,
                                      verbose=False,
                                      max_depth=6,
                                      od_type="Iter",
                                      od_wait=160,
                                      allow_writing_files=False)#1500
            clf.fit(balanced_features, balanced_labels)
            feature_importance = clf.get_feature_importance()
            
            # clf1 = CatBoostClassifier(iterations=1500, learning_rate=0.003,
            #                           verbose=False,
            #                           max_depth=6,
            #                           od_type="Iter",
            #                           od_wait=160,
            #                           allow_writing_files=False)#1500
            # clf1.fit(balanced_features, balanced_labels)
            # feature_importance = clf1.get_feature_importance()
            # clf = {'clf1': clf1, 'regmodels': regmodels}
                    
        elif clfop == "lgb":
            clf = lgb.LGBMClassifier(num_iterations=150, verbose=-100)
            clf.fit(balanced_features, balanced_labels)
            feature_importance = []
            
        elif clfop =="lda":
            scaler = StandardScaler()
            balanced_features = scaler.fit_transform(balanced_features)
            clf1 = LinearDiscriminantAnalysis()
            clf1.fit(balanced_features, balanced_labels)
            clf = {"clf": clf1, "scaler": scaler}
            feature_importance = []
        
        elif clfop == "svm":
            scaler = StandardScaler()
            balanced_features = scaler.fit_transform(balanced_features)
            clf1 = SVC(probability=True)            
            clf1.fit(balanced_features, balanced_labels)
            clf = {"clf": clf1, "scaler": scaler}
            feature_importance = []
        
        elif clfop == "mlp":
            scaler = StandardScaler()
            balanced_features = scaler.fit_transform(balanced_features)
            clf1 = MLPClassifier(hidden_layer_sizes=(70, 70), random_state=1, max_iter=300).fit(balanced_features, balanced_labels)
            clf = {"clf": clf1, "scaler": scaler}
            feature_importance = []
            
            
        elif clfop == "stacking":
            estimators = [
                ('mlp', make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(70, 70), random_state=1, max_iter=300, verbose=False))),
                ('svm', make_pipeline(StandardScaler(), SVC(probability=True))),
                ('lda', make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())),
                ('cat', CatBoostClassifier(iterations=1500, learning_rate=0.003,
                                          verbose=False,
                                          max_depth=6,
                                          od_type="Iter",
                                          od_wait=160,
                                          allow_writing_files=False)),
                ]
            clf = StackingClassifier(estimators=estimators,
                                     final_estimator=MLPClassifier(hidden_layer_sizes=(30), random_state=1, max_iter=300, verbose=False),
                                     cv=5,
                                     stack_method="predict_proba",
                                     passthrough=True,
                                     n_jobs=-1)
            clf.fit(balanced_features, balanced_labels)
            feature_importance = []
            
        elif clfop == "stacking1":
            # Base level estimators
            base_estimators_level1 = [
                ('mlp1', make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=300, verbose=False))),
                ('svm1', make_pipeline(StandardScaler(), SVC(probability=True))),
                ('cat1', CatBoostClassifier(iterations=1500, learning_rate=0.003, verbose=False, max_depth=6, od_type="Iter", od_wait=160, allow_writing_files=False)),
                ('extra1', ExtraTreesClassifier(n_estimators=500, max_features=17, min_samples_split=4)),
                ('lda', make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())),
                ]

            clf = StackingClassifier(estimators=base_estimators_level1,
                                     final_estimator=MLPClassifier(hidden_layer_sizes=(5, 5),
                                                                   max_iter=300,
                                                                   verbose=False),
                                     cv=5,
                                     stack_method="predict_proba",
                                     passthrough=False,
                                     n_jobs=-1)
            clf.fit(balanced_features, balanced_labels)
            feature_importance = []
        # ---------------------------------------------------------------------
        if clfop == "stacking2":
            X_train = X_train["temp_features0"]
            
        if ind_features is not None:
            X_train = X_train[:, ind_features]
        cpc_clf = XGBRegressor()
        cpc_clf.fit(X_train, cpc_labels_train, verbose=False)
        # ---------------------------------------------------------------------
        
        return clf, cpc_clf, feature_importance
    # #########################################################################
    def train_model_wrapper(self, features, timestamps, labels, cpc_labels, pids, clfop, h, ind_feature=None, prepop="mean"):
        
        
        _, medians_nan = self.replace_nan_with_median(np.vstack(features), mode="train") #&
        
        features_list = []
        for iter1 in range(len(pids)):
            
            temp = self.replace_nan_with_median(features[iter1], "test", medians_nan) #&
            
            if clfop == "stacking2":
                temp_feature, temp_feature1, temp_feature2, temp_label, temp_cpc_label, _= \
                    self.create_temporal_datasets_daily(temp, timestamps[iter1], 
                                          labels[iter1], cpc_labels[iter1], h, prepop)
                if len(temp_feature)>0:
                    # ---------------------------------------------------------
                    pos = self.position_encoding_init(temp_feature.shape[0], temp_feature.shape[1])#&
                    temp_feature = temp_feature + pos
                    temp_feature1 = temp_feature1 + pos
                    temp_feature2 = temp_feature2 + pos
                    # ---------------------------------------------------------
                    if iter1 == 0:
                        temp_features0 = temp_feature
                        temp_features1 = temp_feature1
                        temp_features2 = temp_feature2
                        temp_labels = temp_label
                        temp_cpc_labels = temp_cpc_label
                    
                    else:
                        temp_features0 = np.vstack((temp_features0, temp_feature))
                        temp_features1 = np.vstack((temp_features1, temp_feature1))
                        temp_features2 = np.vstack((temp_features2, temp_feature2))
                        temp_labels = np.hstack((temp_labels, temp_label))
                        temp_cpc_labels = np.hstack((temp_cpc_labels, temp_cpc_label))
                    
            
            else:
                temp_feature, temp_label, temp_cpc_label = \
                    self.create_temporal_datasets(temp, timestamps[iter1], 
                                              labels[iter1], cpc_labels[iter1], h, prepop) #&
                
                # -----------------------------------------------------------------
                if len(temp_feature.shape) == 1:
                    temp_feature = np.expand_dims(temp_feature, axis=0)
                    temp_label = temp_label[0]
                    temp_cpc_label = temp_cpc_label[0]
                # -----------------------------------------------------------------
                pos = self.position_encoding_init(temp_feature.shape[0], temp_feature.shape[1])#&
                temp_feature = temp_feature + pos
                features_list.append(temp_feature)
                # -----------------------------------------------------------------
                if iter1 == 0:
                    temp_features = temp_feature
                    temp_labels = temp_label
                    temp_cpc_labels = temp_cpc_label
                
                else:
                    temp_features = np.vstack((temp_features, temp_feature))
                    temp_labels = np.hstack((temp_labels, temp_label))
                    temp_cpc_labels = np.hstack((temp_cpc_labels, temp_cpc_label))
                
        # ---------------------------------------------------------------------
        if clfop == "stacking2":
            temp_features = {"temp_features0": temp_features0, 
                             "temp_features1": temp_features1,
                             "temp_features2":temp_features2}
        # ---------------------------------------------------------------------
        clf, cpc_clf, feature_importance = self.train_model(temp_features, temp_labels, temp_cpc_labels, clfop, ind_feature)

        # ---------------------------------------------------------------------
        return clf, cpc_clf, medians_nan, feature_importance
    # #########################################################################
    def inference(self, pids, features, timestamps, clf, cpc_clf, medians_nan, h, ind_features=None, prepop="mean"):
        
        clfop = "None"
        if isinstance(clf, list):
            clfop = "stacking2"
        
        
        predictions = []
        cpc_predictions = []
        predictions_temps = []
        for iter1 in range(len(pids)):
            
            temp_feature = features[iter1]
            if len(temp_feature.shape) == 1:
                temp_feature = np.expand_dims(temp_feature, axis=0)
            
            
            if ind_features is not None:
                temp_feature = temp_feature[:, ind_features]
            
            temp = self.replace_nan_with_median(temp_feature, "test", medians_nan)
            # -----------------------------------------------------------------
            if clfop == "stacking2":
                
                temp_feature, temp_feature1, temp_feature2, temp_label,\
                    temp_cpc_label, _ = self.create_temporal_datasets_daily(temp, timestamps[iter1], np.nan, np.nan, h, prepop)
                
                # -------------------------------------------------------------
                pos = self.position_encoding_init(temp_feature.shape[0], temp_feature.shape[1])#&
                temp_feature = temp_feature + pos
                temp_feature1 = temp_feature1 + pos
                temp_feature2 = temp_feature2 + pos
                # -------------------------------------------------------------
                predictions_temp = self.stacking_ensemble(temp_feature, temp_feature1, temp_feature2, [], 
                                      learners=clf, mode="test")
            # -----------------------------------------------------------------
            else:
                temp_feature, _, _ = self.create_temporal_datasets(temp, timestamps[iter1], np.nan, np.nan, h, prepop)
            
                pos = self.position_encoding_init(temp_feature.shape[0], temp_feature.shape[1])
                temp_feature = temp_feature + pos
                
                if isinstance(clf, dict):
                    if "scaler" in clf:
                        scaler = clf["scaler"]
                        temp_feature = scaler.transform(temp_feature)
                    if "clf" in clf:
                        clf1 = clf["clf"]
                    if "clf1" in clf:
                        clf1 = clf["clf1"]
                    if "regmodels" in clf:
                        regmodels = clf["regmodels"]
                        temp_feature1 = self.regression_features_pred(temp_feature, regmodels)
                    predictions_temp = clf1.predict_proba(temp_feature1)
                else:
                    predictions_temp = clf.predict_proba(temp_feature)
            # -----------------------------------------------------------------
            # -----------------------------------------------------------------
            if len(predictions_temp.shape) == 1:
                predictions_temp = np.expand_dims(predictions_temp, axis=0)
            # -----------------------------------------------------------------
            predictions_temps.append(predictions_temp)
            if predictions_temp.shape[0] > 1:
                # *************************************************************
                # predictions_temp = np.nanmean(predictions_temp, axis=0, keepdims=True)
                # *************************************************************
                predictions_temps = np.array(predictions_temps)
                predictions_temps = np.squeeze(predictions_temps)
                predictions_temps = predictions_temps[:, 1]
                
                quantiles = np.percentile(predictions_temps, [10, 30, 50, 70, 90])
                weights = np.zeros_like(predictions_temps)
                weights[predictions_temps < quantiles[0]] = 0.2
                weights[(predictions_temps >= quantiles[0]) & (predictions_temps < quantiles[1])] = 0.5
                weights[(predictions_temps >= quantiles[1]) & (predictions_temps < quantiles[2])] = 1
                weights[(predictions_temps >= quantiles[2]) & (predictions_temps < quantiles[3])] = 1.5
                weights[(predictions_temps >= quantiles[3]) & (predictions_temps < quantiles[4])] = 1.8
                weights[predictions_temps >= quantiles[4]] = 2
                
                weighted_avg = np.average(predictions_temps, weights=weights)
                
                predictions_temp = np.zeros((1, 2))
                predictions_temp[0, 1] = weighted_avg
                predictions_temp[0, 0] = 1 - weighted_avg
                # *************************************************************
                # *************************************************************
            # -----------------------------------------------------------------
            cpc_predictions_temp = cpc_clf.predict(temp_feature)
            if cpc_predictions_temp.shape[0] > 1:
                cpc_predictions_temp = np.nanmean(cpc_predictions_temp, axis=0, keepdims=True)
            # -----------------------------------------------------------------
            predictions.append(predictions_temp)
            cpc_predictions.append(cpc_predictions_temp)
        # ---------------------------------------------------------------------
        predictions = np.array(predictions)
        if len(predictions.shape) == 3:
            predictions = np.squeeze(predictions)
        cpc_predictions = np.array(cpc_predictions)
        # ---------------------------------------------------------------------
        return predictions, cpc_predictions, predictions_temps
    # #########################################################################