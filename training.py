from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import precision_recall_curve, auc
import os
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_selection import RFE
from imblearn.over_sampling import RandomOverSampler
import sys
from utilies import cif_reader
from itertools import combinations
from collections import Counter
from utilies import compute_entropy
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import time

def train_rf(data_file_name, save_model, sample_balance, paramater_selection, cv_number, rfe_label, pr_auc, cm_or_not, fi, cross_v, threshold):
    total_start = time.time()

    #loading  and splitting the data
    df = pd.read_csv(data_file_name)  #first line is feature name
    df = df.dropna()

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values


    y = np.where(y > threshold, 'c',
                 np.where((y > 0) & (y <= threshold), 'b', 'a'))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    class_counts = pd.Series(y).value_counts()



    print('xxxxxxxxxxxxxxxxxxxxxxxxx count of each catergray xxxxxxxxxxxxxxxxxxxxxxxx')
    print(class_counts)
    if sample_balance == 'oversampling':
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)

    elif sample_balance == 'undersampling':
        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(X, y)

    elif sample_balance == 'smote':
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    # 打印平衡后的类别数
    if sample_balance in ['oversampling', 'undersampling', 'smote']:
        class_counts_resampled = pd.Series(y).value_counts()
        print('xxxxxxxxxxxxxxxxxxxxxxxxx count of each category (after balancing) xxxxxxxxxxxxxxxxxxxxxxxx')
        print(class_counts_resampled)

    feature_names = df.columns[:-1]
    X_train, X_test_1, y_train, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)


    #selection of hyper_paramater
    if paramater_selection == True:
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
            'max_depth': [13, 15, 17, 19],  # Maximum depth of the tree
            'min_samples_split': [2, 3, 4, 5],  # Minimum number of samples to split a node
            'min_samples_leaf': [1, 2, 3, 4],  # Minimum number of samples in a leaf node
            'max_features': ['sqrt','log2'],  # Number of features to consider for split
            'bootstrap':[True, False]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1,
                                   n_jobs=-1)
        grid_search.fit(X_train_1, y_train_1)
        best_params = grid_search.best_params_
        print("Best parameters found by grid search:", best_params)
        sys.exit()
    else:
        model = RandomForestClassifier(n_estimators=100,
                                   criterion='gini',
                                   max_depth = 15,
                                   min_samples_split=4,
                                   min_samples_leaf=1,
                                   max_features='sqrt',
                                   max_leaf_nodes= None,
                                   min_impurity_decrease=0.0,
                                   bootstrap= False,
                                   oob_score = False,
                                   n_jobs = -1,
                                   random_state=42)

    # Feature selection with RFE
    if rfe_label != False:
        #rfe selector
        rfe = RFE(estimator=model, n_features_to_select=rfe_label)
        rfe.fit(X_train_1, y_train_1)
        selected_features = rfe.support_
        feature_ranking = rfe.ranking_
        print('RFE RANGKING IS {}'.format(feature_ranking))
        X_train_1 = X_train_1[:,selected_features]
        feature_names = feature_names[selected_features]
        X_test_1 = X_test_1[:,selected_features]
        X = X[:,selected_features]
    t0 = time.time()


    model.fit(X_train_1, y_train_1)
    print(f"RF training time: {time.time() - t0:.4f} seconds")
    y_score = model.predict_proba(X_test_1)
    predictions = model.predict(X_test_1)
    all_predictions = model.predict(X)
    y_score_all = model.predict_proba(X)


    plt.rcParams['font.family'] = 'Arial'
    classes = ['a', 'b', 'c']
    label_dict = {'a': 'Stable', 'b': 'Metastable', 'c': 'Unstable'}
    #computing accuracy

    accuracy = accuracy_score(y_test_1, predictions)
    print('test accuracy is {}'.format(accuracy))
    t0 = time.time()
    all_accuracy = accuracy_score(y, all_predictions)
    print('all accuracy is {}'.format(all_accuracy))
    print(f"predicting all 180 k materials stability: {time.time() - t0:.4f} seconds")
    #f1 score
    f1_per_class = f1_score(y, all_predictions, average=None)  # 每个类别
    f1_macro = f1_score(y, all_predictions, average="macro")
    f1_weighted = f1_score(y, all_predictions, average="weighted")
    #confusion matrix
    cm = confusion_matrix(y, all_predictions, labels=['a', 'b', 'c'])

    #auc
    present_classes = [cls for cls in classes if cls in y]  # only 2 classed when ehull = 0
    if len(present_classes) == 2:
        y_true_bin = label_binarize(y, classes=present_classes)  # shape (n_samples, 2)
        # 取正类列的概率
        y_score_pos = y_score_all[:, 1]  # 第二列是正类概率
        auc_per_class = roc_auc_score(y_true_bin, y_score_pos)
        auc_macro = auc_per_class
    if len(present_classes) > 2:
        y_true_bin = label_binarize(y, classes=present_classes)



        auc_per_class = roc_auc_score(y_true_bin, y_score_all, average=None, multi_class="ovr")
        auc_macro = roc_auc_score(y_true_bin, y_score_all, average="macro", multi_class="ovr")

    #save stastic results
    results = {
        'test accuracy':accuracy,
        'all accuracy': all_accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc_macro": auc_macro,
        "confusion_matrix": cm.tolist()  #cm[i][j] represents the number of samples whose true class is i and that are predicted as class j.
    }

    for i, cls in enumerate(present_classes):
        results[f"f1_{cls}"] = f1_per_class[i]
        if len(present_classes) == 2:
            results[f"auc_{cls}"] = auc_per_class
        else:
            results[f"auc_{cls}"] = auc_per_class[i]



    if pr_auc == True:
        # pr data saving
        pr_data = []
        max_length_pr = 0
        classes = list(label_dict.keys())
        pr_dict = {label: {'Recall': [], 'Precision': [], 'Threshold': []} for label in classes}
        for i, label in enumerate(classes):
            precision, recall, thresholds_pr = precision_recall_curve(y_test_1 == label, y_score[:, i])
            pr_auc = auc(recall, precision)
            pr_dict[label]['Recall'] = recall.tolist()
            pr_dict[label]['Precision'] = precision.tolist()
            pr_dict[label]['Threshold'] = thresholds_pr.tolist()
            pr_dict[label]['AUC'] = pr_auc
            max_length_pr = max(max_length_pr, len(recall))

        for label in classes:
            while len(pr_dict[label]['Recall']) < max_length_pr:
                pr_dict[label]['Recall'].append(np.nan)
            while len(pr_dict[label]['Precision']) < max_length_pr:
                pr_dict[label]['Precision'].append(np.nan)
            while len(pr_dict[label]['Threshold']) < max_length_pr:
                pr_dict[label]['Threshold'].append(np.nan)

        for i in range(max_length_pr):
            row = []
            for label in classes:
                row.append(pr_dict[label]['Recall'][i])
                row.append(pr_dict[label]['Precision'][i])
                row.append(pr_dict[label]['Threshold'][i])
            row.append([pr_dict[label]['AUC'] for label in classes])
            pr_data.append(row)

        columns_pr = []
        for label in classes:
            columns_pr.append(f'{label_dict[label]} Recall')
            columns_pr.append(f'{label_dict[label]} Precision')
            columns_pr.append(f'{label_dict[label]} Threshold')
        columns_pr.append('AUC')

        df_pr = pd.DataFrame(pr_data, columns=columns_pr)
        df_pr.to_csv('pr_curve.csv', index=False)

        # roc data save

        y_test_bin = label_binarize(y_test_1, classes=classes)
        roc_data = []
        roc_dict = {label: {'FPR': [], 'TPR': [], 'Threshold': [], 'auc': []} for label in classes}
        max_length_roc = 0
        for i, label in enumerate(classes):
            fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_score[:, i])
            auc_value = roc_auc_score(y_test_bin[:, i], y_score[:, i])

            roc_dict[label]['FPR'] = fpr.tolist()
            roc_dict[label]['TPR'] = tpr.tolist()
            roc_dict[label]['Threshold'] = thresholds.tolist()
            roc_dict[label]['auc'] = auc_value
            max_length_roc = max(max_length_roc, len(fpr))
        for label in classes:
            while len(roc_dict[label]['FPR']) < max_length_roc:
                roc_dict[label]['FPR'].append(np.nan)
            while len(roc_dict[label]['TPR']) < max_length_roc:
                roc_dict[label]['TPR'].append(np.nan)
            while len(roc_dict[label]['Threshold']) < max_length_roc:
                roc_dict[label]['Threshold'].append(np.nan)

        for i in range(max_length_roc):
            row = []
            for label in classes:
                row.append(roc_dict[label]['FPR'][i])
                row.append(roc_dict[label]['TPR'][i])
                row.append(roc_dict[label]['Threshold'][i])
            row.append([roc_dict[label]['auc'] for label in classes])
            roc_data.append(row)

        # 创建DataFrame
        columns_roc = []
        for label in classes:
            columns_roc.append(f'{label_dict[label]} FPR')
            columns_roc.append(f'{label_dict[label]} TPR')
            columns_roc.append(f'{label_dict[label]} Threshold')
        columns_roc.append('AUC')
        df_roc = pd.DataFrame(roc_data, columns=columns_roc)
        df_roc.to_csv('roc_curve.csv', index=False)

    #chart
    report = classification_report(y, all_predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)





    #confusion matrix
    if cm_or_not == True:
        plt.rcParams['font.family'] = 'Arial'
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage,
                                      display_labels=['Stable', 'Metastable', 'Unstable'])

        plt.figure(figsize=(9, 7))
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        disp.plot(cmap=plt.cm.Blues, values_format='.1f', ax=ax)
        ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], fontweight='bold', color='black',
                           size='20')
        ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels()], fontweight='bold', color='black',
                           size='20')

        for text in ax.texts:
            text.set_fontsize(20)

        plt.tight_layout()


        plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=400)



    # feature importance
    if fi == True:
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print(importance_df)

    #cross validation
    if cross_v == True:
        X_cv = X
        for i in cv_number:
            cv_scores = cross_val_score(model, X_cv, y, cv=i, scoring='accuracy')
            print('{} fold cross validation scores:{}'.format(i,cv_scores))
            print("Average cross-validation score and std: {},{}".format(np.mean(cv_scores), np.std(cv_scores)) )

    errors = (y != all_predictions)
    error_rate = np.mean(errors)
    print(error_rate)


    #save model
    if save_model == True:
        joblib_file = 'RF_model'  +str(threshold) + '.pkl'
        joblib.dump(model, joblib_file)
        print(f"Model saved as {joblib_file}")
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx training was done xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    return model, results, accuracy, all_accuracy


model_considered =['knn', 'dt','gbt', 'lr', 'nb']
def train_other(data_file_name,  model_type, cm_or_not, cv_or_not, threshold):
    df = pd.read_csv(data_file_name)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = np.where(y > threshold, 'c',
                 np.where((y > 0) & (y <= threshold), 'b', 'a'))

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    if model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    if model_type == 'dt':
        model = DecisionTreeClassifier(criterion='entropy', max_depth=13, random_state=42)
    if model_type == 'gbt':
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    if model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
    if model_type == 'nb':
        model = GaussianNB()

    X_train, X_test_1, y_train, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=42,
                                                        stratify=y)
    X_train_1, X_val, y_train_1, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    model.fit(X_train_1, y_train_1)


    predictions = model.predict(X)

    true_label = np.where(y == 'a', 'Stable',
                            np.where(y == 'b', 'Metastable',
                            np.where(y == 'c', 'Unstable', y)))

    predicted_label = np.where(predictions == 'a', 'Stable',
                            np.where(predictions == 'b', 'Metastable',
                            np.where(predictions == 'c', 'Unstable', predictions)))

    accuracy = accuracy_score(true_label, predicted_label)
    print(f"Accuracy of {model_type}:", accuracy)
    if cm_or_not == True:
        plt.rcParams['font.family'] = 'Arial'
        cm = confusion_matrix(true_label, predicted_label, labels=['Stable', 'Metastable', 'Unstable'])
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage,
                                      display_labels=['Stable', 'Metastable', 'Unstable'])
        plt.figure(figsize=(9, 7))
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        disp.plot(cmap=plt.cm.Blues, values_format='.1f', ax=ax)
        ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], fontweight='bold', color='black',
                           size='20')
        ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels()], fontweight='bold', color='black',
                           size='20')

        for text in ax.texts:
            text.set_fontsize(20)

        plt.tight_layout()
        plt.savefig(f'confusion matrix of {model_type}.jpg', dpi=400)


    if cv_or_not == True:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print("Cross-validation scores:", cv_scores)
        print("Average cross-validation score:", np.mean(cv_scores))
        print("cross-validation standard error:", np.std(cv_scores))
    return accuracy
from search import AutoEncoder
import torch

'''this is for the investigate of using different ratio of the whole data for training'''
def subset_training():
    train_ratios = [0.05] + [0.1 * i for i in range(1, 10)]
    df = pd.read_csv('data_33_unnormalized.csv')
    df = df.dropna()

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    y = np.where(y > 0.1, 'c',
                 np.where((y > 0) & (y <= 0.1), 'b', 'a'))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # 用于存储结果
    results = []

    for ratio in train_ratios:
        print(f"\nTraining with {ratio*100:.1f}% of the data:")
        X_used, X_unused, y_used, y_unused = train_test_split(
            X, y, test_size=1 - ratio, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,
            oob_score=False,
            n_jobs=-1,
            random_state=42
        )

        X_1, X_test, y_1, y_test = train_test_split(
            X_used, y_used, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_1, y_1, test_size=0.25, random_state=42
        )

        model.fit(X_train, y_train)

        # accuracy1：测试集
        accuracy_test = accuracy_score(y_test, model.predict(X_test))

        # accuracy2：subset 内部所有数据
        accuracy_subset = accuracy_score(y_used, model.predict(X_used))

        # accuracy3：所有数据全集
        accuracy_whole = accuracy_score(y, model.predict(X))

        # 记录结果
        results.append({
            "train_ratio": ratio,
            "test_accuracy": accuracy_test,
            "subset_accuracy": accuracy_subset,
            "whole_accuracy": accuracy_whole
        })

        joblib.dump(model, f"subset_training_{ratio}.pkl")
        print(f"Model saved as subset_training_{ratio}.pkl")

    # 保存 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("subset_training_results.csv", index=False)
    print("\nResults saved to subset_training_results.csv")
    print(results_df)

from sklearn.inspection import permutation_importance
'''data_folder contains cifs, model_file is the model loaded'''
def predict(data_folder, model_file, shap_or_not ,threshold = None, record = True):
    start_time = time.time()
    if os.path.exists("5A1B_r.csv"):
        print("📄 Found hea_data.csv — loading directly ...")
        dfs = pd.read_csv("5A1B_r.csv")
    else:
        df = []
        for i in os.listdir(data_folder):
            df0 = cif_reader(os.path.join(data_folder, i))
            df.append(df0)
            print(i)
        dfs = pd.concat(df, ignore_index=True)
        print('data loading is finished')
        dfs.to_csv("5A1B_r.csv", index=False)
    dfs = dfs.dropna()
    model = joblib.load(model_file)
    encoder_model = AutoEncoder(input_dim=35, latent_dim=16)
    encoder_model.load_state_dict(torch.load("autoencoder_latent_16.pth"))
    encoder_model.eval()
    scaler = StandardScaler()

    X = dfs.iloc[:, 1:].values
    X_scaled = scaler.fit_transform(X)
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        _, X_new = encoder_model(X_scaled)


    X_new = scaler.fit_transform(X_new)
    y_predicted = model.predict(X_new)

    col_name = "y_predicted"
    df_predict = dfs.iloc[:, [0]].copy()  # 取第一列当 id
    df_predict.columns = ["id"]
    df_predict[col_name] = y_predicted


    if record:
        df_save = dfs.copy()
        df_save[col_name] = y_predicted

        df_save.to_csv(f'predict_result_'+ str(threshold) +'1210.csv', index=False)
        label_counts = df_save[col_name].value_counts()
        print(f'predict result (threshold={threshold}):')
        print(label_counts)


    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱ Total running time: {elapsed:.2f} seconds")



    if shap_or_not == True:
        print("\nComputing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_new)
        for i, class_name in enumerate(['Class a', 'Class b', 'Class c']):  # 用实际类别名称
            print(f"SHAP Summary Plot for {class_name}")
            plt.figure(figsize=(8, 7))

            shap.summary_plot(shap_values[:, :, i], X_new, feature_names=list(range(1, 17)), max_display=8, plot_size=(8,7), show=False)
            plt.xlabel("SHAP Value", fontsize=22, fontweight='bold')
            plt.ylabel("Feature", fontsize=22, fontweight='bold')
            plt.xticks(fontsize=18, fontweight='bold')
            plt.yticks(fontsize=18, fontweight='bold')

            output_file_svg = f"shap_summary_plot_{class_name}.tif"
            plt.savefig(output_file_svg, format='tif', dpi=600, bbox_inches='tight')




'''test different feature combinations (1024)'''
def feature_combination():
    df = pd.read_csv('data/0.10.csv')  #first line is feature name
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-3]
    indices = 0
    results = []
    for num_features in range (1, len(feature_names)+1):
        for feature_indices in combinations(range(len(feature_names)), num_features):
            # Select the features for this combination
            X_selected = X[:, feature_indices]
            y_selected = y
            selected_feature_names = feature_names[list(feature_indices)]

            X_train, X_test_1, y_train, y_test_1 = train_test_split(
                X_selected, y, test_size=0.2, random_state=42, stratify=y)
            X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(
                X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)


            model = RandomForestClassifier(n_estimators=100,
                                   criterion='gini',
                                   max_depth = 15,
                                   min_samples_split=4,
                                   min_samples_leaf=1,
                                   max_features='sqrt',
                                   max_leaf_nodes= None,
                                   min_impurity_decrease=0.0,
                                   bootstrap= False,
                                   oob_score = False,
                                   n_jobs = -1,
                                   random_state=42)
            model.fit(X_train_1, y_train_1)

            predictions = model.predict(X_test_1)
            all_predictions = model.predict(X_selected)

    # label conversion
            true_label = np.where(y_test_1 == 'a', 'Stable',
                              np.where(y_test_1 == 'b', 'Metastable',
                                       np.where(y_test_1 == 'c', 'Unstable', y_test_1)))
            all_true_label = np.where(y == 'a', 'Stable',
                              np.where(y == 'b', 'Metastable',
                                       np.where(y == 'c', 'Unstable', y_selected)))
            predicted_label = np.where(predictions == 'a', 'Stable',
                                   np.where(predictions == 'b', 'Metastable',
                                            np.where(predictions == 'c', 'Unstable', predictions)))
            all_predicted_label = np.where(all_predictions == 'a', 'Stable',
                                   np.where(all_predictions == 'b', 'Metastable',
                                            np.where(all_predictions == 'c', 'Unstable', all_predictions)))

    #computing accuracy
            accuracy = accuracy_score(true_label, predicted_label)
            all_accuracy = accuracy_score(all_true_label, all_predicted_label)
            results.append({
            'num_features': num_features,
            'features': selected_feature_names.tolist(),
            'accuracy on test set': accuracy,
            'accuracy on whole dataset': all_accuracy
        })
            print('training with {} is done'.format(selected_feature_names))
            indices += 1
            print(indices)
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    pd.DataFrame(results).to_csv('feature_combination_results.csv', index=False)


def predicting_curve():
    results = []
    for value in np.arange(0, 0.31, 0.01):
        model_file = os.path.join('cerition_training'+(str("{:.2f}".format(value))+'.pkl'))
        model = joblib.load(model_file)
        accuracy_list = []
        for i in np.arange(0, 0.31, 0.01):
            data_file = os.path.join('data_12',(str("{:.2f}".format(i))+'.csv') )
            df = pd.read_csv(data_file)
            X = df.iloc[:, :-3].values
            y = df.iloc[:, -1].values
            y_predicted = model.predict(X)
            accuracy = accuracy_score(y, y_predicted)
            accuracy_list.append(accuracy)

            print('model {} on data file {} is done'.format(value, i))
        results.append([f"{value:.2f}"] + accuracy_list)


    column_names = ["model_used"] + [f"accuracy on _{i:.2f}" for i in np.arange(0, 0.31, 0.01)]

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=column_names)

    # Save to CSV
    results_df.to_csv("model_accuracy_matrix.csv", index=False)

def model_analysis():
    results = []
    for value in np.arange(0, 0.31, 0.01):
        model_file = os.path.join('cerition_training'+(str("{:.2f}".format(value))+'.pkl'))
        model = joblib.load(model_file)

        if hasattr(model, "tree_"):  # 检查是否是单颗决策树
            print(f"Model {value:.2f}: Depth = {model.tree_.max_depth}, Nodes = {model.tree_.node_count}")
        elif hasattr(model, "estimators_"):  # 检查是否是随机森林或梯度提升树
            depths = [est.tree_.max_depth for est in model.estimators_]
            avg_depth = sum(depths) / len(depths)
            leaves = sum(est.tree_.n_leaves for est in model.estimators_) / len(model.estimators_)
            nodes = sum(est.tree_.node_count for est in model.estimators_) / len(model.estimators_)
        data_file = os.path.join('data_12', (str("{:.2f}".format(value)) + '.csv'))
        df = pd.read_csv(data_file)
        X = df.iloc[:, :-3].values
        y = df.iloc[:, -1].values
        y_predicted = model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        results.append([value,  avg_depth, leaves, nodes, accuracy])
    df_results = pd.DataFrame(results, columns=["Threshold",  "Avg Depth", "Leaves", "Nodes", "Accuracy"])
    df_results.to_csv("model_analysis_results.csv", index=False, encoding="utf-8-sig")

def label_entrophy_compare():



    prediction_file = 'predict_result.csv'
    entropy_file = 'ce.csv'

    df_pred = pd.read_csv(prediction_file)     # 第一列：id，第二列：label(a/b/c)
    df_pred = df_pred.iloc[:,[0,-1]]
    df_ent  = pd.read_csv(entropy_file)        # 第一列：id，第二列：entropy
    df_ent = df_ent.iloc[:, [0, 2]]

    # 重命名列，防止重复
    df_pred.columns = ['id', 'label']
    df_ent.columns  = ['id', 'entropy']

    # 合并
    df = pd.merge(df_pred, df_ent, on='id', how='inner')

    # 按 label 顺序排序（a → b → c）
    df['label'] = pd.Categorical(df['label'], categories=['a', 'b', 'c'], ordered=True)
    df = df.sort_values('label')

    # 保存结果
    df.to_csv('label_entropy_merged.csv', index=False)

    label_counts = df['label'].value_counts().reindex(['a', 'b', 'c']).fillna(0).astype(int)
    for label, count in label_counts.items():
        print(f"标签 {label} 的数量: {count}")


import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics import pairwise_distances


def Distance_to_centroid():
    values = [0.04, 0.07, 0.10, 0.13]

    for v in values:
        file_path = os.path.join('data', f"{v:.2f}.csv")
        df = pd.read_csv(file_path)

        X = df.iloc[:, :-3].values
        y = df.iloc[:, -1].values
        classes = np.unique(y)

        # -------- 计算类中心 --------
        centroids = {}
        for c in classes:
            X_c = X[y == c]
            centroids[c] = X_c.mean(axis=0)

        # -------- 每个样本到类中心的距离 --------
        records = []
        for idx in range(len(X)):
            c = y[idx]
            d = np.linalg.norm(X[idx] - centroids[c])
            if c != 'a':
                records.append([idx,c,d])

        df_out = pd.DataFrame(records, columns=["sample_id", "class", "distance"])

        out_path = os.path.join('results', f"distances_{v:.2f}.csv")
        os.makedirs('results', exist_ok=True)
        df_out.to_csv(out_path, index=False)

        # -------- 计算类间距离 --------
        centroid_matrix = np.vstack([centroids[c] for c in classes])
        between_dists = pairwise_distances(centroid_matrix, metric="euclidean")

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                print(v)
                print(between_dists[i, j])

import glob
def hea_stastic():
    files = sorted(glob.glob("predict_result_0.*.pkl.csv"))

    dfs = []
    thresholds = []

    for f in files:
        # 获取阈值，例如 0.05
        th = f.split("_")[-1].replace(".pkl.csv", "")
        thresholds.append(th)

        df = pd.read_csv(f)
        df = df.iloc[:, [0, -1]]  # 只取第一列和最后一列
        df.columns = ["id", f"pred_{th}"]
        dfs.append(df)

    # 2. 合并数据（以 id 为 key）
    from functools import reduce
    df_merge = reduce(lambda left, right: pd.merge(left, right, on="id", how="outer"), dfs)

    # 保存合并文件
    df_merge.to_csv("combined_predictions.csv", index=False)
    print("✅ 合并文件已保存：combined_predictions.csv")

    # 3. 分析 11 个标签是否一致
    label_cols = [col for col in df_merge.columns if col.startswith("pred_")]

    df_merge["all_a"] = df_merge[label_cols].apply(lambda x: all(v == "a" for v in x), axis=1)
    df_merge["all_b"] = df_merge[label_cols].apply(lambda x: all(v == "b" for v in x), axis=1)
    df_merge["all_c"] = df_merge[label_cols].apply(lambda x: all(v == "c" for v in x), axis=1)

    # 提取列表
    all_a_list = df_merge[df_merge["all_a"]]["id"].tolist()
    all_b_list = df_merge[df_merge["all_b"]]["id"].tolist()
    all_c_list = df_merge[df_merge["all_c"]]["id"].tolist()

    print("\n========== 统计结果 ==========")
    print(f"🔹 标签均为 a 的个数：{len(all_a_list)}")
    print(f"🔹 标签均为 b 的个数：{len(all_b_list)}")
    print(f"🔹 标签均为 c 的个数：{len(all_c_list)}")

    # 保存结果
    pd.DataFrame({
        "all_a": all_a_list,
        "all_b": all_b_list,
        "all_c": all_c_list
    }).to_csv("label_consistency_summary.csv", index=False)

    print("📄 一致性统计文件已保存：label_consistency_summary.csv")


import csv

if __name__ == '__main__':
    # for i in np.arange(0.01, 0.16, 0.01):
    # train_rf('encoded_latent_16.csv', True, 'smote', False, 5, False, False, False, False, False, 0.10)
    # filenames = [f"RF_model0.{i:02d}.pkl" for i in range(1, 16)]
    # for i in filenames:
    # predict('generated_oxides_lattice', i, i.split('model')[1])



    predict('5A1B_r', 'RF_model0.10.pkl', True)


