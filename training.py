from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from itertools import cycle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve
import os
import shap
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from matplotlib.lines import Line2D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_selection import RFE
def train_rf(data_file_name,  save_model, paramater_selection, cv_number, rfe_label, plot_show,shap_or_not):
    #loading  and splitting the data
    df = pd.read_csv(data_file_name)  #first line is feature name
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -1].values
    class_counts = pd.Series(y).value_counts()

    # 打印每个类别的数量
    print(class_counts)
    feature_names = df.columns[:-3]
    X_train, X_test_1, y_train, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    print('the length of the training set  is {}'.format(X_train_1.shape[0]))
    print('the length of the test set is {}'.format(X_test_1.shape[0]))
    print('the length of the val set is {}'.format(X_val_1.shape[0]))

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
        print(feature_ranking)
        X_train_1 = X_train_1[:,selected_features]
        feature_names = feature_names[selected_features]
        X_test_1 = X_test_1[:,selected_features]

    model.fit(X_train_1, y_train_1)
    y_score = model.predict_proba(X_test_1)
    predictions = model.predict(X_test_1)
    all_predictions = model.predict(X)

    # label conversion
    true_label = np.where(y_test_1 == 'a', 'Stable',
                              np.where(y_test_1 == 'b', 'Metastable',
                                       np.where(y_test_1 == 'c', 'Unstable', y_test_1)))
    all_true_label = np.where(y == 'a', 'Stable',
                              np.where(y == 'b', 'Metastable',
                                       np.where(y == 'c', 'Unstable', y)))
    predicted_label = np.where(predictions == 'a', 'Stable',
                                   np.where(predictions == 'b', 'Metastable',
                                            np.where(predictions == 'c', 'Unstable', predictions)))
    all_predicted_label = np.where(all_predictions == 'a', 'Stable',
                                   np.where(all_predictions == 'b', 'Metastable',
                                            np.where(all_predictions == 'c', 'Unstable', all_predictions)))

    #computing accuracy
    accuracy = accuracy_score(true_label, predicted_label)
    print("Accuracy:", accuracy)
    plt.rcParams['font.family'] = 'Arial'
    all_accuracy = accuracy_score(all_true_label, all_predicted_label)
    print("Accuracy on the whole dataset:", all_accuracy)



    #pr曲线
    label_dict = {'a':'Stable','b':'Metastable', 'c': 'Unstable'}
    plt.figure(figsize=(8, 7))
    for i, label in enumerate(model.classes_):
        precision, recall, _ = precision_recall_curve(y_test_1 == label, y_score[:, i])
        plt.plot(recall, precision, label=f'{label_dict[label]} (AUC = {-np.trapz(precision, recall):.2f})')

    plt.xlabel('Recall', weight='bold', fontsize=22)
    plt.ylabel('Precision', weight='bold', fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower left", frameon=False, prop={'weight': 'bold', 'size': '16'})
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    plt.savefig('pr_curve.jpg', dpi = 300)
    if plot_show:
        plt.show()

    # Compute ROC curve and ROC area for each class
    y_test_bin = label_binarize(y_test_1, classes=['a', 'b', 'c'])
    n_classes = y_test_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    label_dict = {1: 'Stable',
                      2: 'Metastable',
                      3: 'Unstable'}
    plt.figure(figsize=(8, 7))
    colors = ['blue', 'green', 'red']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f' {label_dict[i + 1]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate', weight='bold', fontsize=16)
    plt.ylabel('True positive rate', weight='bold', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower right", frameon=False, prop={'weight': 'bold', 'size': '16'})
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    plt.savefig('roc_curve.jpg', dpi=300)
    if plot_show == True:
        plt.show()
    #chart
    report = classification_report(y_test_1, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report)

    #
    # sampled_indices = np.random.choice(X_train_1.shape[0], 200, replace=False)
    # X_train_1_selected = X_train_1[sampled_indices]
    #
    # y_train_1_selected = y_train_1[sampled_indices]
    #
    # pca = PCA(n_components=2)
    # X_train_2d = pca.fit_transform(X_train_1)
    #
    # encoder = LabelEncoder()
    # y_train_1_encoded = encoder.fit_transform(y_train_1)
    #
    # model.fit(X_train_2d, y_train_1_encoded)
    # h = 0.1
    # # x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    # # y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    #
    # x_min, x_max = X_train_2d[:300, 0].min() - 1, X_train_2d[:300, 0].max() + 1
    # y_min, y_max = X_train_2d[:300, 1].min() - 1, X_train_2d[:300, 1].max() + 1
    #
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    #
    # plt.contourf(xx, yy, Z, alpha=0.8)
    # plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_1_encoded, edgecolors='k', marker='o', s=50, cmap=plt.cm.coolwarm)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    #
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label='Class A', markerfacecolor=plt.cm.coolwarm(0 / 2), markersize=10),
    #     Line2D([0], [0], marker='o', color='w', label='Class B', markerfacecolor=plt.cm.coolwarm(1 / 2), markersize=10),
    #     Line2D([0], [0], marker='o', color='w', label='Class C', markerfacecolor=plt.cm.coolwarm(2 / 2), markersize=10),
    # ]
    #
    # plt.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=12)
    #
    # # plt.legend(loc="lower right", frameon=False, prop={'weight': 'bold', 'size': '16'})
    # plt.show()





    # SHAP Analysis
    if shap_or_not == True:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_1)

        # for i, class_name in enumerate(['Class a', 'Class b', 'Class c']):  # 用实际类别名称
        #     print(f"SHAP Summary Plot for {class_name}")
        #
        #     shap.summary_plot(shap_values[:,:,i], X_train_1, feature_names = feature_names)
        #     plt.show()
    # for i, class_name in enumerate(['Class a', 'Class b', 'Class c']):  # 用实际类别名称
    #     print(f"SHAP Summary Plot for {class_name}")
    #     if plot_show == True:
    #         shap.summary_plot(shap_values[i], X_train_1, feature_names=feature_names)

    # for feature_idx in range(X_train_1.shape[1]):
    #     if feature_idx >= shap_values[0].shape[1]:
    #         print(f"Skipping feature {feature_names[feature_idx]} as it is out of bounds in shap_values.")
    #         continue
    #     print(f"Dependence plot for feature: {feature_names[feature_idx]}")
    #     if plot_show:
    #         shap.dependence_plot(feature_idx, shap_values[0], X_train_1, feature_names=feature_names)



    #confusion matrix
    cm = confusion_matrix(true_label, predicted_label, labels=['Stable', 'Metastable', 'Unstable'])
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=['Stable', 'Metastable', 'Unstable'])
    fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
    disp.plot(cmap=plt.cm.Blues, values_format='.1f', ax=ax)
    ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], fontweight='bold', color='black')
    ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels()], fontweight='bold', color='black')
    # plt.savefig('confusion_matrix.png', bbox_inches='tight')
    # ax.set_xlabel('Predicted Label', fontweight='bold')
    # ax.set_ylabel('True Label', fontweight='bold')
    if plot_show == True:
        plt.show()

    # feature importance
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

    #cross validation
    X_cv = X[:, selected_features] if rfe_label else X
    for i in cv_number:
        cv_scores = cross_val_score(model, X_cv, y, cv=i, scoring='accuracy')
        print('{} fold cross validation scores:{}'.format(i,cv_scores))
        print("Average cross-validation score and std: {},{}".format(np.mean(cv_scores), np.std(cv_scores)) )
        print("cross-validation std:", np.std(cv_scores))

    #save model
    if save_model == True:
        joblib_file = 'random_forest_model' + data_file_name.split('/')[-1] +'622' + '.pkl'
        joblib.dump(model, joblib_file)
        print(f"Model saved as {joblib_file}")
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx training was done xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    return model


model_considered =['knn', 'dt','gbt', 'lr', 'nb']
def train_other(data_file_name, save_model, model_type):
    df = pd.read_csv(data_file_name)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

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
    y_score = model.predict_proba(X_test_1)

    predictions = model.predict(X)

    true_label = np.where(y == 'a', 'Stable',
                            np.where(y == 'b', 'Metastable',
                            np.where(y == 'c', 'Unstable', y)))

    predicted_label = np.where(predictions == 'a', 'Stable',
                            np.where(predictions == 'b', 'Metastable',
                            np.where(predictions == 'c', 'Unstable', predictions)))

    accuracy = accuracy_score(true_label, predicted_label)


    print(f"Accuracy of {model_type}:", accuracy)

    plt.rcParams['font.family'] = 'Arial'



    cm = confusion_matrix(true_label, predicted_label, labels=['Stable', 'Metastable','Unstable'])
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=['Stable', 'Metastable', 'Unstable'])
    plt.figure(figsize=(9,7))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


    disp.plot(cmap=plt.cm.Blues, values_format='.1f', ax=ax)
    ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], fontweight='bold', color='black', size = '20')
    ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels()], fontweight='bold', color='black',size = '20')

    for text in ax.texts:
        text.set_fontsize(20)

    plt.tight_layout()
    plt.savefig(f'confusion matrix of {model_type}.jpg', dpi = 400)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", np.mean(cv_scores))
    print("cross-validation 标准差:", np.std(cv_scores))



def predict(data_file_name, model_file):
    model = joblib.load(model_file)

    df = pd.read_csv(data_file_name)
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -1].values

    y_predicted = model.predict(X)

    #X, y, y_predicted的类型都是numpy.ndarray
    print(X.shape)
    print(y.shape)
    print(y_predicted.shape)

    result = np.column_stack((X, y, y_predicted))
    print(result.shape)

    df = pd.DataFrame(result, columns=['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'y', 'y_predicted'])
    df.to_csv('predict_result.csv', index=False)

    accuracy = accuracy_score(y, y_predicted)
    print(accuracy)

    # #画出混淆举证
    plt.rcParams['font.family'] = 'Arial'
    true_label = np.where(y == 'a', 'Stable',
                            np.where(y == 'b', 'Metastable',
                            np.where(y == 'c', 'Unstable', y)))

    predicted_label = np.where(y_predicted == 'a', 'Stable',
                            np.where(y_predicted == 'b', 'Metastable',
                            np.where(y_predicted == 'c', 'Unstable', y_predicted)))
    #
    #
    cm = confusion_matrix(true_label, predicted_label, labels=['Stable', 'Metastable','Unstable'])
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=['Stable', 'Metastable', 'Unstable'])
    plt.figure(figsize=(9,7))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


    disp.plot(cmap=plt.cm.Blues, values_format='.1f', ax=ax)
    ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], fontweight='bold', color='black', size = '20')
    ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels()], fontweight='bold', color='black',size = '20')
    for text in ax.texts:
        text.set_fontsize(20)
    plt.tight_layout()

    plt.savefig(f'confusion matrix of {model_file}.jpg', dpi = 400)

from tqdm import tqdm
from sklearn.metrics import silhouette_score
def error_analy(label_tag):
    df = pd.read_csv('predict_result.csv')
    condition = (df.iloc[:, -2] == 'b') & (df.iloc[:, -1] == 'b')
    filtered_values = df.loc[condition, df.columns[:-2]]
    # condition_2 = (df.iloc[:, -2] == 'a') & (df.iloc[:, -1] == 'b')
    # filtered_values_2 = df.loc[condition_2, df.columns[:-2]]
    # combined_values = pd.concat([filtered_values, filtered_values_2])
    combined_values = filtered_values
    scaler = StandardScaler()
    combined_values = scaler.fit_transform(combined_values)

    # labels = np.array([0] * len(filtered_values) + [1] * len(filtered_values_2))
    labels = np.array([0] * len(filtered_values))

    df = pd.DataFrame(combined_values, columns = df.columns[:-2])
    # performing tsne
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity = 50)
    tsne_results = tsne.fit_transform(df)

    tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE Component 1', 't-SNE Component 2'])
    tsne_df['Label'] = labels

    silhouette_scores = []
    K = range(5, 20)  # 选择要评估的聚类数量范围
    print("Evaluating optimal number of clusters using silhouette scores...")
    for k in tqdm(K):
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=500, tol=1e-6)
        cluster_labels = kmeans.fit_predict(tsne_results)
        silhouette_avg = silhouette_score(tsne_results, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    optimal_k = K[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters determined: {optimal_k}")
    print(f"Clustering data into {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, max_iter=500, tol=1e-6)
    cluster_labels = kmeans.fit_predict(tsne_results)

    tsne_df['Cluster'] = cluster_labels
    #cluster center
    cluster_centers = kmeans.cluster_centers_
    #plot detail
    label_color_map = {0: 'red', 1: 'blue'}



    fig, ax = plt.subplots()
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.figure(figsize=(9, 6))

    plt.rcParams['font.family'] = 'Arial'
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(np.unique(cluster_labels))))

    if label_tag == 'mistake':
        for label in np.unique(labels):
            plt.scatter(tsne_df.loc[tsne_df['Label'] == label, 't-SNE Component 1'],
                    tsne_df.loc[tsne_df['Label'] == label, 't-SNE Component 2'],
                    label=f'Label {label}', color=label_color_map[label])
    else:
        for i, cluster in enumerate(np.unique(cluster_labels)):
            plt.scatter(tsne_df.loc[tsne_df['Cluster'] == cluster, 't-SNE Component 1'],
                    tsne_df.loc[tsne_df['Cluster'] == cluster, 't-SNE Component 2'],
                    label=f'Cluster {cluster+1}', color=colors[i])


    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, c='black', label='Cluster Centers')
    plt.legend(frameon=False, prop={'weight': 'bold'}, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('t-SNE Component 1', weight = 'bold', size = 18)
    plt.ylabel('t-SNE Component 2', weight = 'bold',size = 18)



    plt.tight_layout()
    plt.savefig(f"tsne_of_predictin_with {label_tag}.jpg", dpi=600)
    # mean value
    df['Cluster'] = cluster_labels
    cluster_means = df.groupby('Cluster').mean()
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    print("Cluster Means:")
    print(cluster_means.to_string())




    # pca = PCA(n_components=2)
    # pca_results_1 = pca.fit_transform(filtered_values_scaled)
    # pca_results_2 = pca.fit_transform(filtered_values_2_scaled)

    # 绘制 PCA 图
    # plt.figure(figsize=(8, 6))
    # plt.scatter(pca_results_1[:, 0], pca_results_1[:, 1], label='unstable to metastable', color='blue', alpha=0.6)
    # plt.scatter(pca_results_2[:, 0], pca_results_2[:, 1], label='stable to metastable', color='green', alpha=0.6)
    #
    # plt.title('PCA Analysis of Filtered Values')
    # plt.xlabel('PCA Component 1', weight = 'bold')
    # plt.ylabel('PCA Component 2', weight = 'bold')
    # plt.legend(frameon=False, prop={'weight': 'bold'})
    # plt.xticks(fontsize=12)  # 设置x轴刻度字体大小
    # plt.yticks(fontsize=12)
    # plt.show()

def subset_training():
    train_ratios = [0.05] + [0.1 * i for i in range(1, 10)]
    df = pd.read_csv('data_12/0.10.csv')  #first line is feature name
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-3]
    accuracy_1_list = []
    accuracy_2_list = []
    accuracy_3_list = []

    for ratio in train_ratios:
        print(f"\nTraining with { int(1-ratio * 100)}% of the data:")
        X_used, X_unused, y_used, y_unused = train_test_split(X, y, test_size=1 - ratio, random_state=42)
        model = RandomForestClassifier(n_estimators=100,
                                   criterion='gini',
                                   max_depth = 15,
                                   min_samples_split=4,
                                   min_samples_leaf=1,
                                   max_features='sqrt',
                                   min_impurity_decrease=0.0,
                                   bootstrap= False,
                                   oob_score = False,
                                   n_jobs = -1,
                                   random_state=42)
        X_1, X_test, y_1, y_test = train_test_split(X_used, y_used, test_size=0.2, random_state = 42)
        X_train, X_val, y_train, y_val = train_test_split(X_1, y_1, test_size=0.25, random_state = 42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy_1 = accuracy_score(y_test, y_pred)
        accuracy_1_list.append(accuracy_1)

        y_pred_2 = model.predict(X_used)
        accuracy_2 = accuracy_score(y_used, y_pred_2)
        accuracy_2_list.append(accuracy_2)

        y_pred_2 = model.predict(X)
        accuracy_3 = accuracy_score(y, y_pred_2)
        accuracy_3_list.append(accuracy_3)

        joblib_file = 'subset_training' + str(ratio)+'.pkl'
        joblib.dump(model, joblib_file)
        print(f"Model saved as {joblib_file}")

    print('accuracy on tesetset, subset, whole dataset')
    print(accuracy_1_list)
    print(accuracy_2_list)
    print(accuracy_3_list)
from itertools import combinations
def feature_palie():
    df = pd.read_csv('data_12/0.10.csv')  #first line is feature name
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
            'accuracy': accuracy,
            'all_feature': all_accuracy
        })
            print('training with {} is done'.format(selected_feature_names))
            indices += 1
            print(indices)
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    pd.DataFrame(results).to_csv('feature_selection_results.csv', index=False)

def cerition_training():
    results = []
    for value in np.arange(0, 0.31, 0.01):
        file_name = os.path.join('data_12',(str("{:.2f}".format(value))+'.csv'))
        df = pd.read_csv(file_name)  # first line is feature name
        X = df.iloc[:, :-3].values
        y = df.iloc[:, -1].values
        feature_names = df.columns[:-3]

        X_train, X_test_1, y_train, y_test_1 = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

        model = RandomForestClassifier(n_estimators=100,
                                       criterion='gini',
                                       max_depth=15,
                                       min_samples_split=4,
                                       min_samples_leaf=1,
                                       max_features='sqrt',
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       bootstrap=False,
                                       oob_score=False,
                                       n_jobs=-1,
                                       random_state=42)
        model.fit(X_train_1, y_train_1)

        predictions = model.predict(X_test_1)
        all_predictions = model.predict(X)

        # label conversion
        true_label = np.where(y_test_1 == 'a', 'Stable',
                              np.where(y_test_1 == 'b', 'Metastable',
                                       np.where(y_test_1 == 'c', 'Unstable', y_test_1)))
        all_true_label = np.where(y == 'a', 'Stable',
                                  np.where(y == 'b', 'Metastable',
                                           np.where(y == 'c', 'Unstable', y)))
        predicted_label = np.where(predictions == 'a', 'Stable',
                                   np.where(predictions == 'b', 'Metastable',
                                            np.where(predictions == 'c', 'Unstable', predictions)))
        all_predicted_label = np.where(all_predictions == 'a', 'Stable',
                                       np.where(all_predictions == 'b', 'Metastable',
                                                np.where(all_predictions == 'c', 'Unstable', all_predictions)))

        accuracy = accuracy_score(true_label, predicted_label)

        all_accuracy = accuracy_score(all_true_label, all_predicted_label)

        results.append({

            'accuracy': accuracy,
            'all_feature': all_accuracy
        })
        joblib_file = 'cerition_training' + str("{:.2f}".format(value))+'.pkl'
        joblib.dump(model, joblib_file)
        print(f"Model saved as {joblib_file}")
    pd.DataFrame(results).to_csv('cerition_training.csv', index=False)

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


def combined_prediction():
    data_file = 'application_ABX3.csv'
    df = pd.read_csv(data_file)
    df_new = df.iloc[:, 2:]
    df_new = df_new.dropna()

    df_new.iloc[:, -1] = df_new.iloc[:, -1].apply(
            lambda x: 'a' if float(x) == 0 else 'b' if 0 < float(x) < 0.1 else 'c')
    df_new = df_new.iloc[:, :-3].join(df_new.iloc[:, -1:])
    X = df_new.iloc[:, :-1].values
    y = df_new.iloc[:, -1].values

    y_counts = pd.Series(y).value_counts()
    print('真实的结果为:')
    indices_a = [i for i, label in enumerate(y) if label == 'a']
    first_column_values = df.iloc[indices_a, 0].values
    # print(first_column_values)
    print(y_counts)


    model_1_path  = 'cerition_training0.10.pkl'
    model_2_path  = 'cerition_training0.09.pkl'
    model_3_path  = 'cerition_training0.08.pkl'
    model_4_path  = 'cerition_training0.07.pkl'
    model_1 = joblib.load(model_1_path)
    model_2 = joblib.load(model_2_path)
    model_3 = joblib.load(model_3_path)
    model_4 = joblib.load(model_4_path)

    y_predicted_1 = model_1.predict(X)
    y_predicted_2 = model_2.predict(X)
    y_predicted_3 = model_3.predict(X)
    y_predicted_4 = model_4.predict(X)

    def calculate_accuracy_for_category(predicted, true, category='a'):
        # 计算真实为'category'的样本预测为'category'的比例
        true_category = true == category
        predicted_category = predicted == category
        correct_predictions = sum(true_category & predicted_category)
        total_category = sum(true_category)
        category_accuracy = correct_predictions / total_category if total_category > 0 else 0
        return category_accuracy

    print('第一次预测结果为:')
    y_final = y_predicted_1
    print(pd.Series(y_final).value_counts())
    print(accuracy_score(y, y_final))
    print(f"类别 'a' 的准确度为: {calculate_accuracy_for_category(y_final, y, 'a' ):.2f}")
    print(f"类别 'b' 的准确度为: {calculate_accuracy_for_category(y_final, y, 'b' ):.2f}")
    print(f"类别 'c' 的准确度为: {calculate_accuracy_for_category(y_final, y, 'c' ):.2f}")

    print('第2次预测结果为:')
    print(pd.Series(y_predicted_2).value_counts())
    print(accuracy_score(y, y_predicted_2))
    print(f"类别 'a' 的准确度为: {calculate_accuracy_for_category(y_predicted_2, y, 'a'):.2f}")

    print('第3次预测结果为:')
    print(pd.Series(y_predicted_3).value_counts())
    print(accuracy_score(y, y_predicted_3))
    print(f"类别 'a' 的准确度为: {calculate_accuracy_for_category(y_predicted_3, y, 'a'):.2f}")

    print('第4次预测结果为:')
    print(pd.Series(y_predicted_4).value_counts())
    print(accuracy_score(y, y_predicted_4))
    print(f"类别 'a' 的准确度为: {calculate_accuracy_for_category(y_predicted_4, y, 'a'):.2f}")

    print('综合234次预测结果为:')
    y_combined = []
    for i in range(len(y)):
        votes = [y_predicted_2[i], y_predicted_3[i], y_predicted_4[i]]
        majority_vote = max(set(votes), key=votes.count)  # 找到票数最多的标签
        y_combined.append(majority_vote)

    print(pd.Series(y_combined).value_counts())
    print(accuracy_score(y, y_combined))
    print(f"类别 'a' 的准确度为: {calculate_accuracy_for_category(y_combined, y, 'a'):.2f}")

    # 使用 y_combined 矫正 y_final
    y_corrected = y_final.copy()  # 创建 y_final 的副本进行修改
    for i in range(len(y_final)):
        if y_final[i] == 'b' and y_combined[i] == 'a':
            y_corrected[i] = 'a'

        if y_final[i] == 'b' and y_combined[i] == 'c':
            y_corrected[i] = 'c'

    print('矫正后的预测结果为:')
    print(pd.Series(y_corrected).value_counts())
    print('矫正后的准确率:')
    print(accuracy_score(y, y_corrected))
    print(f"类别 'a' 的准确度为: {calculate_accuracy_for_category(y_corrected, y, 'a'):.2f}")

    categories = ['a', 'b', 'c']
    for category in categories:
        true_category = y == category
        predicted_category = y_corrected == category
        correct_predictions = sum(true_category & predicted_category)
        total_category = sum(true_category)
        category_accuracy = correct_predictions / total_category if total_category > 0 else 0
        print(f"类别 {category} 的准确度: {category_accuracy:.2f}")









if __name__ == '__main__':
    # subset_training()
    # for i in model_considered:
    #     train_other('data_12/0.10.csv','a',i)

    # error_analy('cluster')
    # train_rf('data_12/0.10.csv',False,False,[5,10],10,True,False)

    # feature_palie()

    # cerition_training()
    # predict('data_12/0.10.csv', 'cerition_training0.10.pkl')
    # predicting_curve()

    combined_prediction()

