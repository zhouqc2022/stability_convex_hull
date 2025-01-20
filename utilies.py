import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
#




#     #将化学式转化为化学系统
def formula_to_chemsys(formula):

    elements = re.findall(r'[A-Z][a-z]*|\d+', formula)
    elements_without_numbers = [element for element in elements if not element.isdigit()]
    chemsys = '-'.join(sorted(set(elements_without_numbers)))

    return chemsys

#将包含目标的文件转化为可用于CGCNN学习的id_prop.csv
def txt_transfer():
    new_lines = []
    with open('unstable_3_materials.txt','r')as file:
        lines = file.readlines()
    for i in lines:
        list = i.split(',')

        list = list[:2]

        list.append(list[-1])

        new_lines.append(list)

    with open('id_prop.csv','w')as file:
        for sublist in new_lines:
            line = ','.join(map(str, sublist)) + '\n'
            file.write(line)

#计算某一个文件中target的平均值
def average_calculator():
    with open ('unstable_3_formation_energy','r')as file:
        lines = file.readlines()
    lines = [x.split(',')[1] for x in lines]
    lines = [float(x) for x in lines]
    total = 0
    for num in lines:
        total += abs(num)
    average = total/len(lines)
    print('average is {}'.format(average))

#对包含Fe的600余种晶体做回归，随机森林算法
def rf():
    file_path = 'test_new.txt'
    df = pd.read_csv(file_path, header=None, names=["name", "target", "feature1", "feature2", "feature3", "feature4", "feature5"])

    X = df[["feature1", "feature2", "feature3", "feature4", "feature5"]]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 创建和训练随机森林模型
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    # 进行预测
    y_pred = best_model.predict(X_test)




    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    y_test = list(y_test)
    y_pred = list(y_pred)
    data = list(zip(y_test, y_pred))

    with open('output2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入两列数据

        writer.writerows(data)
#对包含Fe的600余种晶体做三分类随机森林算法
def rf_2():
    file_path = 'test_Fe_unstable.txt'
    df = pd.read_csv(file_path, header=None, names=["name", "target", "feature1", "feature2", "feature3", "feature4", "feature5"])

    X = df[["feature1", "feature2", "feature3", "feature4", "feature5"]]
    y = df["target"]

    X = np.array(X)
    y = ['a' if x == 0 else 'b' for x in list(y)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Predictions:", predictions)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    #混淆举证
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=['stable', 'unstable'], yticklabels=['stable', 'unstable'],
                annot_kws={"color": "red"})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


    #ROC曲线
    y_score = model.predict_proba(X_test)[:, 1]  # 获取正类的概率
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label='b')  # 计算ROC曲线
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    #importance
    feature_importance = model.feature_importances_

    # 创建特征重要性的DataFrame
    importance_df = pd.DataFrame({'Feature': df.columns[2:], 'Importance': feature_importance})

    # 排序特征重要性
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)
    # plt.figure(figsize=(10, 6))
    # plt.bar(importance_df['Feature'], importance_df['Importance'])
    # plt.xlabel('Feature')
    # plt.ylabel('Importance')
    # plt.title('Feature Importance')
    # plt.xticks(rotation=45)
    # plt.show()

#对包含Fe的600余种晶体做三分类随机森林算法
def rf_3():
    file_path = 'test_Fe_unstable.txt'
    df = pd.read_csv(file_path, header=None, names=["name", "target", "feature1", "feature2", "feature3", "feature4", "feature5"])

    X = df[["feature1", "feature2", "feature3", "feature4", "feature5"]]
    y = df["target"]

    X = np.array(X)
    y = ['a' if x == 0 else 'b' if 0 < x < 0.1 else 'c' for x in list(y)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Predictions:", predictions)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    y_score = model.predict_proba(X_test)[:, 1]  # 获取正类的概率
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label='b')  # 计算ROC曲线
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    feature_importance = model.feature_importances_

    # 创建特征重要性的DataFrame
    importance_df = pd.DataFrame({'Feature': df.columns[2:], 'Importance': feature_importance})

    # 排序特征重要性
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

#总结formation energy
def all_f():
    with open('formation_all','w')as file:
        a = os.listdir('data/formation energy')
        for i in a:
            file_path = os.path.join('data/formation energy', i)
            with open(file_path,'r')as file_2:
                lines = file_2.readlines()
            for k in lines:
                name = k.split(',')[0]
                for_energy = k.split(',')[1]
                new_line = name+','+for_energy
                file.write('{}'.format(new_line))
        file.close()

#总结e_hull
def all_e():

    with open('energy_above_hull_all_stable', 'w') as file:
        a = os.listdir('data/stable')

        for i in a:
            file_path = os.path.join('data/stable', i)
            with open(file_path, 'r') as file_2:
                lines = file_2.readlines()
            for k in lines:
                name = k.split(',')[0]
                for_energy = k.split(',')[1]
                new_line = name + ',' + for_energy
                file.write('{}\n'.format(new_line))
        file.close()

    with open('energy_above_hull_all_unstable', 'w') as file:
        b = os.listdir('data/unstable')
        for i in b:
            file_path = os.path.join('data/unstable', i)
            with open(file_path, 'r') as file_2:
                lines = file_2.readlines()
            for k in lines:
                name = k.split(',')[0]
                for_energy = k.split(',')[1]
                new_line = name + ',' + for_energy
                file.write('{}\n'.format(new_line))
        file.close()

#绘制e_hull vs ef
def f_vs_e():

    material_dict = {}
    with open ('formation_all', 'r') as file:
        lines = file.readlines()
        file.close()
    with open ('energy_above_hull_all', 'r') as file_2:
        lines_2 = file_2.readlines()



    for i in lines:
        name = i.split(',')[0]
        formation_energy = i.split(',')[1].split('\n')[0]
        material_dict[name] = formation_energy


    for i in lines_2:
        if i.split(',')[0] in material_dict.keys():
            material_dict[(i.split(',')[0])] += (','+ i.split(',')[1].split('\n')[0])

    with open('summary','w') as file:
        for key,value in material_dict.items():
            file.write('{},{}\n'.format(key,value))

#绘制数量柱状图
def count():
    energy_list = []
    count_list = []
    with open('formation_all','r')as file:
        lines = file.readlines()
    for i in lines:
        energy_list.append(float(i.split(',')[1]))

    start_at = -5



    for i in range(1,111,1):
        count_num = 0
        for k in energy_list:
            print(start_at+0.1*i)
            if start_at+0.1*i < k < start_at+0.1*(i+1):
                count_num += 1
        count_list.append(count_num)

    with open('count_2','w')as file:
        for i in count_list:
            file.write('{}\n'.format(float(i)))

#查看不同scale里面样本的数量
def scale():
    e_hull_list = []
    with open ('data/unstable/unstable_4_materials', 'r') as file:
        lines = file.readlines()
    for i in lines:
        e_hull_list.append(float(i.split(',')[1]))

    sorted_e_hull_list = sorted(e_hull_list)



    num_1 = 0

    for i in e_hull_list:
        if i>0.1:
            num_1 +=1



    with open('data/stable/stable_4_materials', 'r') as file:
        lines_2 = file.readlines()
    print('0 meV is {}'.format(len(lines_2)))
    print('0 to 100 meV is {}'.format((len(e_hull_list)-num_1)))
    print('more than 100 meV is {}'.format(num_1))

    print('material containg N is 11396')

element_symbols = {
    "H": "Hydrogen",
    "He": "Helium",
    "Li": "Lithium",
    "Be": "Beryllium",
    "B": "Boron",
    "C": "Carbon",
    "N": "Nitrogen",
    "O": "Oxygen",
    "F": "Fluorine",
    "Ne": "Neon",
    "Na": "Sodium",
    "Mg": "Magnesium",
    "Al": "Aluminum",
    "Si": "Silicon",
    "P": "Phosphorus",
    "S": "Sulfur",
    "Cl": "Chlorine",
    "Ar": "Argon",
    "K": "Potassium",
    "Ca": "Calcium",
    "Sc": "Scandium",
    "Ti": "Titanium",
    "V": "Vanadium",
    "Cr": "Chromium",
    "Mn": "Manganese",
    "Fe": "Iron",
    "Co": "Cobalt",
    "Ni": "Nickel",
    "Cu": "Copper",
    "Zn": "Zinc",
    "Ga": "Gallium",
    "Ge": "Germanium",
    "As": "Arsenic",
    "Se": "Selenium",
    "Br": "Bromine",
    "Kr": "Krypton",
    "Rb": "Rubidium",
    "Sr": "Strontium",
    "Y": "Yttrium",
    "Zr": "Zirconium",
    "Nb": "Niobium",
    "Mo": "Molybdenum",
    "Tc": "Technetium",
    "Ru": "Ruthenium",
    "Rh": "Rhodium",
    "Pd": "Palladium",
    "Ag": "Silver",
    "Cd": "Cadmium",
    "In": "Indium",
    "Sn": "Tin",
    "Sb": "Antimony",
    "Te": "Tellurium",
    "I": "Iodine",
    "Xe": "Xenon",
    "Cs": "Cesium",
    "Ba": "Barium",
    "La": "Lanthanum",
    "Ce": "Cerium",
    "Pr": "Praseodymium",
    "Nd": "Neodymium",
    "Pm": "Promethium",
    "Sm": "Samarium",
    "Eu": "Europium",
    "Gd": "Gadolinium",
    "Tb": "Terbium",
    "Dy": "Dysprosium",
    "Ho": "Holmium",
    "Er": "Erbium",
    "Tm": "Thulium",
    "Yb": "Ytterbium",
    "Lu": "Lutetium",
    "Hf": "Hafnium",
    "Ta": "Tantalum",
    "W": "Tungsten",
    "Re": "Rhenium",
    "Os": "Osmium",
    "Ir": "Iridium",
    "Pt": "Platinum",
    "Au": "Gold",
    "Hg": "Mercury",
    "Tl": "Thallium",
    "Pb": "Lead",
    "Bi": "Bismuth",
    "Po": "Polonium",
    "At": "Astatine",
    "Rn": "Radon",
    "Fr": "Francium",
    "Ra": "Radium",
    "Ac": "Actinium",
    "Th": "Thorium",
    "Pa": "Protactinium",
    "U": "Uranium",
    "Np": "Neptunium",
    "Pu": "Plutonium",
    "Am": "Americium",
    "Cm": "Curium",
    "Bk": "Berkelium",
    "Cf": "Californium",
    "Es": "Einsteinium",
    "Fm": "Fermium",
    "Md": "Mendelevium",
    "No": "Nobelium",
    "Lr": "Lawrencium",
    "Rf": "Rutherfordium",
    "Db": "Dubnium",
    "Sg": "Seaborgium",
    "Bh": "Bohrium",
    "Hs": "Hassium",
    "Mt": "Meitnerium",
    "Ds": "Darmstadtium",
    "Rg": "Roentgenium",
    "Cn": "Copernicium",
    "Nh": "Nihonium",
    "Fl": "Flerovium",
    "Mc": "Moscovium",
    "Lv": "Livermorium",
    "Ts": "Tennessine",
    "Og": "Oganesson"
}


def cohesive_energy_loader(cohesive_energy_file):

    with open(cohesive_energy_file, 'r')as file:
        lines = file.readlines()
    cohesive_dict = {}
    for i in lines:
        name = i.split(',')[0]
        value_in_eV = float(i.split(',')[-1])
        cohesive_dict[name] = value_in_eV
    return cohesive_dict

def box_plot(data):
    labels = ['Atom number', 'Element number', 'Density', 'Magnetic atom number',
              'Crystal system', 'Space group', 'Average cohesive energy']

    # 创建子图
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(20, 6), sharey=False)

    # 自定义颜色
    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink', 'lightgrey', 'lightcoral']

    # 绘制每个特征的箱线图
    for i, ax in enumerate(axes):
        box = ax.boxplot(data[:, i], patch_artist=True, vert=True)
        for patch in box['boxes']:
            patch.set_facecolor(colors[i])
        for flier in box['fliers']:
            flier.set(marker='o', color='red', alpha=0.5, markersize=6)
        ax.set_title(labels[i])
        ax.set_ylabel('Values')

    # 设置总体标题和布局
    plt.suptitle('Box Plot of Feature Data')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 显示图像
    plt.show()

def data_collector():
    df_list = []
    total_rows = 0

    path_list = os.listdir('data/features')
    for i in path_list:
        file_path = os.path.join('data/features', i)
        df = pd.read_csv(file_path)
        df_list.append(df)

        num_rows = df.shape[0]
        print(f"File {i} has {num_rows} rows.")
        total_rows += num_rows

    combined_df = pd.concat(df_list, ignore_index=True)
    final_row_count = combined_df.shape[0]
    print(f"Total number of rows in the final combined file: {final_row_count}")

    combined_df.to_csv('data/features/combined_features.csv', index=False)

def data_miner(target_row_ratio):


    df = pd.read_csv('data/features/combined_features.csv', header=0)
    target_row_count = int((target_row_ratio/100)*len(df))
    df_sampled = df.sample(n=target_row_count, random_state=42)
    df_sampled.to_csv('{}-{}.csv'.format(target_row_count,target_row_ratio), index=False)

def feature_miner():
    df = pd.read_csv('data/features/combined_features_100meV.csv', header=0)

    columns = df.columns.tolist()

    # 分别去掉每一列并保存为新的文件
    for i in range(9):
        # 创建一个新的DataFrame，去掉第 i 列
        df_new = df.drop(columns=[columns[i]])

        # 保存为新的CSV文件，文件名为 1.csv, 2.csv, ..., 9.csv
        df_new.to_csv(f'{i + 1}.csv', index=False)

    print('Files have been created successfully!')
from sklearn.preprocessing import MinMaxScaler
def final_plot():
    df = pd.read_csv('linear.csv', header=0)

    # 提取三列数据
    x = df.iloc[:, 0]  # 第一列作为横坐标
    y = df.iloc[:, 1]  # 第二列作为纵坐标
    size = df.iloc[:, 2]  # 第三列作为散点大小

    # 对第三列进行归一化处理
    scaler = MinMaxScaler(feature_range=(50, 100))  # 设置大小范围
    size_normalized = scaler.fit_transform(size.values.reshape(-1, 1)).flatten()

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=size_normalized, alpha=0.7, c='lightblue', edgecolors='w')

    for i in range(len(x)):
        plt.text(x[i], y[i], f'{size[i]:.0f}', fontsize=9, ha='right', va='bottom')

    coefficients = np.polyfit(x, y, deg=1)  # 一次多项式拟合 (线性拟合)
    slope, intercept = coefficients

    # 生成拟合直线上的点
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept

    # 绘制拟合的直线
    plt.plot(x_fit, y_fit, color='red', label=f'Fitted Line: y = {slope:.2f}x + {intercept:.2f}')

    # 设置图形的标题和标签
    plt.title('Scatter plot with size based on normalized third column')
    plt.xlabel('First Column (X-axis)')
    plt.ylabel('Second Column (Y-axis)')

    # 显示图形
    plt.tight_layout()
    plt.show()

def diff_search():
    df_1  = pd.read_csv('data/features/combined_features_100meV.csv', header=0)
    df_2  = pd.read_csv('data.csv', header=0)
    # Extract the first column from both DataFrames
    col_1 = df_1.iloc[:, 2]  # df_1的第三列
    col_2 = df_2.iloc[:, 4]  # df_2的第五列

    # Round the values to 8 significant digits
    col_1_rounded = col_1.round(12)  # 对df_1的第三列取8位有效数字
    col_2_rounded = col_2.round(12)  # 对df_2的第五列取8位有效数字

    # Find the differences
    # 在col_1中但不在col_2中的值
    diff_in_1 = col_1_rounded[~col_1_rounded.isin(col_2_rounded)]

    # 在col_2中但不在col_1中的值
    diff_in_2 = col_2_rounded[~col_2_rounded.isin(col_1_rounded)]

    # Print the differences
    print("Values in df_1's third column (rounded to 8 digits) not in df_2's fifth column:")
    print(diff_in_1.tolist())  # 将Series转换为列表进行打印

    print("\nValues in df_2's fifth column (rounded to 8 digits) not in df_1's third column:")
    print(diff_in_2.tolist())


def calculate_distance_uniformity(atom_positions):
    # 计算原子之间的所有对的距离
    distances = []
    num_atoms = atom_positions.shape[0]

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # 计算欧几里得距离
            distance = np.linalg.norm(atom_positions[i] - atom_positions[j])
            distances.append(distance)

    distances = np.array(distances)

    # 计算均值和标准差
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    return mean_distance, std_distance


def count_3():
    df = pd.read_csv('data_12/data_12_nonmetal.csv')
    df_new = df.iloc[:, 2:]
    df_new = df_new.dropna()
    df_new.iloc[:, -1] = df_new.iloc[:, -1].apply(
        lambda x: 'a' if float(x) == 0 else 'b' if 0 < float(x) < 0.1 else 'c')

    counts = df_new.iloc[:, -1].value_counts()

    # 打印结果
    print("Counts of each category:")
    print("a:", counts.get('a', 0))
    print("b:", counts.get('b', 0))
    print("c:", counts.get('c', 0))

def variance_threshold_filter():
    df = pd.read_csv('data_12/0.10.csv')
    df_new = df.iloc[:,:-1]

    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_new), columns=df_new.columns)


    mean_values = df_new.mean()
    variance_values = df_new.var()
    median_values = df_new.median()
    quartiles = df_new.quantile([0.25, 0.5, 0.75])
    min_values = df_new.min()
    max_values = df_new.max()

    quartiles_normalized = df_normalized.quantile([0.25, 0.5, 0.75])
    print("\n归一化后的四分位数：\n", quartiles_normalized)

    # 输出统计量
    print("每列的平均值：\n", mean_values)
    print("\n每列的方差：\n", variance_values)
    print("\n每列的中位数：\n", median_values)
    print("\n每列的四分位数：\n", quartiles)
    print("\n每列的最小值：\n", min_values)
    print("\n每列的最大值：\n", max_values)
    plt.rcParams['font.family'] = 'Arial'
    # 绘制箱线图
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_normalized, medianprops=dict(color='red', linewidth=2))

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.xlabel("Features", fontsize=18, fontweight='bold')
    plt.ylabel("Values", fontsize=18, fontweight='bold')


    plt.savefig('feature_box_plot.jpg', dpi = 400)
import shap
def train_rf_test():
    df = pd.read_csv('data_12/0.10.csv')  #first line is feature name
    X = df.iloc[:1000, :-3].values
    y = df.iloc[:1000, -1].values
    feature_names = df.columns[:-3]
    X_train, X_test_1, y_train, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    X_train_1, X_val, y_train_1, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
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
    y_score = model.predict_proba(X_test_1)
    predictions = model.predict(X_test_1)

    plt.rcParams['font.family'] = 'Arial'

    # label conversion
    true_label = np.where(y_test_1 == 'a', 'Stable',
                              np.where(y_test_1 == 'b', 'Metastable',
                                       np.where(y_test_1 == 'c', 'Unstable', y_test_1)))
    predicted_label = np.where(predictions == 'a', 'Stable',
                                   np.where(predictions == 'b', 'Metastable',
                                            np.where(predictions == 'c', 'Unstable', predictions)))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_1)


    print(shap_values.shape)
    print(X_train_1.shape)
    print(X_train.shape)


    for i, class_name in enumerate(['Class a', 'Class b', 'Class c']):  # 用实际类别名称
        print(f"SHAP Summary Plot for {class_name}")
        shap.summary_plot(shap_values[:,:,i], X_train_1, feature_names=feature_names, show = False)


        plt.xlabel("SHAP Value", fontsize=18, fontweight='bold')
        plt.ylabel("Feature", fontsize=18, fontweight='bold')
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')





        # 保存图像
        output_file = f"shap_summary_plot_{class_name}.jpg"
        plt.savefig(output_file, dpi=400, bbox_inches='tight')  # 高分辨率保存图片
        plt.show()

    # for feature_idx in range(X_train_1.shape[1]):
    #     if feature_idx >= shap_values[0].shape[1]:
    #         print(f"Skipping feature {feature_names[feature_idx]} as it is out of bounds in shap_values.")
    #         continue
    #     print(f"Dependence plot for feature: {feature_names[feature_idx]}")
    #     if plot_show:
    #         shap.dependence_plot(feature_idx, shap_values[0], X_train_1, feature_names=feature_names)

def box_and_swarm_plot():
    df = pd.read_csv("feature_selection_results.csv")

    # 提取横纵坐标需要的列
    df['num_features'] = df['num_features'].astype(str)  # 确保横轴为类别型数据
    plt.rcParams['font.family'] = 'Arial'
    # 创建图形
    plt.figure(figsize=(9, 8))

    # 绘制box plot
    sns.boxplot(x='num_features', y='accuracy', data=df, palette='coolwarm', showmeans=True, meanline=True,
                meanprops={"color": "black", "linestyle": "--", "linewidth": 2})

    # 绘制swarm plot
    sns.swarmplot(x='num_features', y='accuracy', data=df, color='black', alpha=0.7)

    # 添加标签和标题
    plt.xlabel("Number of Features", fontsize=22, weight='bold')
    plt.ylabel("Accuracy", fontsize=22, weight='bold')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    # 调整刻度字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 显示图形
    plt.tight_layout()
    plt.show()

import matplotlib.colors as mcolors
def heat_plot():
    # 读取 CSV 文件
    df = pd.read_csv("model_accuracy_matrix.csv")
    # 去掉第一列（"model_used"），并转换为 NumPy 数组以便处理
    data = df.iloc[:, 1:].values  # 排除第一列的字符串列，提取数值
    # 计算调整后的热图数据
    heatmap_data = data - data.diagonal()[:, None]# 按行减去对角元素
    # heatmap_data[heatmap_data > 0] = 1
    heatmap_data_no_ones = np.copy(heatmap_data)  # 创建一个副本以避免修改原数据
    # 找到除去 1 之后的最大值
    max_val = np.max(heatmap_data_no_ones)
    min_val = np.min(heatmap_data_no_ones)  # 找到最小值

    print(min_val, max_val)# 找到最大值
    print(heatmap_data_no_ones[11,:])
    # 使用归一化公式将数据缩放到 [0, 1] 区间
    normalized_data = (heatmap_data_no_ones - min_val) / (max_val - min_val)
    # center_value = (0- min_val) / (max_val - min_val)
    # normalized_data[normalized_data > center_value] = 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        normalized_data,
        cmap='coolwarm',  # 使用 coolwarm 颜色映射
        annot=False,  # 显示数值
        fmt=".2f",   # 格式化数值显示

        xticklabels=[f"{float(i.split('_')[-1])*1000:.0f}" for i in df.columns[1:]],
        yticklabels=[f"{float(i)*1000:.0f}" for i in df["model_used"]],
        )
    # 设置标题和轴标签
    plt.rcParams['font.family'] = 'Arial'
    plt.xlabel("Criterion used for dataset label", fontsize=22, weight='bold')
    plt.ylabel("Model",fontsize=22, weight='bold')
    plt.tight_layout()
    plt.savefig('model_vs_data_matrix.jpg', dpi = 300)


if __name__ == '__main__':
    train_rf_test()