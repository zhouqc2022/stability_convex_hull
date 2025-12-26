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
import os.path
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import pandas as pd
from math import log2
from pymatgen.core import Element
from pymatgen.analysis.local_env import CrystalNN
from search import mass_dict
import re
from search import metals
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import shap
from imblearn.over_sampling import RandomOverSampler, SMOTE

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
#     #将化学式转化为化学系统
def formula_to_chemsys(formula):

    elements = re.findall(r'[A-Z][a-z]*|\d+', formula)
    elements_without_numbers = [element for element in elements if not element.isdigit()]
    chemsys = '-'.join(sorted(set(elements_without_numbers)))

    return chemsys
def cohesive_energy_loader(cohesive_energy_file):

    with open(cohesive_energy_file, 'r')as file:
        lines = file.readlines()
    cohesive_dict = {}
    for i in lines:
        name = i.split(',')[0]
        value_in_eV = float(i.split(',')[-1])
        cohesive_dict[name] = value_in_eV
    return cohesive_dict

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


def cif_reader(cif_file):
    parser = CifParser(cif_file)
    structure = parser.get_structures()[0]

    composition = structure.composition
    nn_finder = CrystalNN()

    composition_dict = {el.symbol: composition.get_atomic_fraction(el) * composition.num_atoms
                        for el in composition}
    feature_list = []
    #1
    feature_list.append(os.path.basename(cif_file))
    #1

    ele_props = ["atomic_mass", 'atomic_radius', 'melting_point', "thermal_conductivity"]
    for p in ele_props:
        vals, weights = [], []
        for sym in composition_dict:
            elem = Element(sym)
            if hasattr(elem, p):
                val = getattr(elem, p)
                if val is not None:
                    vals.append(val)
                    weights.append(composition_dict[sym])
        if len(vals) > 0:
            mean_p = np.average(vals, weights=weights)
            std_p = np.sqrt(np.average((np.array(vals) - mean_p)**2, weights=weights))
        else:
            mean_p, std_p = 0, 0
        feature_list += [mean_p, std_p]

    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    feature_list += [a, b, c, alpha, beta, gamma]
    # 14
    lattice_volume = structure.lattice.volume
    lattice_anisotropy = max(a, b, c) / min(a, b, c)



    latt = np.array(structure.lattice.matrix)
    _, Sigma, _ = np.linalg.svd(latt)
    max_singular_value = Sigma[0]


    coord_nums = [
        len(nn_finder.get_nn_info(structure, i))
        for i in range(len(structure))
    ]
    mean_coord_num = np.mean(coord_nums)
    std_coord_num = np.std(coord_nums)



    atom_number = structure.num_sites
    elements_number = len(composition.elements)

    feature_list += [
        lattice_volume, lattice_anisotropy, max_singular_value,
        mean_coord_num, std_coord_num,atom_number, elements_number]

    #21
    density_atomic = atom_number / lattice_volume
    magnetic_elements = [
        "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",  # 3d 过渡金属
        "Sc", "Ti", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",  # 可选过渡金属
        "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"  # 稀土元素，部分磁性
    ]
    magnetic_count = sum(1 for site in structure if site.specie.symbol in magnetic_elements)


    all_mass = 0
    up = 0
    down = 0
    metal_number = 0
    non_metal_number = 0
    cohesive_dict = cohesive_energy_loader('element_info/cohesive_energy.csv')
    for key, value in composition_dict.items():
        if key in cohesive_dict.keys():
            up += cohesive_dict[key] * value
        else:
            up += 0
        all_mass += mass_dict[key] * value
        if key in metals:
            metal_number += 1
        else:
            non_metal_number += 1
        down += value

    average_cohesive_energy = up / down

    mass_average = all_mass / down

    if non_metal_number != 0:
        metal_non_metal = metal_number / non_metal_number
    else:
        metal_non_metal = 0

    sga = SpacegroupAnalyzer(structure)
    crystal_sys =sga.get_crystal_system()
    space_group_index = sga.get_space_group_number()
    feature_list += [density_atomic, magnetic_count, average_cohesive_energy, metal_non_metal,
                crystal_sys, space_group_index]

    #27

    distances = []



    coordination = structure.cart_coords
    mean_distance, std_distance = calculate_distance_uniformity(coordination)

    for m in range(len(coordination)):
        for n in range(m + 1, len(coordination)):
            distances.append(np.linalg.norm(coordination[m] - coordination[n]))
    if len(distances) > 3:
        skewness = skew(distances)
        kurt = kurtosis(distances)
    else:
        skewness = kurt = 0

    cutoff = 3.0  # 可根据元素类型调整，若要自动我能帮你做

    coord_nums = []

    for m in range(len(coordination)):
        count = 0
        for n in range(len(coordination)):
            if m != n:
                d = np.linalg.norm(coordination[m] - coordination[n])
                if d < cutoff:
                    count += 1
        coord_nums.append(count)

    # 平均配位数
    avg_coordination = np.mean(coord_nums)

    weights = np.array(list(composition_dict.values()), dtype=float)
    en_values = [Element(sym).X for sym in composition_dict.keys() if Element(sym).X]
    if len(en_values) > 1:
        mean_en = np.average(en_values, weights=weights)
        en_diff = np.sqrt(np.average((np.array(en_values) - mean_en) ** 2, weights=weights))
    else:
        mean_en, en_diff = 0, 0

    # 配位数的分布均匀程度（越小越均匀）
    coordination_uniformity = np.std(coord_nums)
    feature_list += [mean_distance, std_distance,
                skewness, kurt, avg_coordination, coordination_uniformity, mean_en, en_diff]

    #35
    df = pd.DataFrame([feature_list])
    return df




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

def train_shap(threshold):
    df = pd.read_csv('encoded_latent_16.csv')  #first line is feature name
    num_samples = min(10000, len(df))
    sampled_idx = np.random.choice(len(df), num_samples, replace=False)

    # 根据随机索引选取数据
    X = df.iloc[sampled_idx, :-1].values
    y = df.iloc[sampled_idx, -1].values

    y = np.where(y > threshold, 'c',
                 np.where((y > 0) & (y <= threshold), 'b', 'a'))

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    feature_names = list(range(1, 17))
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



    '''summary plot'''

    for i, class_name in enumerate(['Class a', 'Class b', 'Class c']):  # 用实际类别名称
        print(f"SHAP Summary Plot for {class_name}")
        plt.figure(figsize=(8, 7))

        shap.summary_plot(shap_values[:,:,i], X_train_1, feature_names=feature_names, show = False)
        plt.xlabel("SHAP Value", fontsize=18, fontweight='bold')
        plt.ylabel("Feature", fontsize=18, fontweight='bold')
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')

        output_file_svg = f"shap_summary_plot_{class_name}.tif"
        plt.savefig(output_file_svg, format='tif', dpi=600, bbox_inches='tight')




    '''decision plot'''


    #
    # shap_values = explainer.shap_values(X_test_1)
    # print(shap_values.shape)#shap_values 的形状通常是 (num_samples, num_features,n_classes )
    #
    # # 选择部分样本绘制决策图
    # num_samples = 200  # 选择 200 个样本
    # sample_indices = np.random.choice(X_test_1.shape[0], num_samples, replace=False)
    # X_test_sample = X_test_1[sample_indices]
    # y_test_sample = y_test_1[sample_indices]
    #
    # for i, class_name in enumerate(["Stable", "Metastable", "Unstable"]):
    #
    #     print(f"shap_values[{i}].shape: {shap_values[i].shape}")  # (num_samples, num_features)
    #     print(f"len(feature_names): {len(feature_names)}")
    #     print(explainer.expected_value.shape)
    #
    #
    #
    #     print(f"SHAP Decision Plot for {class_name}")
    #     plt.figure(figsize=(8, 7))
    #     shap.decision_plot(explainer.expected_value[i], shap_values[:,:,i], X_test_1,
    #                        feature_names=feature_names, show = False)
    #
    #     plt.xlabel("Model Output", fontsize=18, fontweight='bold')
    #     plt.ylabel("Samples", fontsize=18, fontweight='bold')
    #     plt.xticks(fontsize=14, fontweight='bold')
    #     plt.yticks(fontsize=14, fontweight='bold')
    #
    #     # 设置边框线宽
    #     ax = plt.gca()
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(3)
    #
    #     ax.grid(False)
    #
    #
    #     output_file_jpg = f"shap_decision_plot_{class_name}.tif"
    #     plt.savefig(output_file_jpg, format='tif', dpi=400)

    #
    #     if class_name == 'Stable':
    #         for sample_idx in sample_indices:
    #             shap.force_plot(explainer.expected_value[i], shap_values[sample_idx, :, i], X_test_1[sample_idx],
    #                         feature_names=feature_names, matplotlib=True, show=False)
    #
    #             force_plot_filename = f"shap_force_plot_{class_name}_sample_{sample_idx}.png"
    #             plt.savefig(force_plot_filename, format='png', dpi=400, bbox_inches='tight')


def box_and_swarm_plot():
    df = pd.read_csv("feature_selection_results.csv")
    print(df.columns)

    # 提取横纵坐标需要的列
    df['num_features'] = df['num_features'].astype(str)  # 确保横轴为类别型数据
    plt.rcParams['font.family'] = 'Arial'
    # 创建图形
    plt.figure(figsize=(16, 14))

    # 绘制box plot
    sns.boxplot(x='num_features', y='all_feature', data=df, palette='coolwarm', showmeans=True, meanline=True,
                meanprops={"color": "black", "linestyle": "--", "linewidth": 3})

    # 绘制swarm plot
    sns.swarmplot(x='num_features', y='all_feature', data=df, color='black', alpha=0.7)

    # 添加标签和标题
    plt.xlabel("Number of Features", fontsize=36, weight='bold')
    plt.ylabel("Accuracy", fontsize=36, weight='bold')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # 调整刻度字体大小
    plt.xticks(fontsize=26, weight='bold')
    plt.yticks(fontsize=26, weight='bold')

    output_file_jpg = f"box and swarm plot.tif"
    plt.savefig(output_file_jpg, format='tif', dpi=400)


import matplotlib.colors as mcolors


from collections import Counter
from math import log2
def distribution_test():
    values = np.arange(0, 0.31, 0.01)
    records = []

    for value in values:
        file_name = os.path.join('data_12', f"{value:.2f}.csv")
        df = pd.read_csv(file_name)
        y = df.iloc[:, -1].values
        counter = Counter(y)
        total = len(y)

        # === 计算熵 ===
        entropy = 0
        for count in counter.values():
            p = count / total
            entropy -= p * log2(p)

        # === 保存比例 + 熵 ===
        records.append({
            "value": value,
            "a": counter.get("a", 0) / total,
            "b": counter.get("b", 0) / total,
            "c": counter.get("c", 0) / total,
            "entropy": entropy
        })

        print(f"value={value:.2f}, Entropy={entropy:.4f}")

    result = pd.DataFrame(records)

    # === 可选：画图 ===
    result.set_index("value")[["a", "b", "c"]].plot(
        kind="bar", stacked=True, figsize=(12, 6))
    plt.ylabel("Proportion")
    plt.title("Relative proportion of a, b, c in each file")
    plt.legend(title="Class")
    plt.show()

    # === 保存为 CSV ===
    result.to_csv("distribution_summary.csv", index=False)





def calc_config_entropy(cif_file, anion_list=['O']):
    """
    计算给定 CIF 结构的构型熵 S_config
    默认把 O 视为阴离子，其他元素为阳离子
    """
    # 读取结构
    structure = Structure.from_file(cif_file)

    # 统计元素个数
    elem_counts = Counter([str(site.specie) for site in structure])
    total_atoms = sum(elem_counts.values())

    # 区分阳离子和阴离子
    cation_counts = {el: cnt for el, cnt in elem_counts.items() if el not in anion_list}
    anion_counts = {el: cnt for el, cnt in elem_counts.items() if el in anion_list}

    # 计算摩尔分数
    cation_total = sum(cation_counts.values())
    anion_total = sum(anion_counts.values())

    cation_x = {el: cnt / cation_total for el, cnt in cation_counts.items()}
    anion_x = {el: cnt / anion_total for el, cnt in anion_counts.items()} if anion_total > 0 else {}

    # 计算构型熵 (只考虑阳离子/阴离子分布的贡献)
    S_cation = -R * sum(x * np.log(x) for x in cation_x.values())
    S_anion = -R * sum(x * np.log(x) for x in anion_x.values()) if anion_x else 0.0

    # 总构型熵
    S_config = S_cation + S_anion



    return S_config


def hea_stastic():
    # file_dict = ['oversampling', 'undersampling', 'smote', 'unbalanced']
    file_dict = ['smote']

    for i in file_dict:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxx{}xxxxxxxxxxxxxxxxxxxxxxxxxx'.format(i))
        file_path = 'all_threshold_prediction_'+ i+'.csv'
        df = pd.read_csv(file_path)

        vote_b_counts = df['vote_b'].value_counts().sort_index()

        print("vote_b 的种类和数量：")
        for k, v in vote_b_counts.items():
            print(f"vote_b = {k}: {v} 个样本")

        max_vote_b = df['vote_b'].max()
        max_vote_b_rows = df[df['vote_b'] == max_vote_b]

        pred_cols = [col for col in df.columns if col.startswith("pred_")]
        pred_cols = sorted(pred_cols, key=lambda x: float(x.split("_")[1]))

        def has_conflict(row):
            preds = row[pred_cols].tolist()
            seen_b = False
            for p in preds:
                if p == "b":
                    seen_b = True
                if seen_b and p == "c":  # 出现 b→c 反转
                    return True
            return False

        max_vote_b_rows["conflict"] = max_vote_b_rows.apply(has_conflict, axis=1)
        stable_rows = max_vote_b_rows[~max_vote_b_rows["conflict"]]



        cif_dir = 'generated_HEOs_A1B5'
        R = 8.314
        entropies = []
        for cif_id in stable_rows["id"]:
            cif_path = os.path.join(cif_dir, f"{cif_id}")
            S_config = calc_config_entropy(cif_path)
            entropies.append(S_config)

        stable_rows = stable_rows.copy()
        stable_rows["S_config(J/mol·K)"] = entropies
        stable_rows["S_config(R)"] = stable_rows["S_config(J/mol·K)"] / R

        # 保存结果
        name_col = "id"
        stable_rows[[name_col, "vote_b", "S_config(J/mol·K)", "S_config(R)"]].to_csv(
            f'stable_materials_{i}.csv', index=False
        )
        print(f"✅ 稳定材料已保存到 stable_materials_{i}.csv")




        stats = {}
        for col in pred_cols:
            counts = df[col].value_counts()
            stats[col] = {
            "b_count": counts.get("b", 0),
            "c_count": counts.get("c", 0),
            "b_ratio": counts.get("b", 0) / len(df),
            "c_ratio": counts.get("c", 0) / len(df),
            }
        stats_df = pd.DataFrame(stats).T
        stats_df.to_csv('metastbale_ratio_'+i+'.csv')

        conflict_rows = []
        for idx, row in df.iterrows():
            preds = row[pred_cols].tolist()
            seen_b = False
            for p in preds:
                if p == "b":
                    seen_b = True
                if seen_b and p == "c":  # 出现反转
                    conflict_rows.append(row)
                    break

        # 转换成 DataFrame
        conflict_df = pd.DataFrame(conflict_rows)
        print('number of conflict materials')
        print(len(conflict_df))

    # 保存到文件（可选）
        conflict_df.to_csv('conflict_materials_' +i +'_under.csv', index=False)


from sklearn.manifold import TSNE

from sklearn.decomposition import PCA


def s_config_vs_prediction():
    #读取构型熵
    # 读取构型熵
    df_entropy = pd.read_csv('config_entropy_results.csv')
    # 第一列是材料名称，第三列是构型熵
    entropy_dict = dict(zip(df_entropy.iloc[:, 0], df_entropy.iloc[:, 2]))

    # 读取预测值
    pred_dict = {}
    df_pred = pd.read_csv('all_threshold_prediction_smote.csv')

    pred_cols = [col for col in df_pred.columns if col.startswith("pred_")]
    # 假设第19列（索引18）是预测值
    for _, row in df_pred.iterrows():
        first_b_col = None
        for col in pred_cols:
            if row[col] == "b":
                first_b_col = col  # 保存列名
                break
        pred_dict[row[0]] = float(col.replace("pred_", ""))
    # 保证键一致
    keys = entropy_dict.keys()  # 或者 pred_dict.keys()

    # 生成 DataFrame
    df_out = pd.DataFrame({
        'Material': list(keys),
        'Configurational_entropy': [entropy_dict[k] for k in keys],
        'first_b': [pred_dict[k] for k in keys]
    })

    # 保存为 CSV
    df_out.to_csv('entropy_vs_first_b.csv', index=False)
    print("saved to entropy_vs_first_b.csv.csv")
from scipy.stats import ttest_ind










def element_count(site):
    file_name = 'stable_materials_smote_a2b5.csv'
    total_counter = Counter()

    df = pd.read_csv(file_name)
    first_column = df.iloc[:, 0]
    if site == 'A':
        for entry in first_column:
            a = entry.split(')')[0].split('(')[-1]
            elements = re.findall(r'[A-Z][a-z]?', a)
            total_counter.update(elements)
    elif site == 'B':
        for entry in first_column:
            a = entry.split('(')[-1].split(')')[0]
            elements = re.findall(r'[A-Z][a-z]?', a)

            # 更新计数
            total_counter.update(elements)

        # 输出统计结果
    print("元素出现次数统计：")
    for elem, count in total_counter.most_common():
            print(f"{elem}: {count}")




def tsne_hea(auto_cluster, file_path, number_of_features, color, pca_before_tsne=False, pca_dim=10):
    df = pd.read_csv(file_path)
    y = df.iloc[:, -1].values

    # ------------------- labels -------------------
    conditions = [
        (y == 'a'),
        (y == 'b'),
        (y == 'c')
    ]
    labels = np.select(conditions, ['a', 'b', 'c'])

    feature_data = df.iloc[:, 1:number_of_features].values

    # ------------------- StandardScaler -------------------
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)
    feature_data = feature_data_scaled

    # ------------------- Sampling -------------------
    np.random.seed(42)
    if feature_data.shape[0] > 10000:
        sampled_indices = np.random.choice(feature_data.shape[0], 10000, replace=False)
        feature_data_sampled = feature_data[sampled_indices]
        labels_sampled = labels[sampled_indices]
    else:
        feature_data_sampled = feature_data
        labels_sampled = labels

    # ------------------- PCA 可选 -------------------
    if pca_before_tsne:
        print(f"Applying PCA → {pca_dim}D before t-SNE...")
        pca = PCA(n_components=pca_dim, random_state=42)
        feature_data_for_tsne = pca.fit_transform(feature_data_sampled)
    else:
        print("Skipping PCA. Using raw scaled features for t-SNE.")
        feature_data_for_tsne = feature_data_sampled

    df_tmp = pd.DataFrame(feature_data_for_tsne)  # 用于 t-SNE 输入

    # ------------------- t-SNE -------------------
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(df_tmp)

    # ------------------- 保存结果 -------------------
    tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE Component 1', 't-SNE Component 2'])
    tsne_df['Label'] = labels_sampled

    # ------------------- Auto cluster -------------------
    if auto_cluster == 'auto':
        silhouette_scores = []
        K = range(5, 20)
        print("Evaluating optimal number of clusters using silhouette scores...")

        for k in tqdm(K):
            kmeans = KMeans(n_clusters=k, random_state=42, max_iter=500, tol=1e-6)
            cluster_labels_test = kmeans.fit_predict(tsne_results)
            silhouette_avg = silhouette_score(tsne_results, cluster_labels_test)
            silhouette_scores.append(silhouette_avg)

        optimal_k = K[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters determined: {optimal_k}")
    else:
        optimal_k = 6

    # ------------------- Final kmeans -------------------
    print(f"Clustering data into {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, max_iter=500, tol=1e-6)
    cluster_labels = kmeans.fit_predict(tsne_results)

    # ------------------- CH & Silhouette -------------------
    if len(np.unique(cluster_labels)) > 1:
        ch_score = calinski_harabasz_score(tsne_results, cluster_labels)
        silhouette_avg = silhouette_score(tsne_results, cluster_labels)
        print(f"Calinski-Harabasz Score: {ch_score:.4f}")
        print(f"Average Silhouette Score: {silhouette_avg:.4f}")
    else:
        ch_score = np.nan
        silhouette_avg = np.nan
        print("⚠️ Only one cluster. CH/Silhouette cannot be computed.")

    ch = calinski_harabasz_score(feature_data, labels)
    sil = silhouette_score(feature_data, labels)

    # ------------------- Attach cluster labels -------------------
    tsne_df['Cluster'] = cluster_labels

    # ------------------- Plotting (完全保留你的原逻辑) -------------------
    plt.figure(figsize=(9, 6))

    plt.rcParams['font.family'] = 'Arial'
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(np.unique(cluster_labels))))

    # 确保坐标轴加粗（你的原逻辑）
    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    label_color_map = {'a': 'red', 'b': 'blue', 'c': 'green'}

    if color == '3':  # 按 label 上色
        for label in np.unique(labels_sampled):
            plt.scatter(
                tsne_df.loc[tsne_df['Label'] == label, 't-SNE Component 1'],
                tsne_df.loc[tsne_df['Label'] == label, 't-SNE Component 2'],
                label=f'Label {label}', color=label_color_map[label]
            )
    else:  # 按 cluster 上色
        for i, cluster in enumerate(np.unique(cluster_labels)):
            plt.scatter(
                tsne_df.loc[tsne_df['Cluster'] == cluster, 't-SNE Component 1'],
                tsne_df.loc[tsne_df['Cluster'] == cluster, 't-SNE Component 2'],
                label=f'Cluster {cluster+1}',
                color=colors[i]
            )

    # cluster centers
    cluster_centers = kmeans.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                marker='x', s=100, c='black', label='Cluster Centers')

    plt.legend(frameon=False, prop={'weight': 'bold'}, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('t-SNE Component 1', weight='bold', size=18)
    plt.ylabel('t-SNE Component 2', weight='bold', size=18)

    plt.tight_layout()

    outname = f"tsne_of_{number_of_features}_features_PCA{pca_before_tsne}_{os.path.basename(file_path)}.jpg"
    plt.savefig(outname, dpi=600)
    print(f"Saved to: {outname}")

    # ------------------- Cluster means -------------------
    df_mean_calc = pd.DataFrame(feature_data_sampled)
    df_mean_calc['Cluster'] = cluster_labels
    cluster_means = df_mean_calc.groupby('Cluster').mean()

    print("Cluster Means:")
    print(cluster_means.to_string())

    return ch_score, silhouette_avg, ch, sil


crystal_system_dict = {
    'Cubic': 1,
    'cubic': 1,
    'Orthorhombic':2,
    'orthorhombic':2,
    'Hexagonal':3,
    'hexagonal':3,
    'Tetragonal':4,
    'tetragonal':4,
    'Trigonal':5,
    'trigonal':5,
    'Triclinic':6,
    'triclinic':6,
    'Monoclinic':7,
    'monoclinic':7,
}

def cif_reader(cif_file):
    parser = CifParser(cif_file)
    structure = parser.get_structures()[0]

    composition = structure.composition
    nn_finder = CrystalNN()

    composition_dict = {el.symbol: composition.get_atomic_fraction(el) * composition.num_atoms
                        for el in composition}
    feature_list = []
    #1
    feature_list.append(os.path.basename(cif_file))
    #1

    ele_props = ["atomic_mass", 'atomic_radius', 'melting_point', "thermal_conductivity"]
    for p in ele_props:
        vals, weights = [], []
        for sym in composition_dict:
            elem = Element(sym)
            if hasattr(elem, p):
                val = getattr(elem, p)
                if val is not None:
                    vals.append(val)
                    weights.append(composition_dict[sym])
        if len(vals) > 0:
            mean_p = np.average(vals, weights=weights)
            std_p = np.sqrt(np.average((np.array(vals) - mean_p)**2, weights=weights))
        else:
            mean_p, std_p = 0, 0
        feature_list += [mean_p, std_p]

    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    feature_list += [a, b, c, alpha, beta, gamma]
    # 14
    lattice_volume = structure.lattice.volume
    lattice_anisotropy = max(a, b, c) / min(a, b, c)



    latt = np.array(structure.lattice.matrix)
    _, Sigma, _ = np.linalg.svd(latt)
    max_singular_value = Sigma[0]


    coord_nums = [
        len(nn_finder.get_nn_info(structure, i))
        for i in range(len(structure))
    ]
    mean_coord_num = np.mean(coord_nums)
    std_coord_num = np.std(coord_nums)



    atom_number = structure.num_sites
    elements_number = len(composition.elements)

    feature_list += [
        lattice_volume, lattice_anisotropy, max_singular_value,
        mean_coord_num, std_coord_num,atom_number, elements_number]

    #21
    density_atomic = atom_number / lattice_volume
    magnetic_elements = [
        "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",  # 3d 过渡金属
        "Sc", "Ti", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",  # 可选过渡金属
        "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"  # 稀土元素，部分磁性
    ]
    magnetic_count = sum(1 for site in structure if site.specie.symbol in magnetic_elements)


    all_mass = 0
    up = 0
    down = 0
    metal_number = 0
    non_metal_number = 0
    cohesive_dict = cohesive_energy_loader('element_info/cohesive_energy.csv')
    for key, value in composition_dict.items():
        if key in cohesive_dict.keys():
            up += cohesive_dict[key] * value
        else:
            up += 0
        all_mass += mass_dict[key] * value
        if key in metals:
            metal_number += 1
        else:
            non_metal_number += 1
        down += value

    average_cohesive_energy = up / down

    mass_average = all_mass / down

    if non_metal_number != 0:
        metal_non_metal = metal_number / non_metal_number
    else:
        metal_non_metal = 0

    sga = SpacegroupAnalyzer(structure)
    crystal_sys = crystal_system_dict[sga.get_crystal_system()]
    space_group_index = sga.get_space_group_number()
    feature_list += [density_atomic, magnetic_count, average_cohesive_energy, metal_non_metal,
                crystal_sys, space_group_index]

    #27

    distances = []



    coordination = structure.cart_coords
    mean_distance, std_distance = calculate_distance_uniformity(coordination)

    for m in range(len(coordination)):
        for n in range(m + 1, len(coordination)):
            distances.append(np.linalg.norm(coordination[m] - coordination[n]))
    if len(distances) > 3:
        skewness = skew(distances)
        kurt = kurtosis(distances)
    else:
        skewness = kurt = 0

    cutoff = 3.0  # 可根据元素类型调整，若要自动我能帮你做

    coord_nums = []

    for m in range(len(coordination)):
        count = 0
        for n in range(len(coordination)):
            if m != n:
                d = np.linalg.norm(coordination[m] - coordination[n])
                if d < cutoff:
                    count += 1
        coord_nums.append(count)

    # 平均配位数
    avg_coordination = np.mean(coord_nums)

    weights = np.array(list(composition_dict.values()), dtype=float)
    en_values = [Element(sym).X for sym in composition_dict.keys() if Element(sym).X]
    if len(en_values) > 1:
        mean_en = np.average(en_values, weights=weights)
        en_diff = np.sqrt(np.average((np.array(en_values) - mean_en) ** 2, weights=weights))
    else:
        mean_en, en_diff = 0, 0

    # 配位数的分布均匀程度（越小越均匀）
    coordination_uniformity = np.std(coord_nums)
    feature_list += [mean_distance, std_distance,
                skewness, kurt, avg_coordination, coordination_uniformity, mean_en, en_diff]

    #35
    df = pd.DataFrame([feature_list])
    return df

def compute_entropy(label_counter):
    total = sum(label_counter.values())
    if total == 0:
        return 0
    entropy = -sum((count / total) * log2(count / total) for count in label_counter.values() if count > 0)
    return entropy

from pymatgen.core import Structure
from collections import Counter
import numpy as np

R = 8.314

def calc_config_entropy(cif_file, anion_list=['O']):
    """
    计算给定 CIF 结构的构型熵 S_config
    默认把 O 视为阴离子，其他元素为阳离子
    """
    # 读取结构
    structure = Structure.from_file(cif_file)

    # 统计元素个数
    elem_counts = Counter([str(site.specie) for site in structure])
    total_atoms = sum(elem_counts.values())

    # 区分阳离子和阴离子
    cation_counts = {el: cnt for el, cnt in elem_counts.items() if el not in anion_list}
    anion_counts = {el: cnt for el, cnt in elem_counts.items() if el in anion_list}

    # 计算摩尔分数
    cation_total = sum(cation_counts.values())
    anion_total = sum(anion_counts.values())

    cation_x = {el: cnt / cation_total for el, cnt in cation_counts.items()}
    anion_x = {el: cnt / anion_total for el, cnt in anion_counts.items()} if anion_total > 0 else {}

    # 计算构型熵 (只考虑阳离子/阴离子分布的贡献)
    S_cation = -R * sum(x * np.log(x) for x in cation_x.values())
    S_anion = -R * sum(x * np.log(x) for x in anion_x.values()) if anion_x else 0.0

    # 总构型熵
    S_config = S_cation + S_anion



    return S_config

def entrophy_stastic():
    cif_dir = 'oxides_s8'
    a = os.listdir(cif_dir)

    results = []
    all_s = 0
    count = 0

    for i in a:
        cif_path = os.path.join(cif_dir, i)
        S_config = calc_config_entropy(cif_path, anion_list=['O'])

        if S_config is not None:
            results.append({
                "cifname": i,
                "S_config(J/mol·K)": S_config,
                "S_config(R)": S_config / R
            })
            all_s += S_config
            count += 1

    avg_s = all_s / count if count > 0 else 0
    print(f"平均构型熵 = {avg_s:.3f} J/mol·K = {avg_s/R:.3f} R")

    # 保存所有结果到 CSV
    df = pd.DataFrame(results)
    save_file = f"ce_{cif_dir}_{avg_s/R:.3f}R.csv"
    df.to_csv(save_file, index=False)
    print('构型熵结果已保存')

def hea_compare():
    df_a = pd.read_csv('predict_result_5A1B_s.csv')  # 第一列：结构名，最后一列：标签
    df_b = pd.read_csv('5A1B_s_stastic.csv')  # 第一列：结构名，最后一列：构型熵

    # 统一列名（防止列名不同）
    df_a = df_a.rename(columns={
        df_a.columns[0]: 'structure',
        df_a.columns[-1]: 'label'
    })

    df_b = df_b.rename(columns={
        df_b.columns[0]: 'structure',
        df_b.columns[2]: 'config_entropy',
        df_b.columns[3]: 'delta_a'
    })

    # 按结构名合并
    df = pd.merge(df_a[['structure', 'label']],
                  df_b[['structure', 'config_entropy', 'delta_a']],
                  on='structure',
                  how='inner')

    # 按标签顺序 a → b → c 排序
    label_order = ['a', 'b', 'c']
    df['label'] = pd.Categorical(df['label'], categories=label_order, ordered=True)
    df = df.sort_values('label')

    # 保存新文件
    df.to_csv('merged_sorted_5A1B_s.csv', index=False)

def get_site_cation_fractions(structure, site_elements):
    """
    从指定晶体位点元素中提取组成与摩尔分数
    site_elements: list, 该位点允许的元素
    """
    species = [site.specie.symbol for site in structure if site.specie.symbol in site_elements]
    counter = Counter(species)
    total = sum(counter.values())
    if total == 0:
        return [], []
    elements = list(counter.keys())
    fractions = np.array([counter[el] / total for el in elements])
    return elements, fractions

radii_dict = {
    # A site (CN=12)
    "La": 1.36, "Nd": 1.27, "Sr": 1.44, "Ba": 1.61,
    "Ca": 1.34, "Na": 1.39, "K": 1.64, "Rb": 1.72, "Cs": 1.88,

    # B site (CN=6)
    "Fe": 0.645, "Ni": 0.69, "Mn": 0.83, "Co": 0.745,
    "Cr": 0.615, "V": 0.64, "Ti": 0.605,
    "Zr": 0.72, "Hf": 0.71, "Sn": 0.69, "Mg": 0.72
}


def entropy_and_radius_mismatch_statistics(cif_dir, radii_dict,
                                             A_elements=("La", "Nd", "Sr", "Ba", "Ca", "Na", "K", "Rb", "Cs"),
                                             B_elements=("Fe", "Ni", "Mn", "Co", "Cr", "V", "Ti", "Zr", "Hf", "Sn", "Mg")):
    results = []
    all_s = 0
    count = 0

    for fname in os.listdir(cif_dir):
        if not fname.endswith(".cif"):
            continue

        cif_path = os.path.join(cif_dir, fname)

        try:
            structure = Structure.from_file(cif_path)

            # ===== 构型熵 =====
            S_config = calc_config_entropy(cif_path, anion_list=['O'])

            # ===== A 位阳离子 δ =====
            A_sites, A_fractions = get_site_cation_fractions(structure, A_elements)
            if len(A_sites) > 0:
                A_radii = np.array([radii_dict[el] for el in A_sites])
                r_bar_A = np.sum(A_fractions * A_radii)
                delta_A = np.sqrt(np.sum(A_fractions * (1 - A_radii / r_bar_A) ** 2))
            else:
                delta_A, r_bar_A = np.nan, np.nan

            # ===== B 位阳离子 δ =====
            B_sites, B_fractions = get_site_cation_fractions(structure, B_elements)
            if len(B_sites) > 0:
                B_radii = np.array([radii_dict[el] for el in B_sites])
                r_bar_B = np.sum(B_fractions * B_radii)
                delta_B = np.sqrt(np.sum(B_fractions * (1 - B_radii / r_bar_B) ** 2))
            else:
                delta_B, r_bar_B = np.nan, np.nan

        except Exception as e:
            print(f"Skip {fname}: {e}")
            continue

        if S_config is not None:
            results.append({
                "cif_name": fname,
                "S_config_J_per_molK": S_config,
                "S_config_R": S_config / R,
                "delta_A_radius": delta_A,
                "delta_B_radius": delta_B,
                "mean_A_cation_radius": r_bar_A,
                "mean_B_cation_radius": r_bar_B,
                "num_A_cation_species": len(A_sites),
                "num_B_cation_species": len(B_sites)
            })
            all_s += S_config
            count += 1

    avg_s = all_s / count if count > 0 else 0
    avg_s_R = avg_s / R
    print(f"平均构型熵 = {avg_s:.3f} J/mol·K = {avg_s_R:.3f} R")

    df = pd.DataFrame(results)
    output_csv = f"{cif_dir}_stastic.csv"
    df.to_csv(output_csv, index=False)
    print(f"结果已保存至 {output_csv}")

def heo_feature_select():
    '5A1B_s.csv'


if __name__ == '__main__':
    # entropy_and_radius_mismatch_statistics(
    #     '5A1B_r',
    #     radii_dict)
    hea_compare()


