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
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
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

from utilies import cohesive_energy_loader, calculate_distance_uniformity

#api key
api_key = "GlIXXT78HkkUld1quhdk97sLHjRrcL7W"
#CRYSTAL SYSTEM
crystal_system_dict = {
    'Cubic': 1,
    'Orthorhombic':2,
    'Hexagonal':3,
    'Tetragonal':4,
    'Trigonal':5,
    'Triclinic':6,
    'Monoclinic':7,
}
#SPCAE GROUP
space_group_dict = {
    'Pm-3m':1,
    'Cmcm':2,
    'Fd-3m':3,
    'I4/mcm':4,
    'P-6m2':5,
    'C2/m':6,
    'P1':7,
    'Pnnm':8,
    'P4/mmm':9,
    'R3m':10,

}
#ATOM MASS
mass_dict = {
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.01,
    "N": 14.01,
    "O": 16.00,
    "F": 19.00,
    "Ne": 20.18,
    "Na": 22.99,
    "Mg": 24.31,
    "Al": 26.98,
    "Si": 28.09,
    "P": 30.97,
    "S": 32.07,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.10,
    "Ca": 40.08,
    "Sc": 44.96,
    "Ti": 47.87,
    "V": 50.94,
    "Cr": 52.00,
    "Mn": 54.94,
    "Fe": 55.85,
    "Co": 58.93,
    "Ni": 58.69,
    "Cu": 63.55,
    "Zn": 65.38,
    "Ga": 69.72,
    "Ge": 72.63,
    "As": 74.92,
    "Se": 78.97,
    "Br": 79.90,
    "Kr": 83.80,
    "Rb": 85.47,
    "Sr": 87.62,
    "Y": 88.91,
    "Zr": 91.22,
    "Nb": 92.91,
    "Mo": 95.95,
    "Tc": 98.00,
    "Ru": 101.1,
    "Rh": 102.9,
    "Pd": 106.4,
    "Ag": 107.9,
    "Cd": 112.4,
    "In": 114.8,
    "Sn": 118.7,
    "Sb": 121.8,
    "Te": 127.6,
    "I": 126.9,
    "Xe": 131.3,
    "Cs": 132.9,
    "Ba": 137.3,
    "La": 138.9,
    "Ce": 140.1,
    "Pr": 140.9,
    "Nd": 144.2,
    "Pm": 145.0,
    "Sm": 150.4,
    "Eu": 152.0,
    "Gd": 157.3,
    "Tb": 158.9,
    "Dy": 162.5,
    "Ho": 164.9,
    "Er": 167.3,
    "Tm": 168.9,
    "Yb": 173.0,
    "Lu": 175.0,
    "Hf": 178.5,
    "Ta": 180.9,
    "W": 183.8,
    "Re": 186.2,
    "Os": 190.2,
    "Ir": 192.2,
    "Pt": 195.1,
    "Au": 197.0,
    "Hg": 200.6,
    "Tl": 204.4,
    "Pb": 207.2,
    "Bi": 208.98,
    "Po": 209,
    "At": 210,
    "Rn": 222,
    "Fr": 223,
    "Ra": 226,
    "Ac": 227,
    "Th": 232.0,
    "Pa": 231.0,
    "U": 238.0,
    "Np": 237,
    "Pu": 244,
    "Am": 243,
    "Cm": 247,
    "Bk": 247,
    "Cf": 251,
    "Es": 252,
    "Fm": 257,
    "Md": 258,
    "No": 259,
    "Lr": 262,
    "Rf": 267,
    "Db": 270,
    "Sg": 271,
    "Bh": 270,
    "Hs": 277,
    "Mt": 278,
    "Ds": 281,
    "Rg": 282,
    "Cn": 285,
    "Nh": 286,
    "Fl": 289,
    "Mc": 290,
    "Lv": 293,
    "Ts": 294,
    "Og": 294
}
#METAL ELEMENTS
metals = ["Al", "As", "Au", "Ba", "Be", "Ca", "Cd", "Co", "Cr", "Cu", "Fe", "Hg", "K", "Li", "Mg", "Mn", "Na", "Ni", "Pb", "Pt", "K", "Li", "Na", "Ti", "Zn", "Zr"]
#根据material_id寻找结构
def structure_finder():
    id_list = []
    with open ('stable_3_materials.txt', 'r') as file:
        lines = file.readlines()
    for i in lines:
        id_list.append(i.split(',')[0])

    with MPRester(api_key) as mpr:
        for i in id_list:
            if '{}.cif'.format(i) in os.listdir('stable_3_structures'):

                print('{} file exists'.format(i))

            else:


                    structure = mpr.get_structure_by_material_id(i)
                    c = CifWriter(structure)
                    file_name = os.path.join('unstable_2_structures', '{}.cif'.format(i))
                    c.write_file(file_name)
                    print(id_list.index(i), len(id_list))
#按照id逐个寻找fomation_energy,比较慢，已经被formation_energy_finder_new()替代
def formation_energy_finder():
    id_list = []
    with open ('data/unstable/unstable_1_materials', 'r') as file:
        lines = file.readlines()
    for i in lines:
        id_list.append(i.split(',')[0])

    with open('unstable_1_formation_energy', 'r') as file:
        lines = file.readlines()
        name = [x.split(',')[0] for x in lines]
        file.close()



    with open('unstable_1_formation_energy', 'a') as file:


        with MPRester(api_key) as mpr:
            for i in id_list:
                print(i, id_list.index(i), len(id_list))
                if i not in name:
                    print('looking for formation energy of {}'.format(i))

                    #materialsproject上的energy_above_hull和formation_energy单位都是ev/atom
                    docs = mpr.materials.summary.search(material_ids=[i], fields=['formation_energy_per_atom','composition'])
                    for_energy_per_atom = docs[0].formation_energy_per_atom

                    composition = str(docs[0].composition)
                    atom_numbers = re.findall(r'\d+', composition)
                    total_atom_number = sum(int(num) for num in atom_numbers)

                    line = i+','+str(for_energy_per_atom)
                    file.write(f'{line}\n')
                else:
                    print('already exists')
#根据docs寻找fomation energy,更快但是这些材料必须满足一定的共性
def formation_energy_finder_new():
    is_stable = True
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(is_stable=is_stable, chemsys='*-*-*-*-*-*-*-*',
                                  fields=['material_id', 'formation_energy_per_atom'])  # 使用mpr.materials.thermo.search也是可以的。效果一致
        # docs = mpr.summary.search(is_stable=False, chemsys='Cu-*-*',
        #                           fields=['material_id', 'energy_above_hull', 'formula', 'composition',
        #                                   'decomposes_to'])

    with open('stable_8_formation_energy', 'w') as file:

        for x in docs:
            material_id = x.material_id
            formation_energy = x.formation_energy_per_atom

            line = material_id + ',' + str(formation_energy)



            file.write(f'{line}\n')

    return
#按组成材料的元素种类来寻找性质和e_hull
def import_tool_new():



    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(chemsys = 'Fe-*',
                                         fields=['material_id', 'energy_above_hull','symmetry','composition',
                                                 'n' ,'density' ,'structure', 'nelements', 'elements'])  #使用mpr.materials.thermo.search也是可以的。效果一致
        # docs = mpr.summary.search(is_stable=False, chemsys='Cu-*-*',
        #                           fields=['material_id', 'energy_above_hull', 'formula', 'composition',
        #                                   'decomposes_to'])


    with (open('test_new.txt', 'w')as file):

        for x in docs:
            material_id = x.material_id
            e_hull = x.energy_above_hull #此处没有x.formula这个方法

            #
            # decompose_to = x.decomposes_to
            sym_1 = x.symmetry.crystal_system #cubic, orthorhombic
            sym_1_index = crystal_system_dict[sym_1]
            sym_2 = x.symmetry.symbol #P6_3/mmc
            if sym_2 in space_group_dict.keys():
                sym_2_index = space_group_dict[sym_2]
            else:
                space_group_dict[sym_2] = int(len(space_group_dict)+1)
                sym_2_index = space_group_dict[sym_2]
            # sym_3 = x.symmetry.number #unknown

            density = x.density

            a = x.structure.atomic_numbers #a is a tuple

            #这两项用x.structure.atomic_numbers就可以了
            # c = x.nelements  #8,2
            # d = x.elements #[Fe, Sb]

            a = list(a)
            a = [x for x in a if x != 26]
            atomic_number = a[0]
            number_of_atom = len(a)


            line = material_id +','+ str(e_hull) + ',' + str(sym_1_index) + ',' + str(sym_2_index) +  ',' + str(density)
            line = line +','+str(atomic_number) + ','+str(number_of_atom)







            file.write(f'{line}\n')
        print(space_group_dict)

    return
#测试各个性质是什么意思并且找出有用的，现在这个程序已经没用了
def property_test():
    is_stable = False
    cohesive_dict = cohesive_energy_loader('element_info/cohesive_energy.csv')
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(chemsys = '*-*-*-*-*-*-*',
                                  is_stable = is_stable,
                                  fields = ['material_id','chemsys','composition','composition_reduced',
                                            'decomposes_to','density','density_atomic',
                                            'elements','nelements','nsites',
                                            'num_magnetic_sites','num_unique_magnetic_sites','structure',
                                            'symmetry','volume'])

    methods = [attr for attr in dir(docs[0]) if not attr.startswith('__')]
    print(methods)

    a = docs[0].structure
    print(dir(a))


    print('xxxxxxxxxxx')
    for i in docs:
        coordination = i.structure.cart_coords
        mean_distance, std_distance = calculate_distance_uniformity(coordination)
        print(mean_distance,std_distance)

        L = i.structure.lattice.matrix
        print("L 类型:", type(L))
        L = np.array(L)
        print("L 维度:", L.shape)
        U, Sigma, VT = np.linalg.svd(L)
        max_singular_value = Sigma[0]
        print("最大奇异值:", max_singular_value)

#feature search and covert to csv file
def feature_search():
    cohesive_dict = cohesive_energy_loader('element_info/cohesive_energy.csv')
    with MPRester(api_key) as mpr:  #(0,0.0000001),（0.0000001,3.0000001），（3.0000001，）
        docs = mpr.summary.search( band_gap = (3.0000001,15),fields = ['material_id','chemsys','composition','composition_reduced',
                                            'decomposes_to','density_atomic',
                                            'elements','nelements','nsites',
                                            'num_magnetic_sites','num_unique_magnetic_sites','structure',
                                            'symmetry','volume','energy_above_hull','structure'])
    data = None
    for i in docs:
        id = i.material_id
        composition = str(i.composition)
        prop_arr  = []
        #总原子数
        atom_number = i.nsites
        #原子种类
        elements_number = len(i.elements)  #i.elements的类型是list
        #atomic density
        density_atomic = i.density_atomic
        #磁性原子数
        magnetic_sites = i.num_magnetic_sites
        if type(magnetic_sites) != int:
            continue
        #对称性
        crystal_sys = crystal_system_dict[i.symmetry.crystal_system]
        space_group_index = i.symmetry.number
        #cohesive energy, mass, metal or not
        composition_dict = {}
        composition_reduced = str(i.composition_reduced).strip().split(' ')  #composition的形式是Ag2 H8 C2 S2 N5 Cl1 O3
        for j in composition_reduced:
            letters = ''.join(re.findall(r'[A-Za-z]+', j)[0])
            numbers = ''.join(re.findall(r'\d+', j)[0])
            composition_dict[letters] = float(numbers)
        all_mass = 0
        up = 0
        down = 0
        metal_number = 0
        non_metal_number = 0
        for key,value in composition_dict.items():
            if key in cohesive_dict.keys():
                up += cohesive_dict[key]*value
            else:
                up += 0
            all_mass +=  mass_dict[key]*value
            if key in metals:
                metal_number += 1
            else:
                non_metal_number += 1
            down += value
        # cohesive energy
        average_cohesive_energy = up/down
        # 质量
        mass_average = all_mass/down
        # metal or not
        if non_metal_number != 0:
            metal_non_metal = metal_number/non_metal_number
        else:
            metal_non_metal = 0
        #maximum singular value of the lattice
        latt = np.array(i.structure.lattice.matrix)
        U, Sigma, VT = np.linalg.svd(latt)
        max_singular_value = Sigma[0]
        #mean_distance, std_distance
        coordination = i.structure.cart_coords
        mean_distance, std_distance = calculate_distance_uniformity(coordination)


        # feature
        prop_arr.append(id)
        prop_arr.append(composition)
        prop_arr.append(atom_number)
        prop_arr.append(elements_number)
        prop_arr.append(density_atomic)
        prop_arr.append(magnetic_sites)
        prop_arr.append(crystal_sys)
        prop_arr.append(space_group_index)
        prop_arr.append(average_cohesive_energy)
        prop_arr.append(mass_average)
        prop_arr.append(metal_non_metal)
        prop_arr.append(max_singular_value)
        prop_arr.append(mean_distance)
        prop_arr.append(std_distance)
        #target
        target = i.energy_above_hull
        prop_arr.append(target)
        #reshape
        prop_arr = np.array(prop_arr)
        length = len(prop_arr)
        prop_arr = prop_arr.reshape(1, length)

        if data is None:
            data = prop_arr
        else:
            data = np.concatenate((data, prop_arr), axis = 0)
        print(docs.index(i), len(docs))
    df = pd.DataFrame(data, columns=['id', 'composition', 'Natom', 'Nelem', 'D', 'Nmag', 'CS','SG', 'E', 'M', 'MR','MSVL','dmean','dstd','ehull'])
    # 使用 DataFrame.to_csv 将 DataFrame 写入 CSV 文件
    file_name = input('please enter the filename')
    df.to_csv(file_name, index=False)
    feature_data = data[:,2:-1].astype(float)
    y = data[:,-1:]
    nan_positions = np.argwhere(np.isnan(feature_data))
    print(nan_positions)


def data_processing(criterion):
    file_path = 'data_12'
    df_all = pd.DataFrame()

    for i in os.listdir(file_path):
        path = os.path.join(file_path,i)
        df = pd.read_csv(path)
        df_new = df.iloc[:, 2:]
        df_new = df_new.dropna()
        if criterion != 0:
            df_new.iloc[:, -1] = df_new.iloc[:, -1].apply(
            lambda x: 'a' if float(x) == 0 else 'b' if 0 < float(x) < criterion else 'c')
        else:
            df_new.iloc[:, -1] = df_new.iloc[:, -1].apply(
            lambda x: 'a' if float(x) == 0 else 'c')
        df_all = pd.concat([df_all, df_new], ignore_index=True)
    criterion_str = "{:.2f}".format(criterion)
    df_all.to_csv(criterion_str +'.csv', index=False)
    #Do some statistical analysis
    feature_data = df_all.iloc[:, :-1].values
    mean = feature_data.mean(axis = 0)
    std_dev = feature_data.std(axis = 0)
    print('mean: {} /n, std_dev: {}'.format(mean,std_dev))
    counts = df_all.iloc[:, -1].value_counts() ## Count the occurrences of each label in the last column
    print(counts)


def tsne(auto_cluster, file_path, number_of_features, color):
    df = pd.read_csv(file_path)
    labels = df.iloc[:, -1].values
    feature_data = df.iloc[:, :number_of_features].values
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)
    feature_data = feature_data_scaled
    np.random.seed(42)
    if feature_data.shape[0] > 10000:
        sampled_indices = np.random.choice(feature_data.shape[0], 10000, replace=False)
        feature_data_sampled = feature_data[sampled_indices]
        labels_sampled = labels[sampled_indices]
    else:
        feature_data_sampled = feature_data
        labels_sampled = labels

    df = pd.DataFrame(feature_data_sampled, columns = df.columns[:number_of_features])
    # performing tsne
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity = 50)
    tsne_results = tsne.fit_transform(df)

    tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE Component 1', 't-SNE Component 2'])
    tsne_df['Label'] = labels_sampled
    if auto_cluster == 'auto':
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
    else:
        optimal_k = 3
    print(f"Clustering data into {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, max_iter=500, tol=1e-6)
    cluster_labels = kmeans.fit_predict(tsne_results)

    tsne_df['Cluster'] = cluster_labels
    #cluster center
    cluster_centers = kmeans.cluster_centers_
    #plot detail
    label_color_map = {'a': 'red', 'b': 'blue', 'c': 'green'}
    fig, ax = plt.subplots()
    ax.spines['top'].set_linewidth(1)  # 设置上边框的宽度
    ax.spines['right'].set_linewidth(1)  # 设置右边框的宽度
    ax.spines['bottom'].set_linewidth(1)  # 设置下边框的宽度
    ax.spines['left'].set_linewidth(1)
    plt.figure(figsize=(9, 6))

    plt.rcParams['font.family'] = 'Arial'
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(np.unique(cluster_labels))))

    if color == '3':
        for label in np.unique(labels_sampled):
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

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig("tsne_of_{}_features_with {} colors.jpg".format(number_of_features, color), dpi=600)
    # mean value
    df['Cluster'] = cluster_labels
    cluster_means = df.groupby('Cluster').mean()
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    print("Cluster Means:")
    print(cluster_means.to_string())


def pca(components_num = None):
    df = pd.read_csv('data/features/combined_features_100meV.csv')
    feature_data = df.iloc[:, :-1].values
    #数据标准化
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)
    # 将数据转换为DataFrame，设置列名
    df = pd.DataFrame(feature_data_scaled, columns=['Atom number', 'Element number', 'Density', 'Magnetic atom number',
                                             'Crystal system', 'Space group', 'Average cohesive energy','Average mass','Metal or not'])
    pca = PCA(random_state=42)

    # 如果未指定主成分数量，则根据累计解释方差选择
    if components_num is None:
        pca_results = pca.fit_transform(feature_data_scaled)
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_explained_variance >= 0.9) + 1
    else:
        pca = PCA(n_components=components_num, random_state=42)
        pca_results = pca.fit_transform(feature_data_scaled)
        n_components = components_num




    # 打印解释方差
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print("Explained Variance Ratio:", explained_variance_ratio)
    print("Cumulative Explained Variance:", cumulative_explained_variance)
    # 计算主成分载荷
    components = pca.components_
    print("Principal Component Loadings:")
    print(components)
    # 创建 PCA 结果的 DataFrame
    pca_df = pd.DataFrame(pca_results, columns=[f'PCA Component {i+1}' for i in range(pca_results.shape[1])])
    # 绘制解释方差的Scree Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linewidth=2)
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(False)
    plt.show()
    # 绘制主成分载荷图
    plt.figure(figsize=(10, 6))
    for i in range(components.shape[0]):
        plt.plot(df.columns, components[i], label=f'PC{i+1}', linewidth=2)
    plt.title('Principal Component Loadings')
    plt.xlabel('Original Variables')
    plt.ylabel('Loadings')
    plt.legend()
    plt.grid(False)
    plt.show()
    # 绘制 PCA 结果的散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PCA Component 1'], pca_df['PCA Component 2'], s=100, alpha=0.7)
    colors = plt.cm.get_cmap('tab10', n_components)  # 可以根据需要设置更多颜色
    for i, (length, vector) in enumerate(zip(pca.explained_variance_, pca.components_)):
        v = vector * 3 * np.sqrt(length)  # 调整向量长度，以便更好地显示
        plt.quiver(pca.mean_[0], pca.mean_[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, color=colors(i))
    plt.title('PCA Visualization with Principal Components')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()


def corr_matrix():

    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 自动调节显示宽度
    pd.set_option('display.max_colwidth', None)
    #读取数据
    df = pd.read_csv('data_12/0.10.csv')
    feature_data = df.iloc[:, :-1].values
    feature_names = df.columns[:-1]
    # 将数据转换为DataFrame，设置列名
    df = pd.DataFrame(feature_data, columns=feature_names)
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    corr_matrix = df.corr()
    print(corr_matrix)



    ax = sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True)

    #设置colorbar字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14, width=2)
    cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), weight='bold')

    plt.xticks(fontsize=10, rotation=45, weight='bold')
    plt.yticks(fontsize=10, rotation=45 , weight='bold')
    plt.tight_layout()
    plt.savefig("feature_coorelation_heatmap.jpg", dpi=600)  
    plt.show()

def search_for_application():
    cohesive_dict = cohesive_energy_loader('element_info/cohesive_energy.csv')
    with MPRester(api_key) as mpr:  #(0,0.0000001),（0.0000001,3.0000001），（3.0000001，）
        docs = mpr.summary.search( band_gap = (0,15),formula='**Br3',
                                   fields = ['material_id','chemsys','composition','composition_reduced',
                                            'decomposes_to','density_atomic',
                                            'elements','nelements','nsites',
                                            'num_magnetic_sites','num_unique_magnetic_sites','structure',
                                            'symmetry','volume','energy_above_hull','structure'])
    data = None
    index = 0
    for i in docs:
        index += 1
        id = i.material_id
        composition = str(i.composition)
        prop_arr  = []
        #总原子数
        atom_number = i.nsites
        #原子种类
        elements_number = len(i.elements)  #i.elements的类型是list
        #atomic density
        density_atomic = i.density_atomic
        #磁性原子数
        magnetic_sites = i.num_magnetic_sites
        if type(magnetic_sites) != int:
            continue
        #对称性
        crystal_sys = crystal_system_dict[i.symmetry.crystal_system]
        space_group_index = i.symmetry.number
        #cohesive energy, mass, metal or not
        composition_dict = {}
        composition_reduced = str(i.composition_reduced).strip().split(' ')  #composition的形式是Ag2 H8 C2 S2 N5 Cl1 O3
        for j in composition_reduced:
            letters = ''.join(re.findall(r'[A-Za-z]+', j)[0])
            numbers = ''.join(re.findall(r'\d+', j)[0])
            composition_dict[letters] = float(numbers)
        all_mass = 0
        up = 0
        down = 0
        metal_number = 0
        non_metal_number = 0
        for key,value in composition_dict.items():
            if key in cohesive_dict.keys():
                up += cohesive_dict[key]*value
            else:
                up += 0
            all_mass +=  mass_dict[key]*value
            if key in metals:
                metal_number += 1
            else:
                non_metal_number += 1
            down += value
        # cohesive energy
        average_cohesive_energy = up/down
        # 质量
        mass_average = all_mass/down
        # metal or not
        if non_metal_number != 0:
            metal_non_metal = metal_number/non_metal_number
        else:
            metal_non_metal = 0
        #maximum singular value of the lattice
        latt = np.array(i.structure.lattice.matrix)
        U, Sigma, VT = np.linalg.svd(latt)
        max_singular_value = Sigma[0]
        #mean_distance, std_distance
        coordination = i.structure.cart_coords
        mean_distance, std_distance = calculate_distance_uniformity(coordination)


        # feature
        prop_arr.append(id)
        prop_arr.append(composition)
        prop_arr.append(atom_number)
        prop_arr.append(elements_number)
        prop_arr.append(density_atomic)
        prop_arr.append(magnetic_sites)
        prop_arr.append(crystal_sys)
        prop_arr.append(space_group_index)
        prop_arr.append(average_cohesive_energy)
        prop_arr.append(mass_average)
        prop_arr.append(metal_non_metal)
        prop_arr.append(max_singular_value)
        prop_arr.append(mean_distance)
        prop_arr.append(std_distance)
        #target
        target = i.energy_above_hull
        prop_arr.append(target)
        #reshape
        prop_arr = np.array(prop_arr)
        length = len(prop_arr)
        prop_arr = prop_arr.reshape(1, length)

        if data is None:
            data = prop_arr
        else:
            data = np.concatenate((data, prop_arr), axis = 0)
        print(docs.index(i), len(docs))
    df = pd.DataFrame(data, columns=['id', 'composition', 'Natom', 'Nelem', 'D', 'Nmag', 'CS','SG', 'E', 'M', 'MR','MSVL','dmean','dstd','ehull'])
    # 使用 DataFrame.to_csv 将 DataFrame 写入 CSV 文件
    print(index)
    file_name = input('please enter the filename')
    df.to_csv(file_name, index=False)
    feature_data = data[:,2:-1].astype(float)
    y = data[:,-1:]
    nan_positions = np.argwhere(np.isnan(feature_data))
    print(nan_positions)


if __name__ == '__main__':
    # for value in np.arange(0, 0.31, 0.01):
    #     data_processing(value)
    # for i in range(2,11):
    tsne('auto', 'data_12/0.10.csv', 10, 'all')

    # search_for_application()
