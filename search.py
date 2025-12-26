from mp_api.client import MPRester
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import re
import numpy as np
import pandas as pd
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
#api key
api_key = "GlIXXT78HkkUld1quhdk97sLHjRrcL7W"
#CRYSTAL SYSTEM
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



from scipy.stats import skew, kurtosis


from pymatgen.core.periodic_table import Element

from pymatgen.analysis.local_env import CrystalNN
from scipy.stats import entropy
nn_finder = CrystalNN()
#feature search and covert to csv file
def feature_search():
    output_file = "data_33_unnormalized.csv"
    cohesive_dict = cohesive_energy_loader('element_info/cohesive_energy.csv')

    # === 如果存在已有CSV，则读取已完成的material_id ===
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        processed_ids = set(df_existing.iloc[:, 0])  # 第一列是id
        print(f"✅ 已有 {len(processed_ids)} 个样本，将跳过这些。")
    else:
        processed_ids = set()
        df_existing = pd.DataFrame()

    # === 查询条件 ===
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(
            energy_above_hull=(0, 0.3),
            volume=(10, 1000),
            density=(0.5, 25),
            fields=[
                'material_id','chemsys','composition','composition_reduced',
                'decomposes_to','density_atomic','elements','nelements','nsites',
                'num_magnetic_sites','num_unique_magnetic_sites','structure',
                'symmetry','volume','energy_above_hull'
            ]
        )

    print(f"🔍 总共找到 {len(docs)} 个结构。")

    buffer = []  # 用于暂存写入
    save_every = 20  # 每处理20个样本就写入一次文件

    for idx, i in enumerate(docs):
        id = i.material_id
        if id in processed_ids:
            print(f"⏩ 跳过已处理: {id}")
            continue

        try:
            composition = str(i.composition)
            prop_arr = []
            prop_arr.append(id)
            # === 晶格特征 ===
            a, b, c = i.structure.lattice.abc
            alpha, beta, gamma = i.structure.lattice.angles
            lattice_volume = i.structure.lattice.volume
            lattice_anisotropy = max(a, b, c) / min(a, b, c)

            latt = np.array(i.structure.lattice.matrix)
            U, Sigma, VT = np.linalg.svd(latt)
            max_singular_value = Sigma[0]

            # === 局部环境 ===
            coord_nums = [len(nn_finder.get_nn_info(i.structure, site_index)) for site_index in range(len(i.structure))]
            mean_coord_num = np.mean(coord_nums)
            std_coord_num = np.std(coord_nums)

            # === 组成信息 ===
            atom_number = i.nsites
            elements_number = len(i.elements)
            density_atomic = i.density_atomic
            magnetic_sites = i.num_magnetic_sites if isinstance(i.num_magnetic_sites, int) else 0

            composition_dict = {}
            composition_reduced = str(i.composition_reduced).strip().split(' ')
            for j in composition_reduced:
                letters = ''.join(re.findall(r'[A-Za-z]+', j)[0])
                numbers = ''.join(re.findall(r'\d+', j)[0])
                composition_dict[letters] = float(numbers)

            up = 0
            down = 0
            metal_number = 0
            non_metal_number = 0
            for key, value in composition_dict.items():
                up += cohesive_dict.get(key, 0) * value
                metal_number += int(key in metals)
                non_metal_number += int(key not in metals)
                down += value

            average_cohesive_energy = up / down
            metal_non_metal = metal_number / non_metal_number if non_metal_number else 0

            # === 元素平均性质 ===
            ele_props = ["atomic_mass", 'atomic_radius', 'melting_point', "thermal_conductivity"]
            for p in ele_props:
                vals, weights = [], []
                for sym in composition_dict.keys():
                    if hasattr(Element(sym), p):
                        val = getattr(Element(sym), p)
                        if val is not None:
                            vals.append(val)
                            weights.append(composition_dict[sym])
                if len(vals) > 0:
                    mean_p = np.average(vals, weights=weights)
                    std_p = np.sqrt(np.average((np.array(vals) - mean_p) ** 2, weights=weights))
                else:
                    mean_p, std_p = 0, 0
                prop_arr.extend([mean_p, std_p])

            # === 对称性 ===
            crystal_sys = crystal_system_dict[i.symmetry.crystal_system]
            space_group_index = i.symmetry.number

            # === 原子间距分布 ===
            coordination = i.structure.cart_coords
            mean_distance, std_distance = calculate_distance_uniformity(coordination)

            distances = []
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

            # 配位数的分布均匀程度（越小越均匀）
            coordination_uniformity = np.std(coord_nums)

            # === 电负性 ===
            weights = np.array(list(composition_dict.values()), dtype=float)
            en_values = [Element(sym).X for sym in composition_dict.keys() if Element(sym).X]
            if len(en_values) > 1:
                mean_en = np.average(en_values, weights=weights)
                en_diff = np.sqrt(np.average((np.array(en_values) - mean_en) ** 2, weights=weights))
            else:
                mean_en, en_diff = 0, 0

            # === 汇总特征 ===
            target = i.energy_above_hull
            prop_arr = prop_arr + list(i.structure.lattice.abc) + list(i.structure.lattice.angles)
            prop_arr += [
                lattice_volume, lattice_anisotropy, max_singular_value,
                mean_coord_num, std_coord_num, atom_number, elements_number,
                density_atomic, magnetic_sites, average_cohesive_energy, metal_non_metal,
                crystal_sys, space_group_index, mean_distance, std_distance,
                skewness, kurt, avg_coordination, coordination_uniformity, mean_en, en_diff, target
            ]

            buffer.append(prop_arr)
            print(f"✅ {idx+1}/{len(docs)} {id} done")

            # === 定期写入 ===
            if len(buffer) >= save_every:
                df_new = pd.DataFrame(buffer)
                df_new.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                buffer.clear()

        except Exception as e:
            print(f"⚠️ 出错：{id}, 错误信息: {e}")
            continue

    # === 写入剩余数据 ===
    if buffer:
        df_new = pd.DataFrame(buffer)
        df_new.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

    print("✅ 全部完成并保存。")
import torch
import torch.nn as nn
import torch.optim as optim
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=33, latent_dim=8):
        super(AutoEncoder, self).__init__()
        # Encoder 部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder 部分
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
import torch.nn.functional as F
def latent_regularizers(z,
                        l1_coeff=1e-3,
                        cov_coeff=1e-1,
                        var_coeff=1e-2,
                        var_min=0.01):
    """
    z: [B, D] latent feature
    """
    B, D = z.shape
    logs = {}

    # ---- 1. L1 sparsity ----
    l1 = z.abs().mean()

    # ---- 2. zero-mean ----
    z_centered = z - z.mean(dim=0, keepdim=True)

    # ---- 3. covariance matrix ----
    C = (z_centered.t() @ z_centered) / (B - 1 + 1e-6)

    diag = torch.diag(C)
    off_diag = C - torch.diag(diag)

    # 去相关（惩罚非对角元素）
    cov_pen = (off_diag.pow(2).sum()) / (D * D)

    # ---- 4. variance floor ----
    var_pen = F.relu(var_min - diag).mean()

    reg = l1_coeff * l1 + cov_coeff * cov_pen + var_coeff * var_pen

    logs["l1"] = l1.item()
    logs["cov_pen"] = cov_pen.item()
    logs["var_pen"] = var_pen.item()
    logs["diag_mean"] = diag.mean().item()

    return reg, logs

from torch.utils.data import DataLoader, TensorDataset
def auto_encoder(penal_or_not=True):
    # === 读取与处理数据 ===
    data = pd.read_csv('data_33_unnormalized.csv')
    data_clean = data.dropna()

    X = data_clean.iloc[:, 1:-1].astype(float)
    y = data_clean.iloc[:, -1].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled)

    # === 划分训练/验证/测试集 ===
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y.values, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # tensor 转换
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_full_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # === batch size ===
    batch_size = 512
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size, shuffle=False, drop_last=False)

    # 你要尝试的 latent 维度
    # latent_dims = np.array(range(20, 48))
    latent_dims = [16]


    best_loss = float('inf')
    best_dim = None
    results_summary = []  # 保存结果

    for latent_dim in latent_dims:
        print(f"\n==== 训练 AutoEncoder, latent_dim={latent_dim}, penal={penal_or_not} ====\n")

        model = AutoEncoder(input_dim=X_train_tensor.shape[1], latent_dim=latent_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        num_epochs = 120
        loss_records = []

        for epoch in range(num_epochs):

            # ======= 训练 ==========
            model.train()
            epoch_train_loss = 0

            for (batch_x,) in train_loader:
                reconstructed, z = model(batch_x)

                recon_loss = criterion(reconstructed, batch_x)

                # ===== 惩罚项开关 =====
                if penal_or_not:
                    reg_loss, _ = latent_regularizers(z)
                    loss = recon_loss + reg_loss
                else:
                    loss = recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * batch_x.size(0)

            epoch_train_loss /= len(train_loader.dataset)

            # ======= 验证 ==========
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    rec_val, _ = model(batch_x)
                    val_loss = criterion(rec_val, batch_x)
                    epoch_val_loss += val_loss.item() * batch_x.size(0)
            epoch_val_loss /= len(val_loader.dataset)

            loss_records.append([epoch, epoch_train_loss, epoch_val_loss])

        # === 保存 loss CSV ===
        loss_df = pd.DataFrame(loss_records, columns=["epoch", "train_loss", "val_loss"])
        loss_df.to_csv(f"loss_latent_{latent_dim}_penal{penal_or_not}.csv", index=False)
        print(f"📄 Saved loss_latent_{latent_dim}_penal{penal_or_not}.csv")

        # === 计算测试集误差 ===
        model.eval()
        test_loss_total = 0
        with torch.no_grad():
            for (batch_x,) in test_loader:
                rec_test, _ = model(batch_x)
                test_loss_total += criterion(rec_test, batch_x).item() * batch_x.size(0)
        test_loss = test_loss_total / len(test_loader.dataset)

        # === 计算全数据误差 ===
        with torch.no_grad():
            rec_full, _ = model(X_full_tensor)
            full_loss = criterion(rec_full, X_full_tensor).item()

        # === 保存编码后的特征 ===
        with torch.no_grad():
            latent_all = model(X_full_tensor)[1].numpy()
        encoded_df = pd.DataFrame(latent_all, columns=[f"latent_{i+1}" for i in range(latent_dim)])
        encoded_df["target"] = y.values
        encoded_df.to_csv(f"encoded_latent_{latent_dim}_penal{penal_or_not}.csv", index=False)
        print(f"📄 Saved encoded_latent_{latent_dim}_penal{penal_or_not}.csv")

        # === 写 summary 记录 ===
        results_summary.append([
            latent_dim,
            epoch_val_loss,
            test_loss,
            full_loss
        ])

        # === 更新最优模型 ===
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_dim = latent_dim

        torch.save(model.state_dict(), f"autoencoder_latent_{latent_dim}.pth")

    # === 保存 summary ===
    summary_df = pd.DataFrame(results_summary,
                              columns=["latent_dim", "val_loss", "test_loss", "full_loss"])
    summary_df.to_csv(f"autoencoder_latent_summary_penal{penal_or_not}.csv", index=False)

    print("\n==============================")
    print(f"⭐ 最佳 latent 维度： {best_dim} ，验证误差 = {best_loss:.6f}")
    print("==============================\n")

from scipy.stats import spearmanr


def analyze_feature_relationship(
        old_file='data_33_unnormalized.csv',
        new_file='encoded_latent_16_penalTrue.csv',
        output_corr='new_vs_old.csv',
        output_top5='latent_top5.csv',
        output_old_corr='old_vs_old.csv',
        output_old_top5='old_feature_top5.csv',
        output_latent_corr='new_vs_new.csv'
):
    # === 读取数据 ===
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    old_df = pd.read_csv(old_file).dropna()
    new_df = pd.read_csv(new_file)

    # === 提取旧特征 ===
    X_old = old_df.iloc[:, 1:-1]  # 去掉第一列ID和最后一列y
    X_old = pd.DataFrame(StandardScaler().fit_transform(X_old), columns=X_old.columns)

    # === 提取新特征 ===
    X_new = new_df.iloc[:, :-1]  # 去掉最后一列y/target

    # === 检查样本数量 ===
    if len(X_old) != len(X_new):
        raise ValueError(f"样本数不匹配！旧特征: {len(X_old)}，新特征: {len(X_new)}")

    # === 仅保留数值列 ===
    X_old = X_old.select_dtypes(include=[np.number])
    X_new = X_new.select_dtypes(include=[np.number])

    # ==================================================
    #             Part 1：latent–old 相关性分析
    # ==================================================
    corr_matrix = pd.DataFrame(index=X_new.columns, columns=X_old.columns, dtype=float)
    for new_col in X_new.columns:
        for old_col in X_old.columns:
            corr = np.corrcoef(X_new[new_col], X_old[old_col])[0, 1]
            corr_matrix.loc[new_col, old_col] = corr

    corr_matrix.to_csv(output_corr)
    print(f"✅  new_vs_old相关性矩阵已保存：{output_corr}")

    mean_abs_corr_latent_old = corr_matrix.abs().values.mean()
    print(f"📌 new_vs_old相关性绝对值平均值：{mean_abs_corr_latent_old:.4f}")

    # 每个 latent 选前5重要旧特征
    top_related = {}
    for latent in corr_matrix.index:
        top5 = corr_matrix.loc[latent].abs().sort_values(ascending=False).head(5)
        top_related[latent] = list(top5.index)
    pd.DataFrame.from_dict(top_related, orient='index').to_csv(output_top5)
    print(f"✅ latent对应Top5旧特征已保存：{output_top5}")

    # ==================================================
    #             Part 2：旧特征–旧特征 相关性分析
    # ==================================================
    old_corr = X_old.corr()
    old_corr.to_csv(output_old_corr)
    print(f"📊 old_vs_old：{output_old_corr}")



    # 每个旧特征最相关Top5旧特征（排除自己）
    old_top = {}
    for feature in old_corr.columns:
        top5 = old_corr[feature].abs().drop(feature).sort_values(ascending=False).head(5)
        old_top[feature] = list(top5.index)
    pd.DataFrame.from_dict(old_top, orient='index').to_csv(output_old_top5)
    print(f"📌 旧特征内部Top5相关特征已保存：{output_old_top5}")

    mean_abs_corr_old_old = old_corr.abs().values.mean()
    print(f"📌 old–old 相关性绝对值平均值：{mean_abs_corr_old_old:.4f}")

    # ==================================================
    #             Part 3：latent–latent 内部相关性分析
    # ==================================================
    latent_corr = X_new.corr()
    latent_corr.to_csv(output_latent_corr)
    print(f"📊 latent_vs_latent：{output_latent_corr}")

    mean_abs_corr_latent_latent = latent_corr.abs().values.mean()
    print(f"📌 latent–latent 相关性绝对值平均值：{mean_abs_corr_latent_latent:.4f}")

    print("\n=== 全部分析完成 ===")
    print(f"latent–old 相关性矩阵维度: {corr_matrix.shape}")
    print(f"old–old   相关性矩阵维度: {old_corr.shape}")
    print(f"latent–latent 相关性矩阵维度: {latent_corr.shape}")
    print("💡 建议同时检查 latent_top5 与 old_top5，以辅助物理解释。")


feature_dict = {
    "1": "mean_atomic_mass",
    "2": "std_atomic_mass",
    "3": "mean_atomic_radius",
    "4": "std_atomic_radius",
    "5": "mean_melting_point",
    "6": "std_melting_point",
    "7": "mean_thermal_conductivity",
    "8": "std_thermal_conductivity",

    "9": "lattice_a",
    "10": "lattice_b",
    "11": "lattice_c",
    "12": "lattice_alpha",
    "13": "lattice_beta",
    "14": "lattice_gamma",

    "15": "lattice_volume",
    "16": "lattice_anisotropy",
    "17": "max_singular_value",

    "18": "mean_coord_num",
    "19": "std_coord_num",

    "20": "atom_number",
    "21": "elements_number",
    "22": "density_atomic",
    "23": "magnetic_sites",

    "24": "average_cohesive_energy",
    "25": "metal_non_metal",

    "26": "crystal_system",
    "27": "space_group_index",

    "28": "mean_interatomic_distance",
    "29": "std_interatomic_distance",
    "30": "distance_skewness",
    "31": "distance_kurtosis",

    "32": "avg_coordination",
    "33": "coordination_uniformity",

    "34": "mean_electronegativity",
    "35": "electronegativity_difference",

    "36": "energy_above_hull"
}




def tsne(auto_cluster, file_path, number_of_features, color, threshold):


    df = pd.read_csv(file_path)
    y = df.iloc[:, -1].values

    conditions = [
        (y == 0),
        (y >0) & (y <= threshold),
        (y >= threshold)
    ]

    choices = ['a', 'b', 'c']

    labels = np.select(conditions, choices)
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


    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
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
        optimal_k = 6
    print(f"Clustering data into {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, max_iter=500, tol=1e-6)
    cluster_labels = kmeans.fit_predict(tsne_results)

    from sklearn.metrics import calinski_harabasz_score
    #ch_index
    if len(np.unique(cluster_labels)) > 1:
        ch_score = calinski_harabasz_score(tsne_results, cluster_labels)
        silhouette_avg = silhouette_score(tsne_results, cluster_labels)
        print(f"Calinski-Harabasz Score: {ch_score:.4f}")
        print(f"Average Silhouette Score: {silhouette_avg:.4f}")
    else:
        ch_score = np.nan
        silhouette_avg = np.nan
        print("⚠️ 聚类结果只有一个簇，无法计算CH/Silhouette分数。")

    #silhouette score
    # silhouette_avg = silhouette_score(df, labels_sampled)
    # print(f"Average Silhouette Score: {silhouette_avg:.4f}")

    ch = calinski_harabasz_score(feature_data, labels)
    sil = silhouette_score(feature_data, labels)


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
    plt.savefig("tsne_of_{}_features_with {} colors of {}.jpg".format(number_of_features, color, os.path.basename(file_path)), dpi=600)
    # mean value
    df['Cluster'] = cluster_labels
    cluster_means = df.groupby('Cluster').mean()
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    print("Cluster Means:")
    print(cluster_means.to_string())
    return ch_score, silhouette_avg, ch, sil


def labeled_ch(file_path, threshold):
    df = pd.read_csv(file_path)
    y = df.iloc[:, -1].values

    conditions = [
        (y == 0),
        (y >0) & (y <= threshold),
        (y >= threshold)
    ]

    choices = ['a', 'b', 'c']

    labels = np.select(conditions, choices)
    feature_data = df.iloc[:, :16].values
    from sklearn.metrics import calinski_harabasz_score
    ch = calinski_harabasz_score(feature_data, labels)
    sil = silhouette_score(feature_data, labels)
    return ch, sil


def pca(file_path, components_num = None):
    df = pd.read_csv(file_path)
    feature_data = df.iloc[:, :-1].values
    #数据标准化
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)
    # 将数据转换为DataFrame，设置列名
    df = pd.DataFrame(feature_data_scaled)
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
        # file_path = 'encoded_latent_16_penalTrue.csv'
        # a, b = tsne('auto', file_path, 16, 's', 0.1)


    # results = []
    # for value in np.arange(0.01, 0.30, 0.01):
    #     file_path = 'encoded_latent_16_penalTrue.csv'
    #     a, b = labeled_ch( file_path, value)
    #     result = {'ch_index' : a,
    #             'Average Silhouette Score': b}
    #     results.append(result)
    # pd.DataFrame(results).to_csv('labeled_ch index and silhouette score.csv', index=False)



    # tsne(False, 'data_encoded_latent16.csv', 16, '3')
    auto_encoder()
    # tsne(False, 'encoded_latent_16_penalTrue.csv', 16, '3')
    # analyze_feature_relationship()
    # search_for_application()

