from pymatgen.core import Structure
import itertools
import os
import random

output_dir = "5A1B_p"
os.makedirs(output_dir, exist_ok=True)

cif_file = "LaCoO3_221.cif"  #P
# cif_file = "MgAl2O4.cif"  #S
# cif_file = "MgVO_222.cif"  #R
# cif_file = "CeVO2_222.cif"  #F

structure = Structure.from_file(cif_file)

La_replacements = ["Sr", "Ba", "Ca", "Nd", "Na", "K", "Rb", "Cs", 'La']
Co_replacements = ["Fe", "Ni", "Mn", "Ti", "V", "Cr",'Co', 'Mg', 'Sn','Zr', 'Ti', 'Hf']


def random_distribute_sites(sites, elements):
    """
    将 sites 个位置随机分配给 elements 列表中的元素
    返回一个 list，表示每个 site 的具体原子类型
    """
    assignment = []
    for _ in range(sites):
        assignment.append(random.choice(elements))  # 随机选一个元素
    random.shuffle(assignment)
    return assignment

def random_equal_distribution(sites, elements):
    """
    将 sites 个位置等摩尔分配给 elements 中的元素
    若不能整除，则随机分配剩余部分
    返回一个长度为 sites 的列表
    """
    n_elements = len(elements)
    base = sites // n_elements  # 每个元素的基础数量
    remainder = sites % n_elements  # 不能整除的剩余个数

    assignment = []
    # 先平均分配
    for el in elements:
        assignment.extend([el] * base)

    # 再把剩余的随机分给部分元素
    if remainder > 0:
        extra_elements = random.sample(elements, remainder)
        assignment.extend(extra_elements)

    random.shuffle(assignment)
    return assignment

from collections import Counter
from pymatgen.core.lattice import Lattice
import numpy as np
def generate_random_structures(structure, A_replacements, B_replacements, output_dir, n_structures=1000,
                               element_num = 5,equal_distribution = True, lattice_distortion = 0.1, vacancy_prob=0.05,
                               angle_distortion = 10):

    i = 0
    A_site_elements = {"La", "Mg", "Ce"}
    B_site_elements = {"Co", "Al", "Ti", "V"}
    while i < n_structures:
        new_struct = structure.copy()

        # 获取 La / Co 的 site index
        La_sites = [idx for idx, sp in enumerate(new_struct.species) if sp.symbol in A_site_elements]
        Co_sites = [idx for idx, sp in enumerate(new_struct.species) if sp.symbol in B_site_elements]

        # 随机选一些元素替换 A 位

        if equal_distribution == True:
            A_elements = random.sample(A_replacements, 5)
            A_assignment = random_equal_distribution(len(La_sites), A_elements)
        else:
            A_elements = random.sample(A_replacements, random.randint(1, len(A_replacements)))
            A_assignment = random_distribute_sites(len(La_sites), A_elements)
        for idx, el in zip(La_sites, A_assignment):
            new_struct[idx] = el

        # 随机选一些元素替换 B 位

        if equal_distribution == True:
            B_elements = random.sample(B_replacements, 1)
            B_assignment = random_equal_distribution(len(Co_sites), B_elements)
        else:
            B_elements = random.sample(B_replacements, random.randint(1, len(B_replacements)))
            B_assignment = random_distribute_sites(len(Co_sites), B_elements)
        for idx, el in zip(Co_sites, B_assignment):
            new_struct[idx] = el

        if lattice_distortion > 0:
            # distort lattice
            lat = new_struct.lattice.matrix.copy()
            scale_factors = 1 + lattice_distortion * (np.random.random(3) * 2 - 1)
            lat = lat * scale_factors.reshape(3, 1)
            new_struct.lattice = Lattice(lat)

            # atomic jitter
            max_jitter = lattice_distortion / 2
            for at in range(len(new_struct)):
                disp = max_jitter * (np.random.random(3) * 2 - 1)
                new_struct.translate_sites([at], disp, frac_coords=False)




        if vacancy_prob > 0:
            # 删除概率型空位（适合模拟高熵氧化物常见的O空位和A/B位缺陷）
            remove_indices = []
            for idx in range(len(new_struct)):
                if np.random.rand() < vacancy_prob:
                    remove_indices.append(idx)
            if len(remove_indices) > 0:
                new_struct.remove_sites(remove_indices)

        if angle_distortion > 0:
            lat = new_struct.lattice.copy()
            a, b, c = lat.lengths
            alpha, beta, gamma = lat.angles

            # 在每个角度上加入 ± angle_distortion 范围内的随机扰动
            new_alpha = alpha + np.random.uniform(-angle_distortion, angle_distortion)
            new_beta = beta + np.random.uniform(-angle_distortion, angle_distortion)
            new_gamma = gamma + np.random.uniform(-angle_distortion, angle_distortion)

            # 重新构建晶格
            new_lat = Lattice.from_parameters(
                a, b, c,
                new_alpha, new_beta, new_gamma
            )
            new_struct.lattice = new_lat

        if equal_distribution == True:

            if len(A_elements)+len(B_elements) >= element_num and len(A_elements) <= 5 and len(B_elements) <= 5:

            # 保存文件
                A_counts = Counter(A_assignment)
                B_counts = Counter(B_assignment)
                A_ratios = {el: round(A_counts[el] / len(La_sites), 2) for el in A_counts}
                B_ratios = {el: round(B_counts[el] / len(Co_sites), 2) for el in B_counts}

            # 构造比例字符串，如 "La0.25-Sr0.75"
                A_ratio_str = "-".join([f"{el}{A_ratios[el]:.2f}" for el in sorted(A_ratios)])
                B_ratio_str = "-".join([f"{el}{B_ratios[el]:.2f}" for el in sorted(B_ratios)])

            # 文件名中加入比例信息
                filename = f"{i + 1}({A_ratio_str})({B_ratio_str}).cif"
                filepath = os.path.join(output_dir, filename)
                new_struct.to(filename=filepath)
                i += 1
        else:
                A_counts = Counter(A_assignment)
                B_counts = Counter(B_assignment)
                A_ratios = {el: round(A_counts[el] / len(La_sites), 2) for el in A_counts}
                B_ratios = {el: round(B_counts[el] / len(Co_sites), 2) for el in B_counts}

            # 构造比例字符串，如 "La0.25-Sr0.75"
                A_ratio_str = "-".join([f"{el}{A_ratios[el]:.2f}" for el in sorted(A_ratios)])
                B_ratio_str = "-".join([f"{el}{B_ratios[el]:.2f}" for el in sorted(B_ratios)])

            # 文件名中加入比例信息
                filename = f"{i + 1}({A_ratio_str})({B_ratio_str}).cif"
                filepath = os.path.join(output_dir, filename)
                new_struct.to(filename=filepath)
                i += 1



if __name__ == '__main__':
    generate_random_structures(structure, La_replacements, Co_replacements, output_dir)