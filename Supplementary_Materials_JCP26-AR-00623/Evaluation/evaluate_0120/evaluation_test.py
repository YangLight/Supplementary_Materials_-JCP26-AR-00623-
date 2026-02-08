import os
from ase.io import read
import numpy as np
from mace.calculators import MACECalculator

# 创建输出文件夹
os.makedirs("pairplot_test", exist_ok=True)
model_path = "MACE_MPA_ft.model"
device = "cuda"
calc = MACECalculator(model_paths=model_path, device=device, enable_cueq=True)

print("开始计算 MACE 能量、力和stress...")

# 打开三个文件分别用于写入能量、力和stress
with open('pairplot_test/energy_ace.dat', 'w') as fe, \
     open('pairplot_test/forces_ace.dat', 'w') as ff, \
     open('pairplot_test/stress_ace.dat', 'w') as fs:
    
    # 写入文件头（保持不变）
    fe.write("# Structure Index | Potential Energy (eV)\n")
    ff.write("# Structure Index | Forces (eV/Å) for each atom\n")
    fs.write("# Structure Index | Stress (9 components in eV/Å^3)\n")  # stress文件头
    
    for ii in range(74):
        ats = read('test_mace.extxyz', str(ii))
        ats.calc = calc
        
        # 获取计算结果（保持不变）
        energy = ats.get_potential_energy()
        forces = ats.get_forces()
        # 获取stress，确保是3x3矩阵
        stress_matrix = ats.get_stress(voigt=False)  # 3x3矩阵
        
        # 写入能量文件（保持不变）
        fe.write(f"{ii:6d} {energy:18.10f}\n")
        
        # 写入力文件（保持不变）
        ff.write(f"Structure {ii}:\n")
        np.savetxt(ff, forces, fmt='%16.10f')
        ff.write("\n")  # 结构间加空行
        
        # 写入stress文件 - 改为与energy类似的简单格式
        # 将3x3矩阵展平为9个分量
        stress_flat = stress_matrix.flatten()
        fs.write(f"{ii:6d} ")  # 与energy相同的开头格式
        # 写入9个分量
        for j in range(9):
            fs.write(f"{stress_flat[j]:16.10f}")
        fs.write("\n")
        
        # 打印进度（保持不变）
        if (ii+1) % 20 == 0 or ii == 0:
            print(f"Processed {ii+1}/74 structures")

print("MACE 计算完成，开始提取参考数据...")
# 打开文件用于写入参考数据
with open('pairplot_test/energy_ref.dat', 'w') as fe, \
     open('pairplot_test/forces_ref.dat', 'w') as ff, \
     open('pairplot_test/stress_ref.dat', 'w') as fs:

    # 写入文件头（保持不变）
    fe.write("# Structure Index | Reference Energy (eV)\n")
    ff.write("# Structure Index | Reference Forces (eV/Å) for each atom\n")
    fs.write("# Structure Index | Reference stress (9 components in eV/Å^3)\n")  # stress文件头

    for ii in range(74):  # 假设共有 74 个结构
        ats = read('test_mace.extxyz', index=ii)  # 读取第 ii 个结构

        # 从 atoms.info 或 atoms.arrays 获取参考数据（保持不变）
        ref_energy = ats.info.get('REF_energy')  # 参考能量
        ref_forces = ats.arrays.get('REF_forces')  # 参考力（每个原子的力）
        ref_stress = ats.info.get('REF_stress')  # 参考stress（可能是9分量或3x3矩阵）

        # 写入能量文件（保持不变）
        fe.write(f"{ii:6d} {ref_energy:18.10f}\n")

        # 写入力文件（保持不变）
        ff.write(f"Structure {ii}:\n")
        np.savetxt(ff, ref_forces, fmt='%16.10f')
        ff.write("\n")  # 结构间加空行

        # 写入stress文件 - 改为与energy类似的简单格式
        fs.write(f"{ii:6d} ")
        
        if ref_stress is not None:
            # 确保stress是9个分量
            stress_array = np.array(ref_stress)
            
            # 如果stress是3x3矩阵，展平为9个分量
            if stress_array.shape == (3, 3):
                stress_flat = stress_array.flatten()
            # 如果stress已经是9个分量的一维数组
            elif stress_array.shape == (9,):
                stress_flat = stress_array
            # 如果是6分量的Voigt记号，转换为9分量
            elif stress_array.shape == (6,):
                # Voigt: xx, yy, zz, yz, xz, xy
                # 转换为3x3矩阵
                stress_3x3 = np.zeros((3, 3))
                stress_3x3[0, 0] = stress_array[0]  # xx
                stress_3x3[1, 1] = stress_array[1]  # yy
                stress_3x3[2, 2] = stress_array[2]  # zz
                stress_3x3[1, 2] = stress_3x3[2, 1] = stress_array[3]  # yz
                stress_3x3[0, 2] = stress_3x3[2, 0] = stress_array[4]  # xz
                stress_3x3[0, 1] = stress_3x3[1, 0] = stress_array[5]  # xy
                stress_flat = stress_3x3.flatten()
            else:
                print(f"Warning: Structure {ii} REF_stress has shape {stress_array.shape}, using zeros")
                stress_flat = np.zeros(9)
        else:
            print(f"Warning: Structure {ii} has no REF_stress data")
            stress_flat = np.zeros(9)
        
        # 写入9个stress分量
        for j in range(9):
            fs.write(f"{stress_flat[j]:16.10f}")
        fs.write("\n")

    print("参考数据提取完成！")

# 1. 读取原子数（保持不变）
print("读取原子数...")
atom_counts = []
for ii in range(74):  # 共74个结构
    ats = read('test_mace.extxyz', index=ii)
    atom_counts.append(len(ats))
atom_counts = np.array(atom_counts)

# 2. 处理能量数据 (转换为eV/atom)（保持不变）
print("处理能量数据...")
ref_energies = np.loadtxt('pairplot_test/energy_ref.dat', usecols=1)
calc_energies = np.loadtxt('pairplot_test/energy_ace.dat', usecols=1)

# 转换为eV/atom并格式化
with open('pairplot_test/energy_pairplot_test.dat', 'w') as f:
    f.write("参考能量(eV/atom)\t计算能量(eV/atom)\n")  # 表头
    for ref, calc, natom in zip(ref_energies, calc_energies, atom_counts):
        ref_per_atom = ref/natom
        calc_per_atom = calc/natom
        f.write(f"{ref_per_atom:.10f}\t{calc_per_atom:.10f}\n")

# 3. 处理力数据（保持不变）
print("处理力数据...")

def read_forces(filename):
    """读取力数据并展平"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    forces = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('#', 'Structure')):
            parts = line.split()
            if len(parts) == 3:
                forces.extend([float(x) for x in parts])
    return forces

# 读取并格式化力数据
ref_forces = read_forces('pairplot_test/forces_ref.dat')
calc_forces = read_forces('pairplot_test/forces_ace.dat')

# 写入Excel可用格式（所有力排成一列）
with open('pairplot_test/forces_pairplot_test.dat', 'w') as f:
    f.write("参考力(eV/Å)\t计算力(eV/Å)\n")  # 表头
    for ref, calc in zip(ref_forces, calc_forces):
        f.write(f"{ref:.10f}\t{calc:.10f}\n")

# 4. 处理stress数据 - 改为与energy类似的处理方式
print("处理stress数据 (9分量)...")

def read_stress_simple(filename):
    """读取stress数据 - 与读取energy的方式相同"""
    stresses = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 10:  # 索引 + 9个stress分量
                    # 提取9个stress分量
                    stress_values = [float(x) for x in parts[1:10]]
                    stresses.extend(stress_values)  # 展平所有分量
    return stresses

# 读取参考和计算的stress数据
ref_stress_flat = read_stress_simple('pairplot_test/stress_ref.dat')
calc_stress_flat = read_stress_simple('pairplot_test/stress_ace.dat')

print(f"参考stress数据点: {len(ref_stress_flat)}")
print(f"计算stress数据点: {len(calc_stress_flat)}")

# 确保数据长度一致
min_length = min(len(ref_stress_flat), len(calc_stress_flat))
ref_stress_flat = ref_stress_flat[:min_length]
calc_stress_flat = calc_stress_flat[:min_length]

# 保存为stress_pairplot_test.dat - 改为与energy/force完全相同的格式
print("合并所有stress分量到单一文件...")
with open('pairplot_test/stress_pairplot_test.dat', 'w') as f:
    f.write("参考stress(eV/Å³)\t计算stress(eV/Å³)\n")  # 表头，与energy/force格式一致
    
    # 直接按顺序写入所有stress分量
    for ref, calc in zip(ref_stress_flat, calc_stress_flat):
        f.write(f"{ref:.10f}\t{calc:.10f}\n")

print("\n所有数据处理完成！生成的文件：")
print("1. energy_pairplot_test.dat - 能量数据（参考值在前）")
print("2. forces_pairplot_test.dat - 所有力数据（展平，参考值在前）")
print("3. stress_pairplot_test.dat - 所有stress数据（9分量，展平，参考值在前）")

print(f"\n数据统计：")
print(f"- 能量数据：{len(ref_energies)} 对")
print(f"- 力数据：{len(ref_forces)} 对")
print(f"- stress数据：{len(ref_stress_flat)} 对 (9分量×{len(ref_stress_flat)//9}结构)")

print("\nstress分量顺序说明：")
print("每个结构的9分量顺序: [xx, xy, xz, yx, yy, yz, zx, zy, zz]")
print("对应3x3矩阵: [[xx, xy, xz],")
print("             [yx, yy, yz],")
print("             [zx, zy, zz]]")

print("\n注意：所有stress数据单位均为 eV/Å³ (电子伏特每立方埃)")