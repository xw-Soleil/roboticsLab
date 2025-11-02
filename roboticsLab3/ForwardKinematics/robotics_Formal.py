import sympy as sp
import numpy as np

print("=" * 100)
print("实验3：ZJU-I 型机械臂正运动学求解")
print("=" * 100)

# ==================== 使用你已经计算好的符号矩阵 ====================
# 定义符号变量
theta1, theta2, theta3, theta4, theta5, theta6 = sp.symbols('theta1 theta2 theta3 theta4 theta5 theta6')
d1, d4, d5, d6, a2, a3 = sp.symbols('d1 d4 d5 d6 a2 a3')

# 定义三角函数简写
c1, s1 = sp.cos(theta1), sp.sin(theta1)
c2, s2 = sp.cos(theta2), sp.sin(theta2)
c3, s3 = sp.cos(theta3), sp.sin(theta3)
c4, s4 = sp.cos(theta4), sp.sin(theta4)
c5, s5 = sp.cos(theta5), sp.sin(theta5)
c6, s6 = sp.cos(theta6), sp.sin(theta6)

# 定义六个变换矩阵（根据你提供的矩阵）
T_01 = sp.Matrix([
    [c1, -s1, 0, 0],
    [s1,  c1, 0, 0],
    [0,   0,  1, d1],
    [0,   0,  0, 1]
])

T_12 = sp.Matrix([
    [s2,  c2, 0, 0],
    [0,   0,  1, 0],
    [c2, -s2, 0, 0],
    [0,   0,  0, 1]
])

T_23 = sp.Matrix([
    [c3, -s3, 0, a2],
    [s3,  c3, 0, 0],
    [0,   0,  1, 0],
    [0,   0,  0, 1]
])

T_34 = sp.Matrix([
    [-s4, -c4, 0, a3],
    [c4,  -s4, 0, 0],
    [0,    0,  1, d4],
    [0,    0,  0, 1]
])

T_45 = sp.Matrix([
    [-s5, -c5,  0,   0],
    [0,    0,  -1, -d5],
    [c5,  -s5,  0,   0],
    [0,    0,   0,   1]
])

T_56 = sp.Matrix([
    [c6, -s6,  0,   0],
    [0,   0,  -1, -d6],
    [s6,  c6,  0,   0],
    [0,   0,   0,   1]
])

print("\n第1步：逐步计算总变换矩阵")
print("-" * 100)

# 逐步计算总变换矩阵
print("\n计算 T_02 = T_01 × T_12...")
T_02 = T_01 * T_12
T_02 = sp.simplify(T_02)

print("计算 T_03 = T_02 × T_23...")
T_03 = T_02 * T_23
T_03 = sp.simplify(T_03)

print("计算 T_04 = T_03 × T_34...")
T_04 = T_03 * T_34
T_04 = sp.simplify(T_04)

print("计算 T_05 = T_04 × T_45...")
T_05 = T_04 * T_45
T_05 = sp.simplify(T_05)

print("计算 T_06 = T_05 × T_56...")
T_06 = T_05 * T_56
T_06 = sp.simplify(T_06)

print("✓ 符号计算完成")

# 提取位置和旋转矩阵
px = T_06[0, 3]
py = T_06[1, 3]
pz = T_06[2, 3]
R = T_06[0:3, 0:3]

# ==================== 第2步：提取欧拉角 ====================
print("\n\n第2步：从旋转矩阵提取 X'Y'Z' 欧拉角")
print("-" * 100)

r13 = R[0, 2]
r11 = R[0, 0]
r12 = R[0, 1]
r23 = R[1, 2]
r33 = R[2, 2]

beta = sp.asin(r13)
alpha = sp.atan2(-r23, r33)
gamma = sp.atan2(-r12, r11)

print("✓ 欧拉角提取完成")

# ==================== 第3步：代入 ZJU-I 型机械臂的具体参数 ====================
print("\n\n第3步：代入 ZJU-I 型机械臂的具体 DH 参数值")
print("-" * 100)

# ZJU-I 型机械臂的实际参数（单位：mm）
robot_params = {
    d1: 230,
    a2: 185,
    a3: 170,
    d4: 23,
    d5: 77,
    d6: 85.5
}

print("DH 参数值:")
for param, value in robot_params.items():
    print(f"  {param} = {value} mm")

# ==================== 第4步：数值验证（5组关节角） ====================
print("\n\n第4步：计算5组给定关节角的末端位置和姿态")
print("=" * 100)

# 5组关节角参数（从题目中获取，注意这里的角度已经是相对于零位的）
joint_configs = [
    {
        'name': '①',
        'angles_deg': (30, 0, 30, 0, 60, 0),
        'angles_rad': (sp.pi/6, 0, sp.pi/6, 0, sp.pi/3, 0)
    },
    {
        'name': '②',
        'angles_deg': (30, 30, 60, 0, 60, 30),
        'angles_rad': (sp.pi/6, sp.pi/6, sp.pi/3, 0, sp.pi/3, sp.pi/6)
    },
    {
        'name': '③',
        'angles_deg': (90, 0, 90, 60, 60, 30),
        'angles_rad': (sp.pi/2, 0, sp.pi/2, sp.pi/3, sp.pi/3, sp.pi/6)
    },
    {
        'name': '④',
        'angles_deg': (-30, -30, -60, 0, -15, 90),
        'angles_rad': (-sp.pi/6, -sp.pi/6, -sp.pi/3, 0, -sp.pi/12, sp.pi/2)
    },
    {
        'name': '⑤',
        'angles_deg': (15, 15, 15, 15, 15, 15),
        'angles_rad': (sp.pi/12, sp.pi/12, sp.pi/12, sp.pi/12, sp.pi/12, sp.pi/12)
    }
]

results = []

for config in joint_configs:
    print(f"\n配置 {config['name']}: θ = {config['angles_deg']}°")
    print("-" * 100)
    
    # 代入关节角和机器人参数
    subs_dict = {
        theta1: config['angles_rad'][0],
        theta2: config['angles_rad'][1],
        theta3: config['angles_rad'][2],
        theta4: config['angles_rad'][3],
        theta5: config['angles_rad'][4],
        theta6: config['angles_rad'][5],
        **robot_params  # 解包机器人参数
    }
    
    # 计算数值结果
    T_numeric = T_06.subs(subs_dict)
    
    px_val = float(T_numeric[0, 3])
    py_val = float(T_numeric[1, 3])
    pz_val = float(T_numeric[2, 3])
    
    # 提取旋转矩阵并计算欧拉角
    R_numeric = T_numeric[0:3, 0:3]
    
    r13_val = float(R_numeric[0, 2])
    r11_val = float(R_numeric[0, 0])
    r12_val = float(R_numeric[0, 1])
    r23_val = float(R_numeric[1, 2])
    r33_val = float(R_numeric[2, 2])
    
    # 计算欧拉角
    beta_val = np.arcsin(np.clip(r13_val, -1, 1))  # clip 防止数值误差
    alpha_val = np.arctan2(-r23_val, r33_val)
    gamma_val = np.arctan2(-r12_val, r11_val)
    
    print(f"末端位置 (mm):")
    print(f"  px = {px_val:10.4f}")
    print(f"  py = {py_val:10.4f}")
    print(f"  pz = {pz_val:10.4f}")
    
    print(f"\n末端姿态 (X'Y'Z' 欧拉角):")
    print(f"  α = {alpha_val:8.4f} rad = {np.degrees(alpha_val):8.2f}°")
    print(f"  β = {beta_val:8.4f} rad = {np.degrees(beta_val):8.2f}°")
    print(f"  γ = {gamma_val:8.4f} rad = {np.degrees(gamma_val):8.2f}°")
    
    results.append({
        'config': config['name'],
        'joints_deg': config['angles_deg'],
        'joints_rad': config['angles_rad'],
        'position': (px_val, py_val, pz_val),
        'euler_rad': (alpha_val, beta_val, gamma_val),
        'euler_deg': (np.degrees(alpha_val), np.degrees(beta_val), np.degrees(gamma_val))
    })

# ==================== 第5步：保存结果 ====================
print("\n\n第5步：保存结果到文件")
print("=" * 100)

with open('ZJU_I_forward_kinematics_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 120 + "\n")
    f.write("ZJU-I 型机械臂正运动学求解结果\n")
    f.write("=" * 120 + "\n\n")
    
    # DH 参数
    f.write("1. ZJU-I 型机械臂 DH 参数:\n")
    f.write("-" * 120 + "\n")
    f.write(f"  d₁ = {robot_params[d1]} mm\n")
    f.write(f"  a₂ = {robot_params[a2]} mm\n")
    f.write(f"  a₃ = {robot_params[a3]} mm\n")
    f.write(f"  d₄ = {robot_params[d4]} mm\n")
    f.write(f"  d₅ = {robot_params[d5]} mm\n")
    f.write(f"  d₆ = {robot_params[d6]} mm\n\n\n")
    
    # 位置表达式
    f.write("2. 末端位置的符号表达式:\n")
    f.write("-" * 120 + "\n")
    f.write(f"px = {px}\n\n")
    f.write(f"py = {py}\n\n")
    f.write(f"pz = {pz}\n\n\n")
    
    # 5组配置的数值结果
    f.write("3. 五组关节角配置的数值结果:\n")
    f.write("-" * 120 + "\n\n")
    
    for res in results:
        f.write(f"配置 {res['config']}:\n")
        f.write(f"  关节角 (deg): θ₁={res['joints_deg'][0]:6.1f}°, θ₂={res['joints_deg'][1]:6.1f}°, "
                f"θ₃={res['joints_deg'][2]:6.1f}°, θ₄={res['joints_deg'][3]:6.1f}°, "
                f"θ₅={res['joints_deg'][4]:6.1f}°, θ₆={res['joints_deg'][5]:6.1f}°\n")
        
        f.write(f"  末端位置 (mm):\n")
        f.write(f"    px = {res['position'][0]:10.4f} mm\n")
        f.write(f"    py = {res['position'][1]:10.4f} mm\n")
        f.write(f"    pz = {res['position'][2]:10.4f} mm\n")
        
        f.write(f"  末端姿态 (X'Y'Z' 欧拉角):\n")
        f.write(f"    α = {res['euler_rad'][0]:8.4f} rad = {res['euler_deg'][0]:8.2f}°\n")
        f.write(f"    β = {res['euler_rad'][1]:8.4f} rad = {res['euler_deg'][1]:8.2f}°\n")
        f.write(f"    γ = {res['euler_rad'][2]:8.4f} rad = {res['euler_deg'][2]:8.2f}°\n\n")
    
    # 汇总表格
    f.write("\n" + "=" * 120 + "\n")
    f.write("4. 实验结果汇总表:\n")
    f.write("=" * 120 + "\n")
    f.write("配置 | θ₁(°) | θ₂(°) | θ₃(°) | θ₄(°) | θ₅(°) | θ₆(°) |   px(mm)  |   py(mm)  |   pz(mm)  |  α(°)   |  β(°)   |  γ(°)\n")
    f.write("-" * 120 + "\n")
    for res in results:
        j = res['joints_deg']
        p = res['position']
        e = res['euler_deg']
        f.write(f" {res['config']}  | {j[0]:5.1f} | {j[1]:5.1f} | {j[2]:5.1f} | "
                f"{j[3]:5.1f} | {j[4]:5.1f} | {j[5]:5.1f} | "
                f"{p[0]:9.2f} | {p[1]:9.2f} | {p[2]:9.2f} | "
                f"{e[0]:7.2f} | {e[1]:7.2f} | {e[2]:7.2f}\n")
    
    # LaTeX 代码
    f.write("\n\n" + "=" * 120 + "\n")
    f.write("5. LaTeX 代码 (用于实验报告):\n")
    f.write("=" * 120 + "\n")
    f.write(sp.latex(T_06.subs(robot_params)))

print("✓ 结果已保存到 ZJU_I_forward_kinematics_results.txt")

# ==================== 第6步：生成实验报告表格 ====================
print("\n\n第6步：实验结果汇总")
print("=" * 100)
print("\n实验结果汇总表:")
print("-" * 100)
print("配置 | θ₁(°) | θ₂(°) | θ₃(°) | θ₄(°) | θ₅(°) | θ₆(°) |   px(mm)  |   py(mm)  |   pz(mm)  |  α(°)   |  β(°)   |  γ(°)")
print("-" * 100)
for res in results:
    j = res['joints_deg']
    p = res['position']
    e = res['euler_deg']
    print(f" {res['config']}  | {j[0]:5.1f} | {j[1]:5.1f} | {j[2]:5.1f} | "
          f"{j[3]:5.1f} | {j[4]:5.1f} | {j[5]:5.1f} | "
          f"{p[0]:9.2f} | {p[1]:9.2f} | {p[2]:9.2f} | "
          f"{e[0]:7.2f} | {e[1]:7.2f} | {e[2]:7.2f}")

print("\n\n" + "=" * 100)
print("正运动学求解完成！")
print("=" * 100)