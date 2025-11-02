import sympy as sp

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

# 定义六个变换矩阵（根据图片）
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

print("=" * 80)
print("开始计算总变换矩阵 T_6^0...")
print("=" * 80)

# 逐步计算总变换矩阵
print("\n计算 T_02 = T_01 × T_12...")
T_02 = T_01 * T_12
T_02 = sp.simplify(T_02)

print("\n计算 T_03 = T_02 × T_23...")
T_03 = T_02 * T_23
T_03 = sp.simplify(T_03)

print("\n计算 T_04 = T_03 × T_34...")
T_04 = T_03 * T_34
T_04 = sp.simplify(T_04)

print("\n计算 T_05 = T_04 × T_45...")
T_05 = T_04 * T_45
T_05 = sp.simplify(T_05)

print("\n计算 T_06 = T_05 × T_56...")
T_06 = T_05 * T_56
T_06 = sp.simplify(T_06)

print("\n" + "=" * 80)
print("最终的变换矩阵 T_6^0:")
print("=" * 80)
sp.pprint(T_06)

# 提取位置信息
print("\n" + "=" * 80)
print("末端执行器位置 (px, py, pz):")
print("=" * 80)
px = T_06[0, 3]
py = T_06[1, 3]
pz = T_06[2, 3]

print(f"px = {px}")
print(f"py = {py}")
print(f"pz = {pz}")

# 提取旋转矩阵（3x3）
print("\n" + "=" * 80)
print("末端执行器姿态矩阵 R_6^0 (3×3):")
print("=" * 80)
R = T_06[0:3, 0:3]
sp.pprint(R)

# 保存为 LaTeX（可选）
print("\n" + "=" * 80)
print("LaTeX 代码:")
print("=" * 80)
latex_code = sp.latex(T_06)
print(latex_code)

# 也可以用三角函数简化
print("\n" + "=" * 80)
print("使用三角恒等式简化后的结果:")
print("=" * 80)
T_06_trig = sp.trigsimp(T_06)
sp.pprint(T_06_trig)