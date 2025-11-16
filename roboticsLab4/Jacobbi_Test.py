import numpy as np
try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
except ImportError:
    print("请先安装依赖: pip install roboticstoolbox-python")
    print("如果安装失败,请尝试: pip install roboticstoolbox-python spatialmath-python")
    exit(1)


def analytical_jacobian(q):
    """
    根据文档推导计算ZJU-I机械臂的解析雅可比矩阵
    使用Modified DH参数
    
    参数:
        q: 关节角度数组 [q1, q2, q3, q4, q5, q6] (弧度)
    
    返回:
        J: 6x6雅可比矩阵
    """
    q1, q2, q3, q4, q5, q6 = q
    
    # 简化记号
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s3, c3 = np.sin(q3), np.cos(q3)
    s4, c4 = np.sin(q4), np.cos(q4)
    s5, c5 = np.sin(q5), np.cos(q5)
    s6, c6 = np.sin(q6), np.cos(q6)
    
    # 和角
    s23 = np.sin(q2 + q3)
    c23 = np.cos(q2 + q3)
    s234 = np.sin(q2 + q3 + q4)  # Sigma
    c234 = np.cos(q2 + q3 + q4)
    
    # 常数参数
    d1 = 230
    a2 = 185
    a3 = 170
    d4 = 23
    d5 = 77
    d6 = 85.5  # 171/2
    
    # 末端位置 p6
    xe = (-d6 * s1 * s5 - d4 * s1 
          + a2 * s2 * c1 
          + a3 * s23 * c1 
          + d5 * s234 * c1 
          + d6 * c1 * c5 * c234)
    
    ye = (a2 * s1 * s2 
          + a3 * s1 * s23 
          + d5 * s1 * s234 
          + d6 * s1 * c5 * c234 
          + d6 * s5 * c1 + d4 * c1)
    
    ze = (-d6 * s234 * c5 
          + a2 * c2 + a3 * c23 
          + d5 * c234 + d1)
    
    # 初始化雅可比矩阵
    J = np.zeros((6, 6))
    
    # ========== 第1列 (关节1) ==========
    # 线速度部分
    J[0, 0] = (-a2 * s1 * s2 
               - a3 * s1 * s23 
               - d5 * s1 * s234 
               - d6 * s1 * c5 * c234 
               - d6 * s5 * c1 - d4 * c1)
    
    J[1, 0] = (-d6 * s1 * s5 - d4 * s1 
               + a2 * s2 * c1 
               + a3 * s23 * c1 
               + d5 * s234 * c1 
               + d6 * c1 * c5 * c234)
    
    J[2, 0] = 0
    
    # 角速度部分
    J[3, 0] = 0
    J[4, 0] = 0
    J[5, 0] = 1
    
    # ========== 第2列 (关节2) ==========
    # 线速度部分
    Jv2_common = (-d6 * s234 * c5 
                  + a2 * c2 + a3 * c23 
                  + d5 * c234)
    
    J[0, 1] = c1 * Jv2_common
    J[1, 1] = s1 * Jv2_common
    J[2, 1] = (-a2 * s2 - a3 * s23 
               - d5 * s234 - d6 * c5 * c234)
    
    # 角速度部分
    J[3, 1] = -s1
    J[4, 1] = c1
    J[5, 1] = 0
    
    # ========== 第3列 (关节3) ==========
    # 线速度部分
    Jv3_common = (-d6 * s234 * c5 
                  + a3 * c23 + d5 * c234)
    
    J[0, 2] = c1 * Jv3_common
    J[1, 2] = s1 * Jv3_common
    J[2, 2] = (-a3 * s23 - d5 * s234 
               - d6 * c5 * c234)
    
    # 角速度部分
    J[3, 2] = -s1
    J[4, 2] = c1
    J[5, 2] = 0
    
    # ========== 第4列 (关节4) ==========
    # 线速度部分
    Jv4_common = -d6 * s234 * c5 + d5 * c234
    
    J[0, 3] = c1 * Jv4_common
    J[1, 3] = s1 * Jv4_common
    J[2, 3] = -d5 * s234 - d6 * c5 * c234
    
    # 角速度部分
    J[3, 3] = -s1
    J[4, 3] = c1
    J[5, 3] = 0
    
    # ========== 第5列 (关节5) ==========
    # 线速度部分
    J[0, 4] = (-d6 * s1 * c5 
               - d6 * s5 * c1 * c234)
    
    J[1, 4] = (-d6 * s1 * s5 * c234 
               + d6 * c1 * c5)
    
    J[2, 4] = d6 * s5 * s234
    
    # 角速度部分
    J[3, 4] = s234 * c1
    J[4, 4] = s1 * s234
    J[5, 4] = c234
    
    # ========== 第6列 (关节6) ==========
    # 线速度部分
    J[0, 5] = 0
    J[1, 5] = 0
    J[2, 5] = 0
    
    # 角速度部分
    J[3, 5] = -s1 * s5 + c1 * c5 * c234
    J[4, 5] = s1 * c5 * c234 + s5 * c1
    J[5, 5] = -s234 * c5
    
    return J


def create_robot_modified_dh():
    """
    使用Modified DH参数创建ZJU-I机械臂模型
    """
    # Modified DH参数表 [alpha(i-1), a(i-1), d(i), theta(i), offset]
    # 注意: roboticstoolbox的RevoluteMDH构造函数参数顺序
    
    links = [
        rtb.RevoluteMDH(alpha=0,       a=0,   d=230,  offset=0,         qlim=[-np.pi, np.pi]),
        rtb.RevoluteMDH(alpha=-np.pi/2, a=0,   d=0,    offset=-np.pi/2, qlim=[-np.pi, np.pi]),
        rtb.RevoluteMDH(alpha=0,       a=185, d=0,    offset=0,         qlim=[-np.pi, np.pi]),
        rtb.RevoluteMDH(alpha=0,       a=170, d=23,   offset=np.pi/2,  qlim=[-np.pi, np.pi]),
        rtb.RevoluteMDH(alpha=np.pi/2, a=0,   d=77,   offset=np.pi/2,  qlim=[-np.pi, np.pi]),
        rtb.RevoluteMDH(alpha=np.pi/2, a=0,   d=85.5, offset=0,         qlim=[-np.pi, np.pi]),
    ]
    
    robot = rtb.DHRobot(links, name='ZJU-I')
    return robot


def verify_jacobian(q_test):
    """
    验证解析雅可比与roboticstoolbox计算的雅可比是否一致
    
    参数:
        q_test: 测试关节角度
    """
    # 创建机器人模型
    robot = create_robot_modified_dh()
    
    # 使用roboticstoolbox计算雅可比
    J_rtb = robot.jacob0(q_test)
    
    # 使用解析公式计算雅可比
    J_analytical = analytical_jacobian(q_test)
    
    # 计算差异
    difference = J_rtb - J_analytical
    max_diff = np.max(np.abs(difference))
    
    # 验证末端位置
    T = robot.fkine(q_test)
    pos_rtb = T.t
    
    # 从解析公式计算末端位置
    q1, q2, q3, q4, q5, q6 = q_test
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23 = np.sin(q2 + q3)
    c23 = np.cos(q2 + q3)
    s234 = np.sin(q2 + q3 + q4)
    c234 = np.cos(q2 + q3 + q4)
    s5, c5 = np.sin(q5), np.cos(q5)
    
    d1, a2, a3, d4, d5, d6 = 230, 185, 170, 23, 77, 85.5
    
    xe = (-d6 * s1 * s5 - d4 * s1 + a2 * s2 * c1 + a3 * s23 * c1 + d5 * s234 * c1 + d6 * c1 * c5 * c234)
    ye = (a2 * s1 * s2 + a3 * s1 * s23 + d5 * s1 * s234 + d6 * s1 * c5 * c234 + d6 * s5 * c1 + d4 * c1)
    ze = (-d6 * s234 * c5 + a2 * c2 + a3 * c23 + d5 * c234 + d1)
    
    pos_analytical = np.array([xe, ye, ze])
    pos_diff = np.linalg.norm(pos_rtb - pos_analytical)
    
    # 输出结果
    print(f"关节角度: {np.round(np.rad2deg(q_test), 2)}° | 最大雅可比差异: {max_diff:.2e} | 位置差异: {pos_diff:.2e} | {'✓ 通过' if max_diff < 1e-10 else '✗ 失败'}")
    
    return max_diff < 1e-10


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ZJU-I型机械臂雅可比矩阵验证 (Modified DH)")
    print("=" * 80)
    
    # 测试用例
    test_cases = [
        ("测试1-零位", np.zeros(6)),
        ("测试2-全45°", np.ones(6) * np.pi/4),
        ("测试3-交替±30°", np.array([np.pi/6, -np.pi/6, np.pi/6, -np.pi/6, np.pi/6, -np.pi/6])),
        ("测试4-随机角度", np.random.RandomState(42).uniform(-np.pi/2, np.pi/2, 6)),
        ("测试5-大角度", np.array([np.pi/2, np.pi/3, -np.pi/4, np.pi/6, -np.pi/3, np.pi/2]))
    ]
    
    for name, q in test_cases:
        print(f"\n{name}: ", end="")
        verify_jacobian(q)
    
    print("\n" + "=" * 80)
