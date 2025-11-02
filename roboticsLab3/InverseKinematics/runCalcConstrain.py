import numpy as np

class IKSolver:
    def __init__(self, end_lists, joint_limits_deg=None):
        self.end_lists = end_lists
        # 默认关节限位（度），可以根据实际机械臂修改
        if joint_limits_deg is None:
            joint_limits_deg = np.array([
                [-180, 180],  # J1
                [-90, 90],    # J2
                [-150, 150],  # J3
                [-180, 180],  # J4
                [-120, 120],  # J5
                [-360, 360]   # J6
            ])
        self.joint_limits_rad = np.deg2rad(joint_limits_deg)
        
        # D-H参数
        self.a = [0, 0, 0.185, 0.170, 0, 0]
        self.d = [0.230, 0, 0, 0.023, 0.077, 0.0855]

    def is_within_limits(self, th):
        """检查关节角度是否在限位内"""
        for i in range(6):
            if not (self.joint_limits_rad[i, 0] <= th[i] <= self.joint_limits_rad[i, 1]):
                return False
        return True

    def solve_one_pose(self, X, Y, Z, r, p, y):
        """计算单个目标位姿的所有可行解"""
        sgn_lists = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], 
                     [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
        
        valid_solutions = []
        a = self.a
        d = self.d
        
        T_ = np.array([[X], [Y], [Z]])
        red = np.array([0, 0, 0, 1])

        R = np.array([[1, 0, 0],[0, np.cos(r), -np.sin(r)],[0, np.sin(r), np.cos(r)]]) @ \
            np.array([[np.cos(p), 0, np.sin(p)],[0, 1, 0],[-np.sin(p), 0, np.cos(p)]]) @ \
            np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0],[0, 0, 1]])
        T = np.hstack([R, T_])
        T = np.vstack([T, red])

        r11, r12, r13, px = T[0, :]
        r21, r22, r23, py = T[1, :]
        r31, r32, r33, pz = T[2, :]

        # 主循环
        for sgn1, sgn2, sgn3 in sgn_lists:
            # step1: solve theta_1, theta_5, theta_6
            A = d[5] * r13 - px
            B = d[5] * r23 - py
            discriminant = A**2 + B**2 - d[3]**2
            if discriminant < 0:
                continue

            theta_1 = np.arctan2(B, A) + np.arctan2(d[3], sgn1 * np.sqrt(discriminant))

            sin_theta5 = r23 * np.cos(theta_1) - r13 * np.sin(theta_1)
            sin_theta5 = np.clip(sin_theta5, -1, 1)
            theta_5 = sgn2 * np.arcsin(sin_theta5)

            if np.abs(np.cos(theta_5)) < 1e-6:
                theta_6 = 0
            else:
                sin_theta6 = (np.sin(theta_1) * r12 - np.cos(theta_1) * r22) / np.cos(theta_5)
                sin_theta6 = np.clip(sin_theta6, -1, 1)
                theta_6 = np.arcsin(sin_theta6)

            # step2: solve theta_2, theta_3
            M = px * np.cos(theta_1) + py * np.sin(theta_1) - d[5] * (r13 * np.cos(theta_1) + r23 * np.sin(theta_1)) - \
                d[4] * (r21 * np.sin(theta_1) * np.sin(theta_6) + r12 * np.cos(theta_1) * np.cos(theta_6) + \
                r11 * np.cos(theta_1) * np.sin(theta_6) + r22 * np.cos(theta_6) * np.sin(theta_1))
            N = pz - d[0] - d[5] * r33 - d[4] * (r32 * np.cos(theta_6) + r31 * np.sin(theta_6))

            cos_theta3 = (M**2 + N**2 - a[2]**2 - a[3]**2) / (2 * a[2] * a[3])
            if cos_theta3 > 1 + 1e-6 or cos_theta3 < -1 - 1e-6:
                continue
            cos_theta3 = np.clip(cos_theta3, -1, 1)

            theta_3 = sgn3 * np.arccos(cos_theta3)

            k1 = a[2] + a[3] * np.cos(theta_3)
            k2 = a[3] * np.sin(theta_3)

            theta_2 = np.arctan2(M, N) - np.arctan2(k2, k1)

            # step3: solve theta_4
            if np.abs(np.cos(theta_5)) < 1e-6:
                theta_4 = 0
            else:
                sin_theta4 = -r33 / np.cos(theta_5)
                sin_theta4 = np.clip(sin_theta4, -1, 1)
                theta_4 = np.arcsin(sin_theta4) - theta_2 - theta_3

            th = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])

            # 归一化到 [-pi, pi]
            th_ = th < -np.pi
            th[th_] += np.pi * 2
            th_ = th > np.pi
            th[th_] -= np.pi * 2

            # 约束检查
            if np.any(np.isnan(th)):
                continue
            
            if not self.is_within_limits(th):
                continue

            # 检查重复
            is_duplicate = False
            for sol in valid_solutions:
                if np.allclose(th, sol, rtol=1e-3, atol=1e-3):
                    is_duplicate = True
                    break
            if not is_duplicate:
                valid_solutions.append(th)

        # 额外检查theta_3 = 0的特殊情况
        for sgn1, sgn2 in [[-1, -1], [-1, 1], [1, -1], [1, 1]]:
            theta_3 = 0
            A = d[5] * r13 - px
            B = d[5] * r23 - py
            discriminant = A**2 + B**2 - d[3]**2
            if discriminant < 0:
                continue

            theta_1 = np.arctan2(B, A) + np.arctan2(d[3], sgn1 * np.sqrt(discriminant))
            sin_theta5 = r23 * np.cos(theta_1) - r13 * np.sin(theta_1)
            if abs(sin_theta5) > 1 + 1e-6:
                continue
            sin_theta5 = np.clip(sin_theta5, -1, 1)
            theta_5 = sgn2 * np.arcsin(sin_theta5)
            if np.abs(np.cos(theta_5)) < 1e-6:
                theta_6 = 0
            else:
                sin_theta6 = (np.sin(theta_1) * r12 - np.cos(theta_1) * r22) / np.cos(theta_5)
                sin_theta6 = np.clip(sin_theta6, -1, 1)
                theta_6 = np.arcsin(sin_theta6)

            M = px * np.cos(theta_1) + py * np.sin(theta_1) - d[5] * (r13 * np.cos(theta_1) + r23 * np.sin(theta_1)) - \
                d[4] * (r21 * np.sin(theta_1) * np.sin(theta_6) + r12 * np.cos(theta_1) * np.cos(theta_6) + \
                r11 * np.cos(theta_1) * np.sin(theta_6) + r22 * np.cos(theta_6) * np.sin(theta_1))
            N = pz - d[0] - d[5] * r33 - d[4] * (r32 * np.cos(theta_6) + r31 * np.sin(theta_6))
            
            required_dist_sq = M**2 + N**2
            actual_dist_sq = (a[2] + a[3])**2
            if np.abs(required_dist_sq - actual_dist_sq) > 1e-3:
                continue
            theta_2 = np.arctan2(M, N)
            if np.abs(np.cos(theta_5)) < 1e-6:
                theta_4 = 0
            else:
                sin_theta4 = -r33 / np.cos(theta_5)
                sin_theta4 = np.clip(sin_theta4, -1, 1)
                theta_4 = np.arcsin(sin_theta4) - theta_2
            
            th = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
            th_ = th < -np.pi
            th[th_] += np.pi * 2
            th_ = th > np.pi
            th[th_] -= np.pi * 2
            
            if np.any(np.isnan(th)):
                continue
            
            if not self.is_within_limits(th):
                continue
                
            is_duplicate = False
            for sol in valid_solutions:
                if np.allclose(th, sol, rtol=1e-3, atol=1e-3):
                    is_duplicate = True
                    break
            if not is_duplicate:
                valid_solutions.append(th)
        
        return valid_solutions

    def solve(self, print_all=False, continuous_with_positive_theta5=False):
        """
        求解所有目标位姿
        :param print_all: 是否打印所有可行解
        :param continuous_with_positive_theta5: 第一个点选正theta_5，后续用连续性
        """
        current_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        is_first_point = True
        
        for i in range(len(self.end_lists)):
            X, Y, Z, r, p, y = self.end_lists[i]
            print("IK solutions of end %d: " % (i+1))
            
            valid_solutions = self.solve_one_pose(X, Y, Z, r, p, y)
            
            if not valid_solutions:
                print("  No valid solution found within joint limits.")
            else:
                if print_all:
                    # 模式1: 打印所有解
                    for sol in valid_solutions:
                        print("  ", np.rad2deg(sol).round(2))
                elif continuous_with_positive_theta5:
                    # 模式4: 第一个点选正theta_5，后续用连续性
                    if is_first_point:
                        # 第一个点：优先选择正theta_5
                        positive_theta5_sols = [sol for sol in valid_solutions if sol[4] > 0]
                        
                        if positive_theta5_sols:
                            best_sol = max(positive_theta5_sols, key=lambda x: x[4])
                        else:
                            best_sol = max(valid_solutions, key=lambda x: x[4])
                        
                        print("  Selected solution (first point: positive theta_5):")
                        is_first_point = False
                    else:
                        # 后续点：连续性约束
                        min_dist = np.inf
                        best_sol = None
                        for sol in valid_solutions:
                            dist = np.sum((sol - current_q)**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_sol = sol
                        print("  Selected solution (closest to current):")
                    
                    print("  ", np.rad2deg(best_sol).round(2))
                    current_q = best_sol
            print()

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=2)
    
    # 定义目标位姿
    end_lists = [[0.117, 0.334, 0.499, -2.019, -0.058, -2.190],
                 [-0.066, 0.339, 0.444, -2.618, -0.524, -3.141],
                 [0.3, 0.25, 0.26, -2.64, 0.59, -2.35],
                 [0.42, 0, 0.36, 3.14, 1, -1.57],
                 [0.32, -0.25, 0.16, 3, 0.265, -0.84]]
    
    # 创建求解器
    solver = IKSolver(end_lists)
    
    print("=" * 60)
    print("模式1: 打印所有可行解")
    print("=" * 60)
    solver.solve(print_all=True)
    
    print("\n" + "=" * 60)
    print("模式2: theta_5取正，并使用连续性约束")
    print("=" * 60)
    solver.solve(continuous_with_positive_theta5=True)