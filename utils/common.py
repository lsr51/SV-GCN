import numpy as np
import torch
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)   # 零向量返回同 shape 的 0
    return vector / norm


def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_points(p1, p2, p3):
    """ Returns the angle in radians between vectors 'p1' - 'p2' and 'p3' - 'p2'::
    """
    u1 = unit_vector(p1 - p2)
    u2 = unit_vector(p3 - p2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))


def distance_between_points(p, q):
    """
    计算两点之间的欧氏距离
    参数:        p, q : array_like            三维坐标 (x, y, z)
    返回:        float : 两点间距离
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.linalg.norm(p - q)




def dist_between(v1, v2):
    """ Returns the l2-norm distance between vectors 'v1' and 'v2'::
    """
    return np.linalg.norm(v1-v2)


def area_of_triangle(v1, v2, v3):
    a = np.linalg.norm(v1 - v2)
    b = np.linalg.norm(v2 - v3)
    c = np.linalg.norm(v3 - v1)
    s = (a+b+c) / 2.
    return np.sqrt(s * (s-a) * (s-b) * (s-c))


def get_joints(g, s, t):
    root = g[s, t, 0, :]
    spine = g[s, t, 1, :] - root
    neck = g[s, t, 2, :] - root
    head = g[s, t, 3, :] - root
    rshoulder = g[s, t, 4, :] - root
    relbow = g[s, t, 5, :] - root
    rhand = g[s, t, 6, :] - root
    lshoulder = g[s, t, 7, :] - root
    lelbow = g[s, t, 8, :] - root
    lhand = g[s, t, 9, :] - root
    rhip = g[s, t, 10, :] - root
    rknee = g[s, t, 11, :] - root
    rfoot = g[s, t, 12, :] - root
    lhip = g[s, t, 13, :] - root
    lknee = g[s, t, 14, :] - root
    lfoot = g[s, t, 15, :] - root
    root = root - root
    return root, spine, neck, head, rshoulder, relbow, rhand,\
           lshoulder, lelbow, lhand, rhip, rknee, rfoot, lhip, lknee, lfoot


def get_angles(f, s, t, fidx):
    root = f[s, t, fidx + 0]
    spine = f[s, t, fidx + 1]
    neck = f[s, t, fidx + 2]
    head = f[s, t, fidx + 3]
    rshoulder = f[s, t, fidx + 4]
    relbow = f[s, t, fidx + 5]
    rhand = f[s, t, fidx + 6]
    lshoulder = f[s, t, fidx + 7]
    lelbow = f[s, t, fidx + 8]
    lhand = f[s, t, fidx + 9]
    rhip = f[s, t, fidx + 10]
    rknee = f[s, t, fidx + 11]
    rfoot = f[s, t, fidx + 12]
    lhip = f[s, t, fidx + 13]
    lknee = f[s, t, fidx + 14]
    lfoot = f[s, t, fidx + 15]

    return root, spine, neck, head, rshoulder, relbow, rhand,\
           lshoulder, lelbow, lhand, rhip, rknee, rfoot, lhip, lknee, lfoot



def get_velocity(pos_curr, pos_prev):
    # 毫米 -> 米，防 overflow
    pos_curr = pos_curr.astype(np.float64) #/ 1000.0
    pos_prev = pos_prev.astype(np.float64) #/ 1000.0

    vel = pos_curr - pos_prev
    return sanitize_array(np.append(vel, np.linalg.norm(vel)), name="get_velocity")



def get_acceleration(vel_curr, vel_prev):
    acc = vel_curr - vel_prev
    return sanitize_array(np.append(acc, np.linalg.norm(acc)), name="get_acceleration")


'''
def get_dynamics(pos_curr, pos_prev, vel_prev, acc_prev=None):
    vel_curr = get_velocity(pos_curr, pos_prev)
    acc_curr = get_acceleration(vel_curr[:-1], vel_prev)
    if acc_prev is None:
        return np.concatenate((vel_curr, acc_curr))
    return np.concatenate((vel_curr, acc_curr))
'''

def get_dynamics(pos_curr, pos_prev, vel_prev, acc_prev=None):
    """
    获取动力学特征，包含完善的输入验证和错误处理
    参数:
        pos_curr: 当前时刻位置 (x, y, z)
        pos_prev: 前一时刻位置 (x, y, z)
        vel_prev: 前一时刻速度 (vx, vy, vz) 或 (vx, vy, vz, norm)
        acc_prev: 前一时刻加速度 (可选)
    返回: (vx, vy, vz, v_norm, ax, ay, az, a_norm) 共8个特征
    """

    # ================== 1. 输入验证和清理 ==================
    debug_info = {}

    # 检查当前位置
    if pos_curr is None:
        print("🚨 get_dynamics: pos_curr is None")
        pos_curr = np.zeros(3, dtype=np.float64)

    if pos_prev is None:
        print("🚨 get_dynamics: pos_prev is None")
        pos_prev = np.zeros(3, dtype=np.float64)

    # 转换为numpy数组
    pos_curr = np.asarray(pos_curr, dtype=np.float64).flatten()
    pos_prev = np.asarray(pos_prev, dtype=np.float64).flatten()

    # 确保维度正确
    if len(pos_curr) != 3:
        print(f"⚠️ get_dynamics: pos_curr has wrong shape {pos_curr.shape}, expected 3")
        pos_curr = np.zeros(3, dtype=np.float64)

    if len(pos_prev) != 3:
        print(f"⚠️ get_dynamics: pos_prev has wrong shape {pos_prev.shape}, expected 3")
        pos_prev = np.zeros(3, dtype=np.float64)

    # 检测NaN和Inf
    nan_mask_curr = np.isnan(pos_curr)
    nan_mask_prev = np.isnan(pos_prev)
    inf_mask_curr = np.isinf(pos_curr)
    inf_mask_prev = np.isinf(pos_prev)

    if np.any(nan_mask_curr) or np.any(nan_mask_prev):
        nan_count = np.sum(nan_mask_curr) + np.sum(nan_mask_prev)
        print(f"⚠️ get_dynamics: Detected {nan_count} NaN values in positions")
        print(f"    pos_curr NaN positions: {np.where(nan_mask_curr)[0]}")
        print(f"    pos_prev NaN positions: {np.where(nan_mask_prev)[0]}")

        # 记录原始值用于调试
        debug_info['pos_curr_original'] = pos_curr.copy()
        debug_info['pos_prev_original'] = pos_prev.copy()

        # 清理NaN
        #pos_curr = np.nan_to_num(pos_curr, nan=0.0)
        #pos_prev = np.nan_to_num(pos_prev, nan=0.0)

    if np.any(inf_mask_curr) or np.any(inf_mask_prev):
        inf_count = np.sum(inf_mask_curr) + np.sum(inf_mask_prev)
        print(f"⚠️ get_dynamics: Detected {inf_count} Inf values in positions")

        # 处理Inf
        pos_curr = np.where(np.isinf(pos_curr), np.sign(pos_curr) * 1e6, pos_curr)
        pos_prev = np.where(np.isinf(pos_prev), np.sign(pos_prev) * 1e6, pos_prev)

    # 检测零向量或极小值
    norm_curr = np.linalg.norm(pos_curr)
    norm_prev = np.linalg.norm(pos_prev)


    # 检测过大值
    max_abs_curr = np.abs(pos_curr).max()
    max_abs_prev = np.abs(pos_prev).max()

    if max_abs_curr > 1e9:
        print(f"⚠️ get_dynamics: pos_curr has large values (max_abs={max_abs_curr:.2e})")
        print(pos_curr)
        pos_curr = pos_curr / (max_abs_curr / 1e6)
        debug_info['scaled_curr'] = True

    if max_abs_prev > 1e9:
        print(f"⚠️ get_dynamics: pos_prev has large values (max_abs={max_abs_prev:.2e})")
        print(pos_prev)
        pos_prev = pos_prev / (max_abs_prev / 1e6)
        debug_info['scaled_prev'] = True

    # ================== 2. 前一时刻速度验证 ==================
    if vel_prev is not None:
        vel_prev = np.asarray(vel_prev, dtype=np.float64).flatten()

        # 检查维度：可能是3维或4维（包含norm）
        if len(vel_prev) == 4:
            vel_prev_vec = vel_prev[:3]
            vel_prev_norm = vel_prev[3]
        elif len(vel_prev) == 3:
            vel_prev_vec = vel_prev
            vel_prev_norm = np.linalg.norm(vel_prev_vec)
        else:
            print(f"⚠️ get_dynamics: vel_prev has wrong shape {vel_prev.shape}, expected 3 or 4")
            vel_prev_vec = np.zeros(3, dtype=np.float64)
            vel_prev_norm = 0.0

        # 清理vel_prev
        if np.any(np.isnan(vel_prev_vec)) or np.any(np.isinf(vel_prev_vec)):
            print(f"⚠️ get_dynamics: vel_prev contains NaN/Inf")
            vel_prev_vec = np.nan_to_num(vel_prev_vec, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        vel_prev_vec = np.zeros(3, dtype=np.float64)
        vel_prev_norm = 0.0

    # ================== 3. 计算当前速度 ==================
    try:
        # 毫米转米（如果原始数据是毫米）
        pos_curr_m = pos_curr #/ 1000.0
        pos_prev_m = pos_prev #/ 1000.0

        # 计算速度向量
        vel_curr_vec = pos_curr_m - pos_prev_m

        # 计算速度模长
        vel_curr_norm_sq = np.sum(vel_curr_vec ** 2)
        if vel_curr_norm_sq < 0:
            vel_curr_norm = 0.0
        else:
            vel_curr_norm = np.sqrt(vel_curr_norm_sq)

        # 防止数值不稳定
        if vel_curr_norm_sq < 1e-24:  # 非常小的速度
            vel_curr_vec = np.zeros_like(vel_curr_vec)
            vel_curr_norm = 0.0
        elif vel_curr_norm > 1e6:  # 过大的速度
            print(f"⚠️ get_dynamics: Large velocity detected ({vel_curr_norm:.2e})")
            vel_curr_vec = vel_curr_vec / (vel_curr_norm / 1e3)
            vel_curr_norm = np.sqrt(np.sum(vel_curr_vec ** 2))

        # 组合速度特征
        vel_curr = np.append(vel_curr_vec, vel_curr_norm)

    except Exception as e:
        print(f"🚨 get_dynamics: Velocity calculation failed: {e}")
        vel_curr = np.zeros(4, dtype=np.float64)

    # ================== 4. 计算当前加速度 ==================
    try:
        # 计算加速度向量
        acc_curr_vec = vel_curr[:3] - vel_prev_vec

        # 计算加速度模长
        acc_curr_norm_sq = np.sum(acc_curr_vec ** 2)
        if acc_curr_norm_sq < 0:
            acc_curr_norm = 0.0
        else:
            acc_curr_norm = np.sqrt(acc_curr_norm_sq)

        # 防止数值不稳定
        if acc_curr_norm_sq < 1e-24:  # 非常小的加速度
            acc_curr_vec = np.zeros_like(acc_curr_vec)
            acc_curr_norm = 0.0
        elif acc_curr_norm > 1e6:  # 过大的加速度
            print(f"⚠️ get_dynamics: Large acceleration detected ({acc_curr_norm:.2e})")
            acc_curr_vec = acc_curr_vec / (acc_curr_norm / 1e3)
            acc_curr_norm = np.sqrt(np.sum(acc_curr_vec ** 2))

        # 组合加速度特征
        acc_curr = np.append(acc_curr_vec, acc_curr_norm)

    except Exception as e:
        print(f"🚨 get_dynamics: Acceleration calculation failed: {e}")
        acc_curr = np.zeros(4, dtype=np.float64)

    # ================== 5. 最终验证和清理 ==================
    # 合并所有特征
    result = np.concatenate((vel_curr, acc_curr))

    # 最终NaN/Inf检查
    nan_mask = np.isnan(result)
    inf_mask = np.isinf(result)

    if np.any(nan_mask) or np.any(inf_mask):
        nan_count = np.sum(nan_mask)
        inf_count = np.sum(inf_mask)
        print(f"🚨 get_dynamics: Result contains NaN={nan_count}, Inf={inf_count}")
        print(f"    Result shape: {result.shape}")
        print(f"    NaN positions: {np.where(nan_mask)[0]}")
        print(f"    Inf positions: {np.where(inf_mask)[0]}")

        # 强制清理
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    # 检查特征范围
    result_abs_max = np.abs(result).max()
    if result_abs_max > 1e6:
        print(f"⚠️ get_dynamics: Result has large values (max_abs={result_abs_max:.2e})")
        result = result / (result_abs_max / 1e3)

    # ================== 6. 返回结果 ==================
    # 可选：记录调试信息
    if debug_info:
        debug_info['final_result_range'] = (result.min(), result.max())
        debug_info['final_result_mean'] = result.mean()
        # 可以将debug_info保存或记录到日志

    return result





def get_velocity_angle(angle_curr, angle_prev):
    angle_vel = angle_curr - angle_prev
    return sanitize_array(angle_vel, name="get_velocity_angle")


def get_acceleration_angle(angle_vel_curr, angle_vel_prev):
    angle_acc = angle_vel_curr - angle_vel_prev
    return sanitize_array(angle_acc, name="get_acceleration_angle")

def get_dynamics_angle(angle_curr, angle_prev, angle_vel_prev):
    angle_vel_curr = get_velocity_angle(angle_curr, angle_prev)
    angle_acc_curr = get_acceleration_angle(angle_vel_curr, angle_vel_prev)

    return np.array([angle_vel_curr, angle_acc_curr])







'''
def get_dynamics_relative(i, f, s, t, fidx):
    N = 4
    affective_features = np.zeros(N * 14)
    if i == 0:
        return np.random.normal(0, 1e-6, N * 14)
    else:
        f_i = f[s, t, (fidx + N * i):(fidx + N * i) + N]

        idx = 0
        for j in range(1, 16):
            if j != i:
                f_j = f[s, t, (fidx + N * j):(fidx + N * j) + N]

                # 使用对数比值：log(f_j / f_i) = log(f_j) - log(f_i)
                # 首先确保所有值为正，添加小的偏移量
                f_i_safe = np.sign(f_i) * np.maximum(np.abs(f_i), 1e-8)
                f_j_safe = np.sign(f_j) * np.maximum(np.abs(f_j), 1e-8)

                # 计算对数比值
                log_ratio = np.log(np.abs(f_j_safe) + 1e-8) - np.log(np.abs(f_i_safe) + 1e-8)
                # 保留符号信息
                sign_info = np.sign(f_j_safe) * np.sign(f_i_safe)

                affective_features[idx:idx + N] = log_ratio * sign_info
                idx += N

    return affective_features
'''

def get_dynamics_relative_bk(i, f, s, t, fidx, N):
    """
    计算节点 i 相对于根节点(0)的帧内特征，保留符号信息。
    返回: shape (N,) 的相对特征向量
    """
    affective_features = np.zeros(N, dtype=np.float64)
    if i == 15:
        return affective_features

    #  正确获取节点 i 的特征
    f_i   = f[s, t, fidx + i * N : fidx + (i + 1) * N]
    f_15  = f[s, t, fidx + 15 * N : fidx + (15 + 1) * N]# 根节点 15 的特征

    # 清洗 NaN/Inf
    f_i = np.nan_to_num(f_i, nan=0.0, posinf=0.0, neginf=0.0)
    f_15= np.nan_to_num(f_15, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-8
    f_i_safe = np.where(f_i == 0, eps, f_i)
    sign_i = np.sign(f_i_safe)

    f_i_p = np.maximum(np.abs(f_i), eps)
    f_15_p = np.maximum(np.abs(f_15), eps)
    log_diff = np.log(f_i_p + eps) - np.log(f_15_p + eps)

    affective_features[:] = sign_i * log_diff

    return sanitize_array(affective_features, name="get_dynamics_relative")

def get_dynamics_relative(i, f, s, t, fidx, N):
    """
    计算节点 i 相对于根节点(0)的有界归一化差分特征。
    返回: shape (N,) 的相对特征向量，范围 [-1, 1]
    """
    affective_features = np.zeros(N, dtype=np.float64)
    if i == 0:          # 根节点自身相对特征置零
        return affective_features

    # 获取节点 i 和根节点 0 的特征
    f_i = f[s, t, fidx + i * N : fidx + (i + 1) * N]
    f_0 = f[s, t, fidx + 0 * N : fidx + (0 + 1) * N]   # 根节点索引 0

    # 清洗 NaN/Inf
    f_i = np.nan_to_num(f_i, nan=0.0, posinf=0.0, neginf=0.0)
    f_0 = np.nan_to_num(f_0, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-8
    # 有界归一化差分: (a - b) / (|a| + |b| + eps)
    denom = np.abs(f_i) + np.abs(f_0) + eps
    rel_diff = (f_i - f_0) / denom

    affective_features[:] = rel_diff

    return sanitize_array(affective_features, name="get_dynamics_relative")



def get_affective_features(gaits):
    # 0: root, 1: spine, 2: neck, 3: head
    # 4: rshoulder, 5: relbow, 6: rhand
    # 7: lshoulder, 8: lelbow, 9: lhand
    # 10: rhip, 11: rknee, 12: rfoot
    # 13: lhip, 14: lknee, 15: lfoot

    # 清洗整个输入
    #gaits = np.nan_to_num(gaits, nan=0.0, posinf=0.0, neginf=0.0)

    num_samples = gaits.shape[0]
    num_tsteps = gaits.shape[1]
    num_features = 224+160
    up_vector = np.array([0, 1, 0])
    affective_features = np.zeros((num_samples, num_tsteps, num_features))
    #Y = np.array(get_joints(gaits, 0, 0)).transpose()


    Y = np.array(
        [[0.0, -8.40212115e-03, -9.53851533e-03, -4.03871819e-03, -7.52738843e-02, -2.81678708e-01, -4.17936960e-01,
          7.96288259e-02, 2.76516953e-01, 4.11510516e-01, -6.74779788e-02, -1.01493726e-01, -1.10780176e-01,
          5.77831078e-02, 7.79183409e-02, 8.74310686e-02],
         [0.0, 1.20554661e-01, 2.27028674e-01, 2.97235946e-01, 2.26821076e-01, 1.68745608e-01, 1.50716715e-01,
          2.08176310e-01, 1.58846755e-01, 1.27152295e-01, -6.93229158e-02, -3.71018802e-01, -6.64287975e-01,
          -7.96507542e-02, -3.71755971e-01, -6.69173159e-01],
         [0.0, 2.61917976e-02, 2.57063591e-02, 8.99503207e-03, -6.27215921e-02, -5.69554221e-02, -2.57920354e-02,
          -5.61114550e-02, -7.55548450e-02, -4.25150186e-02, 8.19714895e-03, 2.70399580e-02, -6.41822474e-02,
          -3.39123402e-04, 7.18219105e-03, -6.64537982e-02]])


    for sidx in range(num_samples):
        #if sidx%100 == 0:
        #    print("sidx: ", sidx)
        X = np.array(get_joints(gaits, sidx, 0)).transpose()
        R, c, t = get_transformation(X, Y)

        if R.shape != (3, 3):
            print(f"Sample {sidx}: R shape {R.shape} != (3,3), skipping")
            continue
            # 修正：c是标量，直接打印值
        #print(f"Sample {sidx}: R shape={R.shape}, c={c}, t shape={t.shape}")

        for tidx in range(num_tsteps):
            fidx = 0
            Xtx = np.array(get_joints(gaits, sidx, tidx)).transpose()
            affective_features[sidx, tidx, fidx:fidx+48] = (np.dot(c*R, Xtx) + np.tile(
                np.reshape(t, (t.shape[0], 1)), (1, Xtx.shape[1]))).transpose().flatten()
            fidx += 48

            # within frame  16f
            j_0, j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9, j_10, j_11, j_12, j_13, j_14, j_15 = get_joints(gaits, sidx, tidx)
            affective_features[sidx, tidx, fidx] = angle_between_points(j_10, j_0, j_13)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_0, j_1, j_2)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_4, j_2, j_7)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_1, j_2, j_3) #3 head
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_2, j_4, j_5)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_4, j_5, j_6)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_2, j_6, j_9) #6 hand
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_2, j_7, j_8)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_7, j_8, j_9)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_2, j_9, j_6) #9 hand
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_0, j_10, j_11)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_10, j_11, j_12)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_0, j_12, j_15) #foot
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_0, j_13, j_14)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_13, j_14, j_15)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(j_0, j_15, j_12) #foot
            fidx += 1

            # between frames 160
            if tidx > 0:

                current_joints = list(get_joints(gaits, sidx, tidx))
                prev_joints = list(get_joints(gaits, sidx, tidx-1))


                current_angles = list(get_angles(affective_features, sidx, tidx, fidx-16))
                prev_angles = list(get_angles(affective_features, sidx, tidx-1, fidx-16))

                ############################## Linear velocity ###############################
                for i in range(16):
                    affective_features[sidx, tidx, fidx:fidx + 8] \
                        = get_dynamics(current_joints[i], prev_joints[i], affective_features[sidx, tidx - 1, fidx:fidx + 3]
                    )
                    fidx += 8
                ############################## angular velocity ###############################
                for i in range(16):
                    affective_features[sidx, tidx, fidx:fidx + 2] \
                        = get_dynamics_angle(current_angles[i], prev_angles[i], affective_features[sidx, tidx - 1, fidx]
                    )
                    fidx += 2

                ############################## Intra-frame relative Linear velocity 16*8=128 ###############################
                fidx_base = fidx
                for i in range(16):
                    affective_features[sidx, tidx, fidx:fidx + 8] \
                        = get_dynamics_relative(i, affective_features, sidx, tidx, fidx_base-160, 8)
                    fidx += 8

                ############################## Intra-frame relative angular velocity 16*2=32 ###############################
                fidx_base = fidx
                for i in range(16):
                    affective_features[sidx, tidx, fidx:fidx + 2] \
                        = get_dynamics_relative(i, affective_features, sidx, tidx, fidx_base-160, 2)
                    fidx += 2





    return affective_features


def sanitize_array(arr, name="array"):
    """
    防御性清洗：NaN/Inf → 0
    返回清洗后的副本，并打印警告
    """
    if np.isnan(arr).any() or np.isinf(arr).any():
        print(f"⚠️ NaN/Inf detected in {name}, replacing with zeros")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def get_transformation(X, Y):
    """

    Args:
        X: k x n source shape
        Y: k x n destination shape such that Y[:, i] is the correspondence of X[:, i]

    Returns: rotation R, scaling c, translation t such that ||Y - (cRX+t)||_2 is minimized.

    """
    """
    Copyright: Carlo Nicolini, 2013
    Code adapted from the Mark Paskin Matlab version
    from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 
    """

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    M = np.dot(Yc, Xc.T) / n

    U, D, V = np.linalg.svd(M, full_matrices=True, compute_uv=True)
    V = V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.linalg.matrix_rank(M)
    d = np.linalg.det(M)
    S = np.eye(m)
    if r > (m - 1):
        if np.linalg.det(M) < 0:
            S[m - 1, m - 1] = -1
    elif r == m - 1:
        if np.linalg.det(U) * np.linalg.det(V) < 0:
            S[m - 1, m - 1] = -1
    else:
        R = np.eye(2)
        c = 1
        t = np.zeros(2)
        return R, c, t

    R = np.dot(np.dot(U, S), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R, c, t



def get_affective_features_optimized(gaits):
    num_samples, num_tsteps, n_joints, _ = gaits.shape
    affective_features = np.zeros((num_samples, num_tsteps, 384), dtype=np.float64)

    angle_pairs = np.array([
        (10, 0, 13), (0, 1, 2), (4, 2, 7), (1, 2, 3), (2, 4, 5), (4, 5, 6),
        (2, 6, 9), (2, 7, 8), (7, 8, 9), (2, 9, 6), (0, 10, 11), (10, 11, 12),
        (0, 12, 15), (0, 13, 14), (13, 14, 15), (0, 15, 12)
    ], dtype=np.int32)

    Y = np.array([
        [0.0, -8.40212115e-03, -9.53851533e-03, -4.03871819e-03, -7.52738843e-02,
         -2.81678708e-01, -4.17936960e-01, 7.96288259e-02, 2.76516953e-01, 4.11510516e-01,
         -6.74779788e-02, -1.01493726e-01, -1.10780176e-01, 5.77831078e-02, 7.79183409e-02, 8.74310686e-02],
        [0.0, 1.20554661e-01, 2.27028674e-01, 2.97235946e-01, 2.26821076e-01, 1.68745608e-01,
         1.50716715e-01, 2.08176310e-01, 1.58846755e-01, 1.27152295e-01, -6.93229158e-02,
         -3.71018802e-01, -6.64287975e-01, -7.96507542e-02, -3.71755971e-01, -6.69173159e-01],
        [0.0, 2.61917976e-02, 2.57063591e-02, 8.99503207e-03, -6.27215921e-02, -5.69554221e-02,
         -2.57920354e-02, -5.61114550e-02, -7.55548450e-02, -4.25150186e-02, 8.19714895e-03,
         2.70399580e-02, -6.41822474e-02, -3.39123402e-04, 7.18219105e-03, -6.64537982e-02]
    ], dtype=np.float64)

    # 预计算相对坐标
    abs_joints = gaits
    root = abs_joints[:, :, 0, :]
    rel_joints = abs_joints - root[:, :, None, :]  # (S, T, 16, 3)

    for sidx in range(num_samples):
        #if sidx%100 == 0:
        #    print("sidx: ", sidx)
        # 变换特征
        X0 = rel_joints[sidx, 0].T
        R, c, t = get_transformation(X0, Y)
        if R.shape != (3, 3):
            continue
        rel_T = rel_joints[sidx].transpose(0, 2, 1)
        transformed = c * (R @ rel_T) + t[:, None]
        transformed_flat = transformed.transpose(0, 2, 1).reshape(num_tsteps, -1)
        affective_features[sidx, :, :48] = transformed_flat

        # 角度特征：逐关节对计算（避免向量化维度错误）
        angle = np.zeros((num_tsteps, 16), dtype=np.float64)
        for idx, (p1, p2, p3) in enumerate(angle_pairs):
            v1 = rel_joints[sidx, :, p1, :] - rel_joints[sidx, :, p2, :]  # (T, 3)
            v2 = rel_joints[sidx, :, p3, :] - rel_joints[sidx, :, p2, :]
            norm1 = np.linalg.norm(v1, axis=1)
            norm2 = np.linalg.norm(v2, axis=1)
            # 计算角度
            cos_angle = np.clip(np.sum(v1 * v2, axis=1) / (norm1 * norm2 + 1e-12), -1.0, 1.0)
            # 原始 angle_between_points 对零向量返回 pi/2
            ang = np.full(num_tsteps, np.pi/2.0, dtype=np.float64)
            valid = (norm1 > 1e-8) & (norm2 > 1e-8)
            ang[valid] = np.arccos(cos_angle[valid])
            angle[:, idx] = ang
        affective_features[sidx, :, 48:64] = angle

        # 帧间特征
        for tidx in range(1, num_tsteps):
            curr_joints = rel_joints[sidx, tidx]
            prev_joints = rel_joints[sidx, tidx-1]
            curr_angles = angle[tidx]
            prev_angles = angle[tidx-1]

            prev_linear = affective_features[sidx, tidx-1, 64:192].reshape(16, 8)
            prev_angular = affective_features[sidx, tidx-1, 192:224].reshape(16, 2)

            base_linear = 64
            for i in range(16):
                vel_prev = prev_linear[i, :3]
                dyn = get_dynamics(curr_joints[i], prev_joints[i], vel_prev)
                affective_features[sidx, tidx, base_linear + i*8 : base_linear + (i+1)*8] = dyn

            base_angular = 192
            for i in range(16):
                angle_vel_prev = prev_angular[i, 0]
                angle_dyn = get_dynamics_angle(curr_angles[i], prev_angles[i], angle_vel_prev)
                affective_features[sidx, tidx, base_angular + i*2 : base_angular + (i+1)*2] = angle_dyn

            base_rel_linear = 224
            for i in range(16):
                rel_dyn = get_dynamics_relative(i, affective_features, sidx, tidx, 64, 8)
                affective_features[sidx, tidx, base_rel_linear + i*8 : base_rel_linear + (i+1)*8] = rel_dyn

            base_rel_angular = 352
            for i in range(16):
                rel_ang = get_dynamics_relative(i, affective_features, sidx, tidx, 192, 2)
                affective_features[sidx, tidx, base_rel_angular + i*2 : base_rel_angular + (i+1)*2] = rel_ang

    return affective_features


def get_affective_features_optimized_v2(gaits):
    """
    纯 NumPy 向量化版本，完全复刻 loader0_1.py 中 get_affective_features_batch 逻辑。
    输入: gaits, shape (N, T, 16, 3)  坐标 (毫米)
    输出: affective_features, shape (N, T, 384)
    所有特征均基于 Procrustes 对齐后的坐标，根节点为0，相对特征使用有界归一化差分。
    """
    num_samples, num_tsteps, n_joints, _ = gaits.shape
    affective_features = np.zeros((num_samples, num_tsteps, 384), dtype=np.float64)

    # 预定义角度对 (16 组)
    angle_pairs = np.array([
        (10, 0, 13), (0, 1, 2), (4, 2, 7), (1, 2, 3), (2, 4, 5), (4, 5, 6),
        (2, 6, 9), (2, 7, 8), (7, 8, 9), (2, 9, 6), (0, 10, 11), (10, 11, 12),
        (0, 12, 15), (0, 13, 14), (13, 14, 15), (0, 15, 12)
    ], dtype=np.int32)

    Y = np.array([
        [0.0, -8.40212115e-03, -9.53851533e-03, -4.03871819e-03, -7.52738843e-02,
         -2.81678708e-01, -4.17936960e-01, 7.96288259e-02, 2.76516953e-01, 4.11510516e-01,
         -6.74779788e-02, -1.01493726e-01, -1.10780176e-01, 5.77831078e-02, 7.79183409e-02, 8.74310686e-02],
        [0.0, 1.20554661e-01, 2.27028674e-01, 2.97235946e-01, 2.26821076e-01, 1.68745608e-01,
         1.50716715e-01, 2.08176310e-01, 1.58846755e-01, 1.27152295e-01, -6.93229158e-02,
         -3.71018802e-01, -6.64287975e-01, -7.96507542e-02, -3.71755971e-01, -6.69173159e-01],
        [0.0, 2.61917976e-02, 2.57063591e-02, 8.99503207e-03, -6.27215921e-02, -5.69554221e-02,
         -2.57920354e-02, -5.61114550e-02, -7.55548450e-02, -4.25150186e-02, 8.19714895e-03,
         2.70399580e-02, -6.41822474e-02, -3.39123402e-04, 7.18219105e-03, -6.64537982e-02]
    ], dtype=np.float64)

    # 相对坐标（根节点0）
    root = gaits[:, :, 0, :]
    rel_joints = gaits - root[:, :, None, :]  # (N, T, 16, 3)

    # 提前提取角度对索引
    p1 = angle_pairs[:, 0]
    p2 = angle_pairs[:, 1]
    p3 = angle_pairs[:, 2]

    eps = 1e-8  # 用于数值稳定

    for sidx in range(num_samples):
        # ---- Procrustes 对齐 ----
        X0 = rel_joints[sidx, 0].T  # (3, 16)
        R, c, t = get_transformation(X0, Y)
        if R.shape != (3, 3):
            continue

        # 应用变换到所有帧
        rel_T = rel_joints[sidx].transpose(0, 2, 1)  # (T, 3, 16)
        transformed = c * (R @ rel_T) + t[:, None]   # (T, 3, 16)
        transformed = transformed.transpose(0, 2, 1) # (T, 16, 3)
        transformed_flat = transformed.reshape(num_tsteps, -1)  # (T, 48)
        affective_features[sidx, :, :48] = transformed_flat

        # ---- 角度特征 (48~63) 基于 transformed ----
        v1 = transformed[:, p1, :] - transformed[:, p2, :]  # (T, 16, 3)
        v2 = transformed[:, p3, :] - transformed[:, p2, :]  # (T, 16, 3)
        norm1 = np.linalg.norm(v1, axis=2)   # (T, 16)  对坐标轴求范数
        norm2 = np.linalg.norm(v2, axis=2)   # (T, 16)
        dot = np.sum(v1 * v2, axis=2)        # (T, 16)
        cos_angle = np.clip(dot / (norm1 * norm2 + eps), -1.0, 1.0)
        angle = np.full((num_tsteps, 16), np.pi/2.0, dtype=np.float64)
        valid = (norm1 > 0) & (norm2 > 0)    # (T, 16)
        angle[valid] = np.arccos(cos_angle[valid])  # 一维索引填充，合法
        affective_features[sidx, :, 48:64] = angle

        # ---- 帧间动力学 (基于 transformed) ----
        for tidx in range(1, num_tsteps):
            curr_joints = transformed[tidx]      # (16, 3)
            prev_joints = transformed[tidx-1]    # (16, 3)
            curr_angles = angle[tidx]            # (16,)
            prev_angles = angle[tidx-1]          # (16,)

            # 线性动力学 (64~191)
            vel_vec = curr_joints - prev_joints   # (16, 3)
            vel_norm = np.linalg.norm(vel_vec, axis=1, keepdims=True)  # (16,1)
            vel_feat = np.concatenate([vel_vec, vel_norm], axis=1)     # (16,4)

            prev_linear = affective_features[sidx, tidx-1, 64:192].reshape(16, 8)
            prev_vel_vec = prev_linear[:, :3]      # (16,3)

            acc_vec = vel_vec - prev_vel_vec
            acc_norm = np.linalg.norm(acc_vec, axis=1, keepdims=True)
            acc_feat = np.concatenate([acc_vec, acc_norm], axis=1)     # (16,4)
            lin_dyn = np.concatenate([vel_feat, acc_feat], axis=1)     # (16,8)
            affective_features[sidx, tidx, 64:192] = lin_dyn.reshape(-1)

            # 角度动力学 (192~223)
            angle_vel = curr_angles - prev_angles   # (16,)
            prev_angular = affective_features[sidx, tidx-1, 192:224].reshape(16, 2)
            prev_angle_vel = prev_angular[:, 0]     # (16,)
            angle_acc = angle_vel - prev_angle_vel
            ang_dyn = np.stack([angle_vel, angle_acc], axis=1)  # (16,2)
            affective_features[sidx, tidx, 192:224] = ang_dyn.reshape(-1)

            # 相对线性 (224~351)  有界归一化差分，根节点0
            root_lin = lin_dyn[0:1, :]   # (1,8)
            denom_lin = np.abs(lin_dyn) + np.abs(root_lin) + eps
            rel_lin = (lin_dyn - root_lin) / denom_lin
            rel_lin[0, :] = 0.0   # 根节点自身置零
            affective_features[sidx, tidx, 224:352] = rel_lin.reshape(-1)

            # 相对角度 (352~383)
            root_ang = ang_dyn[0:1, :]   # (1,2)
            denom_ang = np.abs(ang_dyn) + np.abs(root_ang) + eps
            rel_ang = (ang_dyn - root_ang) / denom_ang
            rel_ang[0, :] = 0.0
            affective_features[sidx, tidx, 352:384] = rel_ang.reshape(-1)

    return affective_features



test_data = np.random.randn(2, 10, 16, 3)  # 8样本,10帧
orig = get_affective_features_optimized_v2(test_data)   # 原始函数（需先定义）
opt = get_affective_features_optimized(test_data)
#print(orig)
#print(opt)
#np.testing.assert_allclose(orig, opt, rtol=1e-5, atol=1e-6, verbose=True)
#print(np.allclose(orig, opt, atol=1e-6))   # 应输出 True

def compare_blocks(a, b):
    blocks = [
        ("Procrustes", 0, 48),
        ("Angles", 48, 64),
        ("Linear Dyn", 64, 192),
        ("Angular Dyn", 192, 224),
        ("Relative Linear", 224, 352),
        ("Relative Angular", 352, 384),
    ]
    for name, s, e in blocks:
        diff = a[..., s:e] - b[..., s:e]
        print(f"{name:>16}: max_abs={np.max(np.abs(diff)):.2e}, mean_abs={np.mean(np.abs(diff)):.2e}")

#compare_blocks(orig, opt)
