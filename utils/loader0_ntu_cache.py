# ==================== loader0_ntu_cache.py (修改版) ====================
import h5py
import os
import numpy as np
import sys
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
from torch.utils.data import Dataset



from utils import common

# ================== 1. 修改 load_data_ntu_cache ==================
def load_data_ntu_cache(_path, _ftype_ntu, coords, joints, cycles=1,
                        use_cache=True, cache_dir=None, exit_after_cache=True):

    if cache_dir is None:
        cache_dir = os.path.join(_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    cache_name = f"ntu_raw_{_ftype_ntu}_{coords}_{joints}_{cycles}.npz"
    cache_name = cache_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    cache_file = os.path.join(cache_dir, cache_name)

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached raw data from {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        data = cached['data']                     # (N, max_len, J*C)
        labels_ntu = cached['labels_ntu']
        original_lengths = cached['original_lengths'].tolist()
        # Filter categories 49~59
        skip_classes = list(range(49, 60))
        keep_mask = ~np.isin(labels_ntu, skip_classes)
        data = data[keep_mask]
        labels_ntu = labels_ntu[keep_mask]
        original_lengths = [original_lengths[i] for i in range(len(original_lengths)) if keep_mask[i]]
        #
        data_train, data_test, labels_train, labels_test, lengths_train, lengths_test = train_test_split(
            data, labels_ntu, original_lengths, test_size=0.1, stratify=labels_ntu)

        return (data, labels_ntu, data_train, labels_train,
                data_test, labels_test, original_lengths,
                lengths_train, lengths_test)

    # --------------------
    print("Processing NTU raw data from scratch (no cache found)...")
    file_feature_ntu = os.path.join(_path, 'features' + _ftype_ntu + '.h5')
    ffs = h5py.File(file_feature_ntu, 'r')
    file_label_ntu = os.path.join(_path, 'labels' + _ftype_ntu + '.h5')
    fls = h5py.File(file_label_ntu, 'r')

    ffs_keys = list(ffs.keys())
    fls_keys = list(fls.keys())
    num_samples_ntu = len(ffs_keys)
    print("num_samples_ntu:", num_samples_ntu)

    labels_ntu = np.empty(num_samples_ntu)
    original_lengths = np.zeros(num_samples_ntu, dtype=np.int32)
    data_list_ntu = [None] * num_samples_ntu

    for si in range(num_samples_ntu):
        if si % 1000 == 0:
            print(f"Loading sample {si}/{num_samples_ntu}")
        ffs_group_key = ffs_keys[si]
        data_list_ntu[si] = ffs[ffs_group_key][:]  # (T, J*C)
        labels_ntu[si] = fls[fls_keys[si]][()]
        original_lengths[si] = len(data_list_ntu[si])

    max_original_len = original_lengths.max()
    data = np.zeros((num_samples_ntu, max_original_len, joints * coords))
    for si in range(num_samples_ntu):
        seq = data_list_ntu[si]
        actual_len = original_lengths[si]
        data[si, :actual_len, :] = seq[:actual_len]

    del data_list_ntu

    skip_classes = list(range(49, 60))
    keep_mask = ~np.isin(labels_ntu, skip_classes)
    data = data[keep_mask]
    labels_ntu = labels_ntu[keep_mask]
    original_lengths = [original_lengths[i] for i in range(len(original_lengths)) if keep_mask[i]]
    num_samples = len(labels_ntu)
    print(f"After filtering classes {skip_classes}: {num_samples} samples remaining")


    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.1, stratify=labels_ntu)

    data_train = data[train_idx]
    labels_train = labels_ntu[train_idx]
    lengths_train = [original_lengths[i] for i in train_idx]

    data_test = data[test_idx]
    labels_test = labels_ntu[test_idx]
    lengths_test = [original_lengths[i] for i in test_idx]

    # ========== Save original coordinates to cache ==========
    if exit_after_cache:
        print(f"Saving raw data cache to {cache_file}")
        np.savez_compressed(cache_file,
                            data=data,
                            labels_ntu=labels_ntu,
                            data_train=data_train,
                            labels_train=labels_train,
                            data_test=data_test,
                            labels_test=labels_test,
                            original_lengths=original_lengths,
                            lengths_train=np.array(lengths_train),
                            lengths_test=np.array(lengths_test))

    if hasattr(original_lengths, 'tolist'):
        original_lengths = original_lengths.tolist()
    lengths_train = lengths_train.tolist() if hasattr(lengths_train, 'tolist') else lengths_train
    lengths_test = lengths_test.tolist() if hasattr(lengths_test, 'tolist') else lengths_test

    return (data, labels_ntu, data_train, labels_train,
            data_test, labels_test, original_lengths,
            lengths_train, lengths_test)


def last_frame_pad(seq, target_len):
    seq_array = np.array(seq)
    if len(seq_array) >= target_len:
        return seq_array[:target_len]
    last_frame = seq_array[-1]
    extended = np.zeros((target_len, seq_array.shape[1]), dtype=seq_array.dtype)
    extended[:len(seq_array)] = seq_array
    extended[len(seq_array):] = last_frame
    return extended






# ================== Custom Dataset (Real time Augmentation+Feature Calculation) ==================
class NTUTrainTestLoader(Dataset):

    def __init__(self, data, label, lengths, joints, coords, augment=False,
                 target_frames=48, downsample_rate=3, random_downsample=True,
                 random_choose=True, random_shift=True, random_remove=True):
        """
        data: raw coordinates, shape (N, max_len, joints*coords)
        label: labels (N,)
        lengths: list of true lengths for each sample (N,)
        joints, coords: number of joints and coordinate dimensions
        augment: whether to enable augmentation
        target_frames: final output frame count
        downsample_rate: downsampling step size (takes effect only when > 1)
        random_downsample: if True, use random block downsampling (randomly sample one frame from every rate frames);
                           if False, use fixed-step downsampling (take the first frame every rate frames).
        random_choose, random_shift, random_remove: other augmentation switches
        """
        self.raw_coords = data.reshape(data.shape[0], data.shape[1], joints, coords)
        self.label = label
        self.lengths = lengths
        self.joints = joints
        self.coords = coords
        self.augment = augment
        self.target_frames = target_frames
        self.downsample_rate = downsample_rate
        self.random_downsample = random_downsample
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_remove = random_remove

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        coords_full = self.raw_coords[index]           # (max_len, J, 3)
        orig_len = self.lengths[index]
        coords = coords_full[:orig_len]                # (orig_len, J, 3)

        #  4D (C, T, J, M)
        data_4d = self.to_4d(coords)                   # (3, orig_len, J, 1)
        new_len = data_4d.shape[1]


        if self.downsample_rate > 1 and new_len >= self.downsample_rate:
            if self.augment and self.random_downsample:
                data_4d, new_len = self._random_block_downsample(data_4d, new_len, self.downsample_rate)
            else:
                data_4d = data_4d[:, ::self.downsample_rate, :, :]
                new_len = data_4d.shape[1]

        if self.augment:
            min_eff = 26
            if self.random_shift and np.random.rand() < 0.5:
                data_4d, new_len = self._temporal_shift_left_zero_pad(
                    data_4d, rand_shift=True, shift_ratio=0.1, min_effective=min_eff)
            if self.random_choose and np.random.rand() < 0.9:
                data_4d, new_len = self._random_choose(
                    data_4d, self.target_frames, choose_ratio=1, min_effective=min_eff)

        # ---------- Spatial augmentation  ----------
        if self.augment and self.random_remove and np.random.rand() < 0.7:
            coords_aug = self.from_4d(data_4d)
            coords_aug = self.random_3d_remove(coords_aug)
            data_4d = self.to_4d(coords_aug)

        # Uniformly fill/extract to target_frames
        if new_len < self.target_frames:
            pad = self.target_frames - new_len
            data_padded = np.pad(data_4d, ((0, 0), (0, pad), (0, 0), (0, 0)), mode='constant')
            eff_len = new_len
        else:
            data_padded = data_4d[:, :self.target_frames, :, :]
            eff_len = min(new_len, self.target_frames)

        coords_padded = self.from_4d(data_padded)
        return coords_padded, self.label[index], eff_len

    def to_4d(self, coords):
        return coords.transpose(2, 0, 1)[..., np.newaxis]

    def from_4d(self, arr_4d):
        return arr_4d.transpose(1, 2, 0, 3).squeeze(-1)


    def _random_block_downsample(self, data, orig_len, rate):
        """
        Random block-wise sampling: divide the orig_len frames into groups of rate frames each, and randomly sample 1 frame from each group.
        data: (C, T, J, M)
        rate: downsampling step size (e.g., 5)
        Returns: (downsampled data, new length)
        """
        if orig_len < rate or rate <= 1:
            return data, orig_len

        indices = []
        for start in range(0, orig_len, rate):
            end = min(max(0, start + rate - 1), orig_len)  # 保证索引不越界
            if start < end:
                idx = np.random.randint(start, end)
                indices.append(idx)

        if not indices:
            return data, orig_len

        new_data = data[:, indices, :, :]
        return new_data, len(indices)


    def _temporal_shift_left_zero_pad(self, data, rand_shift=True, shift_ratio=0.1, min_effective=30):
        C, T, V, M = data.shape
        if T <= min_effective:
            return data.copy(), T
        max_allowed_shift = T - min_effective
        max_shift = min(int(T * shift_ratio), max_allowed_shift)
        if max_shift <= 0:
            return data.copy(), T
        if rand_shift:
            shift_abs = np.random.randint(0, max_shift + 1)
        else:
            shift_abs = max(1, int(max_shift * 0.5))
        effective_len = T - shift_abs
        shifted = data[:, shift_abs:, :, :].copy()
        return shifted, effective_len

    def _random_choose(self, data_numpy, size, choose_ratio=0.8, min_effective=None):
        C, T, V, M = data_numpy.shape
        if T == 0:
            return np.zeros((C, 0, V, M), dtype=data_numpy.dtype), 0
        low = max(1, int(choose_ratio * size))
        if min_effective is not None:
            low = max(low, min_effective)
        high = size
        if T <= low:
            L = T
        else:
            L = random.randint(low, min(high, T))
        if L == T:
            begin = 0
        else:
            begin = random.randint(0, T - L)
        cropped = data_numpy[:, begin:begin + L, :, :]
        return cropped, L


    def random_3d_remove(self, seq, angle_candidate=[-10., -5., 0., 5., 10.],
                         scale_candidate=[0.9, 1.0, 1.1],
                         transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                         move_time_candidate=[1]):
        T, J, C = seq.shape
        coords = seq[:, :, :3]
        move_time = random.choice(move_time_candidate)
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)
        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)
        T_z = np.random.choice(transform_candidate, num_node)
        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)
        t_z = np.zeros(T)
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
            t_z[node[i]:node[i + 1]] = np.linspace(T_z[i], T_z[i + 1], node[i + 1] - node[i])
        for i_frame in range(T):
            frame_coords = coords[i_frame, :, :]
            cos_a = np.cos(a[i_frame])
            sin_a = np.sin(a[i_frame])
            R = np.array([[cos_a, -sin_a, 0],
                          [sin_a, cos_a, 0],
                          [0, 0, 1]])
            rotated = frame_coords @ R.T
            scaled = rotated * s[i_frame]
            shifted = scaled + np.array([t_x[i_frame], t_y[i_frame], t_z[i_frame]])
            coords[i_frame, :, :] = shifted
        seq[:, :, :3] = coords
        return seq


def get_affective_features_batch(coords, lengths):
    """
    coords: (B, T, J, 3) padded to uniform length T
    lengths: (B,) effective frame count for each sample (<= T)
    Returns: features (B, T, 384) with padded frame features set to zero
    """
    B, T, J, _ = coords.shape
    device = coords.device
    eps = 1e-8
    pi_2 = torch.tensor(np.pi / 2.0, dtype=torch.float64, device=device)

    coords = coords.to(torch.float64)
    dtype = torch.float64

    root = coords[:, :, 0:1, :]          # (B, T, 1, 3)
    rel_coords = coords - root           # (B, T, J, 3)

    # ---------- 2. Procrustes transformation  ----------
    Y = torch.tensor([
        [0.0, -8.40212115e-03, -9.53851533e-03, -4.03871819e-03, -7.52738843e-02,
         -2.81678708e-01, -4.17936960e-01, 7.96288259e-02, 2.76516953e-01, 4.11510516e-01,
         -6.74779788e-02, -1.01493726e-01, -1.10780176e-01, 5.77831078e-02, 7.79183409e-02, 8.74310686e-02],
        [0.0, 1.20554661e-01, 2.27028674e-01, 2.97235946e-01, 2.26821076e-01, 1.68745608e-01,
         1.50716715e-01, 2.08176310e-01, 1.58846755e-01, 1.27152295e-01, -6.93229158e-02,
         -3.71018802e-01, -6.64287975e-01, -7.96507542e-02, -3.71755971e-01, -6.69173159e-01],
        [0.0, 2.61917976e-02, 2.57063591e-02, 8.99503207e-03, -6.27215921e-02, -5.69554221e-02,
         -2.57920354e-02, -5.61114550e-02, -7.55548450e-02, -4.25150186e-02, 8.19714895e-03,
         2.70399580e-02, -6.41822474e-02, -3.39123402e-04, 7.18219105e-03, -6.64537982e-02]
    ], dtype=dtype, device=device)

    Y_T = Y.unsqueeze(0).expand(B, -1, -1)          # (B, 3, 16)
    my = Y_T.mean(dim=2, keepdim=True)              # (B, 3, 1)
    Yc = Y_T - my

    X0_T = rel_coords[:, 0, :, :].transpose(1, 2)   # (B, 3, 16)
    mx = X0_T.mean(dim=2, keepdim=True)             # (B, 3, 1)
    Xc = X0_T - mx

    M = torch.bmm(Yc, Xc.transpose(1, 2)) / J       # (B, 3, 3)
    U, S, Vh = torch.linalg.svd(M)
    det = torch.linalg.det(M)
    S_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    S_mat[det < 0, 2, 2] = -1
    R = torch.bmm(torch.bmm(U, S_mat), Vh)          # (B, 3, 3)

    trace = torch.diagonal(torch.bmm(torch.diag_embed(S), S_mat), dim1=-2, dim2=-1).sum(dim=-1)
    sx = (Xc ** 2).sum(dim=(1, 2)) / J
    c = trace / (sx + 1e-12)                        # (B,)

    R_mx = torch.bmm(R, mx)                         # (B, 3, 1)
    t = my - c.view(B, 1, 1) * R_mx                # (B, 3, 1)
    t = t.squeeze(-1)                               # (B, 3)

    # 变换坐标
    rel_T = rel_coords.transpose(2, 3)              # (B, T, 3, J)
    R_exp = R.unsqueeze(1)                          # (B, 1, 3, 3)
    transformed = c.view(B, 1, 1, 1) * torch.matmul(R_exp, rel_T) + t.view(B, 1, 3, 1)
    transformed = transformed.transpose(2, 3)       # (B, T, J, 3)

    # ---------- 3. Angle features ----------
    angle_pairs = torch.tensor([
        (10, 0, 13), (0, 1, 2), (4, 2, 7), (1, 2, 3), (2, 4, 5), (4, 5, 6),
        (2, 6, 9), (2, 7, 8), (7, 8, 9), (2, 9, 6), (0, 10, 11), (10, 11, 12),
        (0, 12, 15), (0, 13, 14), (13, 14, 15), (0, 15, 12)
    ], dtype=torch.long, device=device)

    p1 = transformed[:, :, angle_pairs[:, 0], :]
    p2 = transformed[:, :, angle_pairs[:, 1], :]
    p3 = transformed[:, :, angle_pairs[:, 2], :]
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = torch.norm(v1, dim=-1)
    norm2 = torch.norm(v2, dim=-1)
    valid = (norm1 > 0) & (norm2 > 0)
    angle = torch.full((B, T, 16), pi_2, dtype=dtype, device=device)
    cos_angle = torch.clamp((v1 * v2).sum(dim=-1) / (norm1 * norm2 + 1e-12), -1.0, 1.0)
    angle[valid] = torch.acos(cos_angle[valid])

    # --------- --------
    features = torch.zeros(B, T, 384, dtype=dtype, device=device)
    feat_48 = transformed.reshape(B, T, -1)[:, :, :48]
    features[:, :, :48] = feat_48
    features[:, :, 48:64] = angle

    # ----------Inter frame dynamics ----------
    base_linear = 64
    base_angular = 192
    base_rel_linear = 224
    base_rel_angular = 352

    for b in range(B):
        L = lengths[b].item()
        if L <= 1:
            continue

        coords_b = transformed[b, :L, :, :]          # (L, J, 3)
        angle_b = angle[b, :L]                       # (L, 16)

        prev_vel_vec = torch.zeros(J, 3, dtype=dtype, device=device)
        prev_angle_vel = torch.zeros(16, dtype=dtype, device=device)

        for t in range(1, L):

            vel_vec = (coords_b[t] - coords_b[t-1]) #/ 1000.0
            vel_norm = torch.norm(vel_vec, dim=-1, keepdim=True)
            vel = torch.cat([vel_vec, vel_norm], dim=-1)
            acc_vec = vel_vec - prev_vel_vec
            acc_norm = torch.norm(acc_vec, dim=-1, keepdim=True)
            acc = torch.cat([acc_vec, acc_norm], dim=-1)
            lin_dyn = torch.cat([vel, acc], dim=-1)          # (J, 8)
            features[b, t, base_linear:base_linear+128] = lin_dyn.reshape(-1)

            angle_vel = angle_b[t] - angle_b[t-1]
            angle_acc = angle_vel - prev_angle_vel
            ang_dyn = torch.stack([angle_vel, angle_acc], dim=1)  # (16, 2)
            features[b, t, base_angular:base_angular+32] = ang_dyn.reshape(-1)

            # Relative linearity (bounded normalized difference)
            root_lin = lin_dyn[0:1, :]  # (1, 8)
            denom_lin = torch.abs(lin_dyn) + torch.abs(root_lin) + eps
            rel_lin = (lin_dyn - root_lin) / denom_lin  # (J, 8)
            rel_lin[0, :] = 0.0  # 根节点自身相对特征置零
            features[b, t, base_rel_linear:base_rel_linear + 128] = rel_lin.reshape(-1)

            # Relative angle (bounded normalized difference)
            root_ang = ang_dyn[0:1, :]  # (1, 2)
            denom_ang = torch.abs(ang_dyn) + torch.abs(root_ang) + eps
            rel_ang = (ang_dyn - root_ang) / denom_ang  # (16, 2)
            rel_ang[0, :] = 0.0
            features[b, t, base_rel_angular:base_rel_angular + 32] = rel_ang.reshape(-1)

            prev_vel_vec = vel_vec
            prev_angle_vel = angle_vel

    return features



def collate_fn(batch, mean=None, std=None, device='cuda', return_raw=False):
    """
    batch: list of tuples (coords, label, length, is_real)
    coords: np.ndarray of shape (T, J, 3) padded to target_frames
    mean, std: precomputed global mean and standard deviation (shape (384,)); if None, no normalization is applied
    return_raw: if True, return raw features (B, T, 384); otherwise, return model input (B, C, T, J, 1)
    """
    coords_list, labels, lengths = zip(*batch)

    max_len = max(c.shape[0] for c in coords_list)
    B = len(batch)
    J = coords_list[0].shape[1]
    padded_coords = torch.zeros(B, max_len, J, 3, dtype=torch.float64)
    for i, c in enumerate(coords_list):
        padded_coords[i, :c.shape[0], :, :] = torch.from_numpy(c)

    padded_coords = padded_coords.to(device)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    #  (B, T, 384)
    features = get_affective_features_batch(padded_coords, lengths_t)

    if mean is not None and std is not None:
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float64, device=device)
        else:
            mean = mean.to(device, dtype=torch.float64)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float64, device=device)
        else:
            std = std.to(device, dtype=torch.float64)
        features = (features - mean) / (std + 1e-8)

    if return_raw:
        return features, labels_t, lengths_t

    #  (B, 14, T, J, 1)
    B, T, _ = features.shape
    gait = features[:, :, :48].reshape(B, T, J, 3).permute(0, 3, 1, 2).unsqueeze(-1)
    pos = features[:, :, 48:64].reshape(B, T, J, 1).permute(0, 3, 1, 2).unsqueeze(-1)
    move_intra = features[:, :, 224:352].reshape(B, T, J, 8).permute(0, 3, 1, 2).unsqueeze(-1)
    angle_intra = features[:, :, 352:384].reshape(B, T, J, 2).permute(0, 3, 1, 2).unsqueeze(-1)
    combined = torch.cat([gait, move_intra, pos, angle_intra], dim=1)  # (B, 14, T, J, 1)
    return combined, labels_t, lengths_t
