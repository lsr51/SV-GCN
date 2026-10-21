# sys
import h5py
import os
import numpy as np


from sklearn.model_selection import train_test_split
from utils import common

# torch
import torch
from torchvision import datasets, transforms

from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
import random





def load_data_real(_path, _ftype_real, _ftype_real2, coords, joints, split=False):
    """
    Load real data, perform only frame padding, and return raw coordinates (no emotion feature extraction).
    Parameters:
        _path: data path
        _ftype_real: suffix for real data files (e.g., '_real')
        _ftype_real2: suffix for synthetic data (unused, kept for compatibility)
        coords: coordinate dimensions (usually 3)
        joints: number of joints
        split: whether to split (fixed as False here, splitting is done externally)
    Returns:
        data: raw coordinates of shape (N, final_frames, joints*coords)
        labels: label array
        effective_lengths: list of effective frame counts for each sample
    """
    file_feature_real = os.path.join(_path, 'features' + _ftype_real + '.h5')
    ffr = h5py.File(file_feature_real, 'r')
    file_label_real = os.path.join(_path, 'labels' + _ftype_real + '.h5')
    flr = h5py.File(file_label_real, 'r')

    data_list_real = []
    num_samples_real = len(ffr.keys())
    print("num_samples_real:", num_samples_real)

    original_frames = 240
    down_step = 1
    frames_after_down = original_frames // down_step
    final_frames = 240

    labels_real = np.empty(num_samples_real)
    effective_lengths = []

    for si in range(num_samples_real):
        ffr_group_key = list(ffr.keys())[si]
        seq = np.array(ffr[ffr_group_key])            # (240, 48)
        seq_down = seq[::down_step]                   # (240, 48)
        effective_lengths.append(frames_after_down)
        if seq_down.shape[0] < final_frames:
            seq_down = last_frame_pad(seq_down, final_frames)
        else:
            seq_down = seq_down[:final_frames]
        data_list_real.append(seq_down)
        labels_real[si] = flr[list(flr.keys())[si]][()]

    #  (N, final_frames, joints*coords)
    data = np.zeros((num_samples_real, final_frames, joints * coords))
    for si in range(num_samples_real):
        data[si, :, :] = data_list_real[si]


    if split:
        is_real_all = [True] * len(labels_real)
        data_train, data_test, labels_train, labels_test, lengths_train, lengths_test, is_real_train, is_real_test = train_test_split(
            data, labels_real, effective_lengths, is_real_all, test_size=0.1)
        return data, labels_real, data_train, labels_train, data_test, labels_test, effective_lengths, lengths_train, lengths_test, \
            is_real_all, is_real_train, is_real_test
    else:
        return data, labels_real, effective_lengths



def load_data_real2(_path, _ftype_real, _ftype_real2, coords, joints, split=False):
    """
    Load synthetic data, perform only frame padding, and return raw coordinates (no emotion feature extraction).
    Parameters are the same as in load_data_real.
    Returns:
        data: raw coordinates of shape (N, final_frames, joints*coords)
        labels: label array
        effective_lengths: list of effective frame counts for each sample
    """
    file_feature_real2 = os.path.join(_path, 'features' + _ftype_real2 + '.h5')
    ffs = h5py.File(file_feature_real2, 'r')
    file_label_real2 = os.path.join(_path, 'labels' + _ftype_real2 + '.h5')
    fls = h5py.File(file_label_real2, 'r')

    data_list_real2 = []
    num_samples_real2 = len(ffs.keys())
    print("num_samples_real2:", num_samples_real2)

    final_frames = 240
    labels_real2 = np.empty(num_samples_real2)
    effective_lengths = []

    for si in range(num_samples_real2):
        ffs_group_key = list(ffs.keys())[si]
        seq = np.array(ffs[ffs_group_key])            # (T_i, D)
        raw_len = seq.shape[0]
        eff_len = min(raw_len, final_frames)
        effective_lengths.append(eff_len)

        if eff_len < 24:
            print(f"⚠️ Warning: sample {si} has length {eff_len} < 24")
        if raw_len < final_frames:
            seq = last_frame_pad(seq, final_frames)
        else:
            seq = seq[:final_frames]
        data_list_real2.append(seq)
        labels_real2[si] = fls[list(fls.keys())[si]][()]

    data = np.zeros((num_samples_real2, final_frames, joints * coords))
    for si in range(num_samples_real2):
        data[si, :, :] = data_list_real2[si]


    if split:
        is_real_all = [False] * len(labels_real2)
        data_train, data_test, labels_train, labels_test, lengths_train, lengths_test, is_real_train, is_real_test = train_test_split(
            data, labels_real2, effective_lengths, is_real_all, test_size=0.1)
        return data, labels_real2, data_train, labels_train, data_test, labels_test, effective_lengths, lengths_train, \
            lengths_test, is_real_all, is_real_train, is_real_test
    else:
        return data, labels_real2, effective_lengths


def load_data(_path, _ftype_real, _ftype_real2, coords, joints, split=False):
    """
    Load and merge real/synthetic data (raw coordinates), uniformly split into train/test sets at a 9:1 ratio,
    and also return the source identifier for each sample (True = real, False = synthetic).
    Parameters:
        split: reserved parameter (unused here, splitting is done uniformly)
    Returns:
        data, labels, data_train, labels_train, data_test, labels_test,
        effective_lengths, lengths_train, lengths_test,
        is_real_all, is_real_train, is_real_test
    """

    data_real, labels_real, lengths_real = load_data_real(
        _path, _ftype_real, _ftype_real2, coords, joints, False)
    data_real2, labels_real2, lengths_real2 = load_data_real2(
        _path, _ftype_real, _ftype_real2, coords, joints, False)

    is_real_all = [True] * len(labels_real) + [False] * len(labels_real2)

    data = np.concatenate([data_real, data_real2], axis=0)
    labels = np.concatenate([labels_real, labels_real2], axis=0)
    lengths = lengths_real + lengths_real2

    data_train, data_test, labels_train, labels_test, \
        lengths_train, lengths_test, is_real_train, is_real_test = train_test_split(
        data, labels, lengths, is_real_all, test_size=0.1)

    return (data, labels,
            data_train, labels_train, data_test, labels_test,
            lengths, lengths_train, lengths_test,
            is_real_all, is_real_train, is_real_test)



def last_frame_pad(seq, target_len):
    """
    seq: (T_ori, C)
    return (T_new, C)
    """
    seq_array = np.array(seq)

    if len(seq_array) >= target_len:
        return seq_array[:target_len]

    last_frame = seq_array[-1]


    extended = np.zeros((target_len, seq_array.shape[1]), dtype=seq_array.dtype)
    extended[:len(seq_array)] = seq_array
    extended[len(seq_array):] = last_frame

    return extended





class TrainTestLoader(torch.utils.data.Dataset):
    def __init__(self, data, label, lengths, joints, coords, is_real=None, augment=False,
                 target_frames=48,
                 downsample_rate = 5,
                 random_choose=True,
                 random_shift=True,
                 random_remove=True,
                 normalization=True):
        """
        Parameters:
            data: raw coordinate data, shape (N, T_total, joints*coords)
            label: label array (N,)
            lengths: list of effective frame counts for each sample (N,)
            joints: number of joints
            coords: coordinate dimensions (usually 3)
            is_real: list of source identifiers (True = real, False = synthetic), used to adjust augmentation parameters
            augment: whether to enable augmentation
            target_frames: final output frame count (usually 48 during training)
            random_choose: whether to enable random cropping
            random_shift: whether to enable random left shift
            random_remove: whether to enable spatial augmentation (rotation/scaling/translation)
            normalization: whether to enable normalization (will be implemented in collate)
        """

        self.raw_coords = data.reshape(data.shape[0], data.shape[1], joints, coords)
        self.label = label
        self.lengths = lengths
        self.joints = joints
        self.is_real = is_real if is_real is not None else [True] * len(label)
        self.augment = augment
        self.target_frames = target_frames
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_remove = random_remove
        self.normalization = normalization
        self.downsample_rate = downsample_rate

        self.mean = None
        self.std = None

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        coords_full = self.raw_coords[index]  # (T_total, J, 3)
        orig_len = self.lengths[index]
        coords = coords_full[:orig_len]  # (orig_len, J, 3)

        #  (C, T, J, 1)， C=3
        data_4d = self.to_4d(coords)  # (3, orig_len, J, 1)
        new_len = data_4d.shape[1]  # orig_len

        if self.is_real[index]:
            if self.augment:
                data_4d, new_len = self._random_block_downsample(data_4d, new_len, self.downsample_rate)
            else:
                data_4d = data_4d[:, ::self.downsample_rate, :, :]
                new_len = data_4d.shape[1]


        #  Time augmentation (shift/truncation) - returns a variable length sequence without padding
        if self.augment:
            min_eff = 42 if self.is_real[index] else 28
            r1 = 0.3 if self.is_real[index] else 0.3
            r2 = 0.3 if self.is_real[index] else 0.3
            sr = 0.1 if self.is_real[index] else 0.1
            cr = 0.9 if self.is_real[index] else 1


            if self.random_shift and np.random.rand() < r1:
                data_4d, new_len = self._temporal_shift_left_zero_pad(
                    data_4d, rand_shift=True, shift_ratio=sr, min_effective=min_eff)

            if self.random_choose and np.random.rand() < r2:
                data_4d, new_len = self._random_choose(
                    data_4d, self.target_frames, choose_ratio=cr, min_effective=min_eff)
            # data_4d (3, new_len, J, 1)
        else:
            data_4d = data_4d[:, :new_len, :, :]

        #  Spatial augmentation (rotation/scaling/translation) - only applies to valid frames
        if self.augment and self.random_remove and np.random.rand() < 0.7:
            #  (T, J, 3)
            coords_aug = self.from_4d(data_4d)  # (new_len, J, 3)
            coords_aug = self.random_3d_remove(coords_aug)
            data_4d = self.to_4d(coords_aug)  # (3, new_len, J, 1)

        # Uniformly fill/extract to target_frames
        if new_len < self.target_frames:
            pad = self.target_frames - new_len
            data_padded = np.pad(data_4d, ((0, 0), (0, pad), (0, 0), (0, 0)), mode='constant')
            eff_len = new_len
        else:
            data_padded = data_4d[:, :self.target_frames, :, :]
            eff_len = min(new_len, self.target_frames)

        coords_padded = self.from_4d(data_padded)  # (target_frames, J, 3)

        return coords_padded, self.label[index], eff_len, self.is_real[index]


    def to_4d(self, coords):
        # coords: (T, J, 3) -> (3, T, J, 1)
        return coords.transpose(2, 0, 1)[..., np.newaxis]

    def from_4d(self, arr_4d):
        # arr_4d: (3, T, J, 1) -> (T, J, 3)
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
            end = min(max(0, start + rate - 2), orig_len)
            if start < end:
                idx = np.random.randint(start, end)
                indices.append(idx)

        if not indices:
            return data, orig_len

        new_data = data[:, indices, :, :]
        return new_data, len(indices)


    def _temporal_shift_left_zero_pad(self, data, rand_shift=True, shift_ratio=0.1, min_effective=30):
        """
        Shift the sequence to the left (discard several frames at the beginning), returning the remaining valid frames (variable length).
        The original implementation padded with zeros at the end; now it is changed to direct truncation to avoid zero-padded frames contaminating subsequent spatial augmentation.

        Parameters:
            data: (C, T, V, M) input sequence (4D tensor)
            rand_shift: whether to use random shift amount; if False, use a fixed ratio (half of max_shift)
            shift_ratio: maximum shift ratio (relative to T)
            min_effective: minimum valid frame count constraint (ensures that the frame count after shifting is not less than this value)

        Returns:
            shifted: (C, effective_len, V, M) variable-length sequence, without padded frames
            effective_len: new effective length (= T - shift_abs)
        """
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
        shifted = data[:, shift_abs:, :, :].copy()  # (C, effective_len, V, M)

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

    def random_3d_remove(self, seq,
                         angle_candidate=[-10., -5., 0., 5., 10.],
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

    # ----------  Procrustes transformation ----------
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

    rel_T = rel_coords.transpose(2, 3)              # (B, T, 3, J)
    R_exp = R.unsqueeze(1)                          # (B, 1, 3, 3)
    transformed = c.view(B, 1, 1, 1) * torch.matmul(R_exp, rel_T) + t.view(B, 1, 3, 1)
    transformed = transformed.transpose(2, 3)       # (B, T, J, 3)

    # ----------  Angle features ----------
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

    # ----------  ----------
    features = torch.zeros(B, T, 384, dtype=dtype, device=device)
    feat_48 = transformed.reshape(B, T, -1)[:, :, :48]
    features[:, :, :48] = feat_48
    features[:, :, 48:64] = angle

    # ---------- Inter frame dynamics ----------
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
            rel_lin[0, :] = 0.0
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
    coords_list, labels, lengths, is_reals = zip(*batch)

    max_len = max(c.shape[0] for c in coords_list)
    B = len(batch)
    J = coords_list[0].shape[1]
    padded_coords = torch.zeros(B, max_len, J, 3, dtype=torch.float64)
    for i, c in enumerate(coords_list):
        padded_coords[i, :c.shape[0], :, :] = torch.from_numpy(c)

    padded_coords = padded_coords.to(device)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    is_real_t = torch.tensor(is_reals, dtype=torch.bool, device=device)

    # (B, T, 384)
    features = get_affective_features_batch(padded_coords, lengths_t)

    if mean is not None and std is not None:
        if isinstance(mean, torch.Tensor):
            mean_t = mean.to(device, dtype=torch.float64)
        else:
            mean_t = torch.tensor(mean, dtype=torch.float64, device=device)
        if isinstance(std, torch.Tensor):
            std_t = std.to(device, dtype=torch.float64)
        else:
            std_t = torch.tensor(std, dtype=torch.float64, device=device)
        features = (features - mean_t) / (std_t + 1e-8)

    if return_raw:
        return features, labels_t, lengths_t, is_real_t

    #  (B, C, T, J, 1)  where C=14
    gait = features[:, :, :48].reshape(B, max_len, J, 3).permute(0, 3, 1, 2).unsqueeze(-1)          # (B, 3, T, J, 1)
    pos = features[:, :, 48:64].reshape(B, max_len, J, 1).permute(0, 3, 1, 2).unsqueeze(-1)         # (B, 1, T, J, 1)
    move= features[:, :, 64:192].reshape(B, max_len, J, 8).permute(0, 3, 1, 2).unsqueeze(-1) # (B, 8, T, J, 1)
    angle = features[:, :, 192:224].reshape(B, max_len, J, 2).permute(0, 3, 1, 2).unsqueeze(-1) # (B, 2, T, J, 1)
    move_intra = features[:, :, 224:352].reshape(B, max_len, J, 8).permute(0, 3, 1, 2).unsqueeze(-1) # (B, 8, T, J, 1)
    angle_intra = features[:, :, 352:384].reshape(B, max_len, J, 2).permute(0, 3, 1, 2).unsqueeze(-1) # (B, 2, T, J, 1)
    combined = torch.cat([gait, move_intra, pos, angle_intra], dim=1)  # (B, 14, T, J, 1)

    return combined, labels_t, lengths_t, is_real_t
