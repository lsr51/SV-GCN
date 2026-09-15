import argparse
import os
import numpy as np
from utils import loader0_ntu_cache, processor0_ntu_cache
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, WeightedRandomSampler



base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, './data')
ftype_real = '_ELMD'
ftype_real2 = ''
ftype_ntu = '_NTU_180'

coords = 3
joints = 16
cycles = 1
feat_dim = 10 + 4
model_path = os.path.join(base_path, 'model_classifier_stgcn/features' + ftype_ntu)

parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--train', type=bool, default=True, metavar='T',
                    help='train the model (default: True)')
parser.add_argument('--save-features', type=bool, default=True, metavar='SF',
                    help='save penultimate layer features (default: True)')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 48)')
parser.add_argument('--num-worker', type=int, default=0, metavar='W',
                    help='number of workers (must be 0 for GPU collate)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num_epoch', type=int, default=300, metavar='NE',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='optimizer (default: Adam)')
parser.add_argument('--base-lr', type=float, default=0.001, metavar='L',
                    help='base learning rate (default: 0.001)')
parser.add_argument('--step', type=list, default=[0.5, 0.75, 0.875], metavar='[S]',
                    help='fraction of steps when learning rate will be decreased')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--topk', type=list, default=[1], metavar='[K]',
                    help='top K accuracy to show (default: [1])')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')
parser.add_argument('--work-dir', type=str, default=model_path, metavar='WD',
                    help='path to save')
parser.add_argument('--mean-std-file', type=str, default=os.path.join(base_path, 'mean_std_ntu_180.npy'),
                    help='path to save/load mean and std for NTU features')
# 可添加下采样参数
parser.add_argument('--downsample-rate', type=int, default=3,
                    help='downsample rate for NTU data')
parser.add_argument('--random-downsample', action='store_true', default=True,
                    help='use random block downsampling in training')


def main():
    args = parser.parse_args()
    device = 'cuda:3' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"Using device: {device}")

    # ----- Loading data (raw coordinates) -----
    data, labels, data_train, labels_train, data_test, labels_test, effective_lengths, lengths_train, lengths_test = \
        loader0_ntu_cache.load_data_ntu_cache(
            data_path, ftype_ntu, coords, joints, cycles=1,
            use_cache=True, cache_dir=None, exit_after_cache=True
        )

    num_classes = np.unique(labels_train).shape[0]
    print("Number of classes:", num_classes)

    # ----- Mean and standard deviation: loaded or calculated -----
    mean_std_file = args.mean_std_file
    if os.path.exists(mean_std_file):
        print(f"Loading mean and std from {mean_std_file}")
        stats = np.load(mean_std_file, allow_pickle=True).item()
        mean = stats['mean']
        std = stats['std']
    else:
        print("Computing mean and std from training set (no augmentation)...")
        #Create a non augmented statistical dataset
        stat_dataset = loader0_ntu_cache.NTUTrainTestLoader(
            data_train, labels_train, lengths_train, joints, coords,
            augment=False,
            random_choose=False,
            random_shift=False,
            random_remove=False
        )
        stat_loader = DataLoader(
            stat_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: loader0_ntu_cache.collate_fn(
                batch, mean=None, std=None, device=device, return_raw=True
            )
        )

        all_feat_list = []
        with torch.no_grad():
            for features, labels_t, lengths_t in stat_loader:
                # features: (B, T, 384)
                for b in range(features.size(0)):
                    L = lengths_t[b].item()
                    if L > 0:
                        all_feat_list.append(features[b, :L, :].cpu().numpy())
        if all_feat_list:
            all_feat = np.concatenate(all_feat_list, axis=0)
            mean = all_feat.mean(axis=0)
            std = all_feat.std(axis=0) + 1e-8
            print("Mean and std computed.")
            np.save(mean_std_file, {'mean': mean, 'std': std})
            print(f"Saved mean and std to {mean_std_file}")
        else:
            mean = np.zeros(384)
            std = np.ones(384)
            print("Warning: No valid frames, using zeros/ones.")

        del stat_dataset, stat_loader

    mean_t = torch.tensor(mean, dtype=torch.float64, device=device)
    std_t = torch.tensor(std, dtype=torch.float64, device=device)

    # ----- Building datasets and DataLoader -----
    train_dataset = loader0_ntu_cache.NTUTrainTestLoader(
        data_train, labels_train, lengths_train, joints, coords,
        augment=True,
        random_choose=True,
        random_shift=True,
        random_remove=True
    )

    # Sample weight
    labels_np = labels_train.astype(int)
    unique_classes, counts = np.unique(labels_np, return_counts=True)
    class_weight = 1.0 / counts.astype(np.float32)
    class_weight = class_weight / class_weight.sum() * len(unique_classes)
    sample_weights = class_weight[labels_np]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    def train_collate(batch):
        return loader0_ntu_cache.collate_fn(batch, mean=mean_t, std=std_t, device=device, return_raw=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_worker,
        pin_memory=False,
        collate_fn=train_collate
    )

    test_dataset = loader0_ntu_cache.NTUTrainTestLoader(
        data_test, labels_test, lengths_test, joints, coords,
        augment=False,
        random_choose=False,
        random_shift=False,
        random_remove=False
    )

    def test_collate(batch):
        return loader0_ntu_cache.collate_fn(batch, mean=mean_t, std=std_t, device=device, return_raw=False)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_worker,
        pin_memory=False,
        collate_fn=test_collate
    )

    data_loader = {'train': train_loader, 'test': test_loader}


    unique_train, counts_train = np.unique(labels_train, return_counts=True)
    print("Distribution of training set classes:")
    for cls, count in zip(unique_train, counts_train):
        print(f"Class {int(cls)}: {count} samples")
    unique_test, counts_test = np.unique(labels_test, return_counts=True)
    print("Distribution of testing set classes:")
    for cls, count in zip(unique_test, counts_test):
        print(f"Class {int(cls)}: {count} samples")

    graph_dict = {'strategy': 'spatial'}
    print('Number of classes: {:d}'.format(num_classes))

    pr = processor0_ntu_cache.Processor(args, data_loader, feat_dim, num_classes, graph_dict, device=device)
    if args.train:
        pr.train()



if __name__ == '__main__':
    mp.freeze_support()
    main()