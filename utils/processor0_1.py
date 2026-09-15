import h5py
import math
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from net import classifier0_1
from torchlight import torchlight
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def find_all_substr(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_best_epoch_and_accuracy(path_to_model_files):
    all_models = os.listdir(path_to_model_files)
    acc_list = np.zeros(len(all_models))
    for i, model in enumerate(all_models):
        acc = str.split(model, '_')
        if len(acc) > 1:
            acc_list[i] = float(acc[1][3:])
    best_model = all_models[np.argmax(acc_list)]
    all_us = list(find_all_substr(best_model, '_'))
    return int(best_model[5:all_us[0]]), float(best_model[all_us[0]+4:all_us[1]])



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LinearWarmup(object):
    def __init__(self, optimizer, warmup_epoch, base_lr, start_lr=1e-3, gamma=0.99):
        self.optimizer = optimizer
        self.warmup_epoch = warmup_epoch
        self.base_lr = base_lr
        self.start_lr = start_lr
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        if self.epoch < self.warmup_epoch:
            lr = self.start_lr + \
                 (self.base_lr - self.start_lr) * self.epoch / self.warmup_epoch
        else:
            lr = self.base_lr * (self.gamma ** (self.epoch - self.warmup_epoch))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.epoch += 1


class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, data_loader, C, num_classes, graph_dict, device='cuda:0'):

        self.args = args
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.device = device
        log_filename = datetime.datetime.now().strftime("log_%m%d_%H%M.txt")
        self.io = torchlight.IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log,
            log_file=log_filename
        )


        self.class_labels = ['Happy', 'Sad', 'Angry', 'Neutral']
        #
        start_time_str = datetime.datetime.now().strftime("%m%d_%H%M")
        self.confusion_matrix_dir = os.path.join(self.args.work_dir, f'confusion_matrices_{start_time_str}')
        if not os.path.exists(self.confusion_matrix_dir):
            os.makedirs(self.confusion_matrix_dir)
            self.io.print_log(f'Created confusion matrix directory: {self.confusion_matrix_dir}')
            print(self.confusion_matrix_dir)


        # model
        if not os.path.isdir(self.args.work_dir):
            os.mkdir(self.args.work_dir)
        self.model = classifier0_1.Classifier(C, num_classes, graph_dict)
        self.model.cuda('cuda:0')
        self.model.apply(weights_init)

        '''
        if hasattr(self.args, 'pretrained') and self.args.pretrained is not None:
            checkpoint = torch.load(self.args.pretrained, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            #  Remove layers with mismatched dimensions
            state_dict.pop('fcn.weight', None)
            state_dict.pop('fcn.bias', None)

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            self.io.print_log(f"Loaded pretrained model from {self.args.pretrained} (FC layer reinitialized)")
            if missing:
                self.io.print_log(f"   Missing keys (randomly initialized): {missing}")
            if unexpected:
                self.io.print_log(f"   Unexpected keys (ignored): {unexpected}")
        '''

        # ========== Layer name comparison, only print matching layers ==========
        if hasattr(self.args, 'pretrained') and self.args.pretrained is not None:
            current_keys = set(self.model.state_dict().keys())
            pretrained_keys = set(state_dict.keys())  # state_dict

            common = current_keys & pretrained_keys
            #
            matched_layers = []
            for k in common:
                if state_dict[k].shape == self.model.state_dict()[k].shape:
                    matched_layers.append(k)

            print(f"\nMatching layers: {len(matched_layers)} / {len(current_keys)}")
            print("List of matched layer names:")
            for k in sorted(matched_layers):
                print(f"    {k}")

            #
            mismatched = [k for k in common if k not in matched_layers]
            if mismatched:
                print(f"\nThere are {len(mismatched)} common layers but their shapes do not match (skipped):")
                for k in mismatched:
                    print(f"    {k} (pre-training {state_dict[k].shape} vs Current {self.model.state_dict()[k].shape})")
        # ============================================================




        #self.loss = nn.CrossEntropyLoss()
        weight = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        weight = (weight / weight.sum()) * len(weight)
        self.loss = nn.CrossEntropyLoss(weight=weight).to('cuda:0')


        self.best_loss = math.inf
        #self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_epoch = None
        self.best_accuracy = np.zeros((1, np.max(self.args.topk)))
        self.accuracy_updated = False

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr

        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)


        self.scheduler = LinearWarmup(self.optimizer,
                                      warmup_epoch=7,
                                      base_lr=self.lr,
                                      start_lr=self.lr*0.1,
                                      gamma=0.99)


    def show_epoch_info(self):

        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.args.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def show_topk(self, k):

        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
        if accuracy > self.best_accuracy[0, k-1]:
            self.best_accuracy[0, k-1] = accuracy
            self.accuracy_updated = True
        else:
            self.accuracy_updated = False
        print_epoch = self.best_epoch if self.best_epoch is not None else 0
        self.io.print_log('\tTop{}: {:.2f}%. Best so far: {:.2f}% (epoch: {:d}).'.
                          format(k, accuracy, self.best_accuracy[0, k-1], print_epoch))

    def show_topk_tt(self, k, phase='Test', update_best=True):

        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)

        if update_best:
            if accuracy > self.best_accuracy[0, k - 1]:
                self.best_accuracy[0, k - 1] = accuracy
                self.accuracy_updated = True
            else:
                self.accuracy_updated = False
            print_epoch = self.best_epoch if self.best_epoch is not None else 0

            self.io.print_log('\t[{}] Top{}: {:.2f}%. Best so far: {:.2f}% (epoch: {:d}).'.
                              format(phase, k, accuracy, self.best_accuracy[0, k - 1], print_epoch))
        else:
            self.io.print_log('\t[{}] Top{}: {:.2f}%.'.format(phase, k, accuracy))

    def per_train(self):

        self.model.train()

        self.scheduler.step()

        loader = self.data_loader['train']
        loss_value = []

        result_frag = []
        label_frag = []
        # =====================================

        for data, label, lengths, _ in loader:
            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            # forward
            output, _ = self.model(data, lengths)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            #self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.iter_info['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            loss_value.append(self.iter_info['loss'])

            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())


            self.show_iter_info()
            self.meta_info['iter'] += 1

        # ==========
        if result_frag:
            self.result = np.concatenate(result_frag)
            self.label = np.concatenate(label_frag)

            accuracy_updated_backup = self.accuracy_updated

            for k in self.args.topk:
                self.show_topk_tt(k, phase='Train', update_best=False)

            self.accuracy_updated = accuracy_updated_backup
        # =========================================================

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()
        # for k in self.args.topk:
        #     self.calculate_topk(k, show=False)
        # if self.accuracy_updated:
            # self.model.extract_feature()


    def per_test(self, evaluation=True, is_show_ca=False, epoch=0):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        is_real_frag = []

        save_ca = True
        for data, label, lengths, is_real in loader:

            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)


            # inference
            with torch.no_grad():
                output, _ = self.model(data, lengths)

                is_show_ca = False
                if is_show_ca and save_ca and epoch % 10 == 0:
                    first_ca = self.model.st_gcn_networks[0].ca
                    att_weights = first_ca._att_buffer.squeeze(-1).squeeze(-1)  # (N, C)
                    #att_map = att_weights[0].transpose(0, 1)  # (V, C)

                    save_ca = False
                    plt.rcParams.update({'font.size': 26, 'axes.unicode_minus': False})

                    batch_size = att_weights.shape[0]
                    for i in range(batch_size):
                        fig, ax = plt.subplots(figsize=(10, 4))  # 高度留足
                        plot_data = att_weights[i].cpu().numpy().reshape(1, -1)
                        h = sns.heatmap(plot_data,
                                        cmap="viridis",
                                        cbar=True,
                                        cbar_kws={'label': 'Weight',
                                                  'orientation': 'horizontal',
                                                  'shrink': 0.5,
                                                  'pad': 0.5,
                                                  'aspect': 12},
                                        xticklabels=1,
                                        # yticklabels=[f'V{v}' for v in range(plot_data.shape[0])],
                                        ax=ax)

                        ax.tick_params(axis='x', labelsize=20)
                        ax.tick_params(axis='y', labelsize=20)
                        #plt.title("Channel Attention")
                        plt.xlabel("Channel Index", fontsize=20)
                        plt.tight_layout()

                        true_label = self.class_labels[label[i].item()].replace(' ', '_')
                        filename = f"epoch{epoch}_sample{i}_{true_label}_CA.png"
                        save_path = os.path.join(self.confusion_matrix_dir, filename)
                        plt.savefig(save_path, dpi=100, bbox_inches='tight')
                        plt.close()

            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())
                is_real_frag.append(is_real.cpu().numpy() if torch.is_tensor(is_real) else np.array(is_real))

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.is_real_all = np.concatenate(is_real_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # plot confusion matrix
            if hasattr(self, 'class_labels') and len(self.class_labels) == self.num_classes:
                epoch = self.meta_info['epoch']
                if epoch is None or epoch % 10 == 0 or epoch == 'final':
                    cm, cm_accuracy = self.plot_confusion_matrix(
                        self.label, self.result,
                        epoch=self.meta_info['epoch'],
                        is_test=True
                    )
                    self.io.print_log(f'\tConfusion Matrix Accuracy (Overall): {cm_accuracy:.2%}')

                    real_mask = self.is_real_all.astype(bool)
                    synth_mask = ~real_mask

                    if np.any(real_mask):
                        real_label = self.label[real_mask]
                        real_pred = self.result[real_mask]
                        cm_real, acc_real = self.plot_confusion_matrix(
                            real_label, real_pred,
                            epoch=epoch,
                            is_test=True,
                            subtitle='_real'
                        )
                        self.io.print_log(f'\tConfusion Matrix Accuracy (Real): {acc_real:.2%}')

                    if np.any(synth_mask):
                        synth_label = self.label[synth_mask]
                        synth_pred = self.result[synth_mask]
                        cm_synth, acc_synth = self.plot_confusion_matrix(
                            synth_label, synth_pred,
                            epoch=epoch,
                            is_test=True,
                            subtitle='_synth'
                        )
                        self.io.print_log(f'\tConfusion Matrix Accuracy (Synthetic): {acc_synth:.2%}')

            # show top-k accuracy
            for k in self.args.topk:
                self.show_topk_tt(k, phase='Test', update_best=True)

    def train(self):
        print("train begin")
        self.io.print_log('------------------------------------------------------------')

        self.io.print_log('TRAINING PARAMETERS:')
        self.io.print_log('=' * 50)
        for arg in vars(self.args):
            self.io.print_log(f'{arg}: {getattr(self.args, arg)}')
        self.io.print_log('=' * 50)
        self.io.print_log('')


        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                show_ca = False
                if (self.best_accuracy.item() > 85.00):
                    show_ca = True
                self.per_test(is_show_ca=show_ca, epoch=epoch)
                self.io.print_log('Done.')

            # save model and weights
            if self.accuracy_updated:
                if self.epoch_info['mean_loss'] < self.best_loss:
                    self.best_loss = self.epoch_info['mean_loss']
                    self.best_epoch = epoch
                    if (self.best_accuracy.item() > 88.00):
                        torch.save(self.model.state_dict(), os.path.join(self.args.work_dir,
                                'epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))

    def test(self):

        # the path of weights must be appointed
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.args.model))
        self.io.print_log('Weights: {}.'.format(self.args.weights))

        # evaluation
        self.io.print_log('Evaluation Start:')
        self.per_test()
        self.io.print_log('Done.\n')


        if hasattr(self, 'label') and hasattr(self, 'result') and hasattr(self, 'class_labels'):
            final_cm, final_accuracy = self.plot_confusion_matrix(
                self.label, self.result,
                epoch='final',
                is_test=True
            )
            self.io.print_log(f'Final Test Confusion Matrix Accuracy: {final_accuracy:.2%}')

        # save the output of model
        if self.args.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')

    def save_best_feature(self, ftype_real, ftype_synth, data, joints, coords):
        if self.best_epoch is None:
            self.best_epoch, best_accuracy = get_best_epoch_and_accuracy(self.args.work_dir)
        else:
            best_accuracy = self.best_accuracy.item()
        filename = os.path.join(self.args.work_dir,
                                'epoch{}_acc{:.2f}_model.pth.tar'.format(self.best_epoch, best_accuracy))
        self.model.load_state_dict(torch.load(filename))
        features = np.empty((0, 64))
        fr = h5py.File('../data/features'+ftype_real+'.h5', 'r')
        fl = h5py.File('../data/features'+ftype_synth+'.h5', 'r')
        frkeys = fr.keys()
        flkeys = fl.keys()
        df_save = h5py.File('../data/deepFeatures'+ftype_real+'+'+ftype_synth+'.h5', 'w')
        for i, (each_data, each_key) in enumerate(zip(data[:len(frkeys)], frkeys)):

            # get data
            each_data = np.reshape(each_data, (1, each_data.shape[0], joints, coords, 1))
            each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
            each_data = torch.from_numpy(each_data).float().to(self.device)

            # get feature
            with torch.no_grad():
                _, feature = self.model(each_data)
                fname = [each_key][0]+'_real'
                df_save.create_dataset(fname, data=feature)
                features = np.append(features, np.array(feature).reshape((1, feature.shape[0])), axis=0)
        for i, (each_data, each_key) in enumerate(zip(data[len(frkeys):], flkeys)):

            # get data
            each_data = np.reshape(each_data, (1, each_data.shape[0], joints, coords, 1))
            each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
            each_data = torch.from_numpy(each_data).float().to(self.device)

            # get feature
            with torch.no_grad():
                _, feature = self.model(each_data)
                fname = [each_key][0]+'_synth'
                df_save.create_dataset(fname, data=feature)
                features = np.append(features, np.array(feature).reshape((1, feature.shape[0])), axis=0)
        df_save.close()
        return features

    def plot_confusion_matrix(self, y_true, y_pred, epoch=None, is_test=True, subtitle=''):
        """
        Plot a confusion matrix and display the accuracy of each cell as a percentage.

        Parameters:
            y_true: ground truth labels (numpy array or list)
            y_pred: predicted results (numpy array or list), can be class indices or one-hot probabilities
            epoch: current epoch, used for file name and title display; if None, not displayed
            is_test: whether it is the test phase (used only for title identification)
            subtitle: suffix appended to the file name and printed information, e.g., '_real', '_synth', used to distinguish different subsets
        Returns:
            cm: confusion matrix (numpy array)
            accuracy: overall accuracy (float)
        """


        plt.rcParams.update({'font.size': 26})

        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred


        cm = confusion_matrix(y_true, y_pred_classes)

        accuracy = np.trace(cm) / np.sum(cm)


        row_sums = cm.sum(axis=1, keepdims=True)

        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_percent = cm.astype('float') / row_sums * 100

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
            cbar_kws={'label': 'Accuracy Percentage (%)'}
        )


        phase = "Test" if is_test else "Validation"
        epoch_str = f" - Epoch {epoch}" if epoch is not None else ""


        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()


        save_dir = self.args.work_dir
        filename = f"confusion_matrix_{'test' if is_test else 'val'}"
        if epoch is not None:
            filename += f"_epoch{epoch}"
        filename += f"{subtitle}.png"

        save_path = os.path.join(self.confusion_matrix_dir, filename)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        self.io.print_log(f'\tConfusion matrix saved to: {save_path}')
        self.io.print_log(f'\tOverall accuracy (from confusion matrix): {accuracy:.2%}')

        return cm, accuracy