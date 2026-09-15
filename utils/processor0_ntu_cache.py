import h5py
import math
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from net import classifier0_ntu_cache
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
        if self.epoch < self.warmup_epoch:          # 仍在 warmup 阶段
            lr = self.start_lr + \
                 (self.base_lr - self.start_lr) * self.epoch / self.warmup_epoch
        else:                                       # 进入指数衰减
            lr = self.base_lr * (self.gamma ** (self.epoch - self.warmup_epoch))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.epoch += 1





class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, data_loader, C, num_classes, graph_dict, device='cuda:3'):

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


        start_time_str = datetime.datetime.now().strftime("%m%d_%H%M")
        self.confusion_matrix_dir = os.path.join(self.args.work_dir, f'confusion_matrices_{start_time_str}')
        if not os.path.exists(self.confusion_matrix_dir):
            os.makedirs(self.confusion_matrix_dir)
            self.io.print_log(f'Created confusion matrix directory: {self.confusion_matrix_dir}')
            print(self.confusion_matrix_dir)


        # model
        if not os.path.isdir(self.args.work_dir):
            os.mkdir(self.args.work_dir)
        self.model = classifier0_ntu_cache.Classifier(C, num_classes, graph_dict)
        self.model.cuda('cuda:3')
        self.model.apply(weights_init)


        self.loss = nn.CrossEntropyLoss().to('cuda:3')
        #weight = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        #weight = (weight / weight.sum()) * len(weight)  # 归一化到近似 1
        #self.loss = nn.CrossEntropyLoss(weight=weight).to('cuda:3')


        self.best_loss = math.inf
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
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
                                      base_lr=self.lr,  # 目标最大 LR
                                      start_lr=self.lr*0.1,  # 从 0.001 开始
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

        for data, label, lengths in loader:
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
            # ===============================

            self.show_iter_info()
            self.meta_info['iter'] += 1

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

        save_ca = True
        for data, label, lengths in loader:

            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)


            # inference
            with torch.no_grad():
                output, _ = self.model(data, lengths)


            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()


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
                #if (self.best_accuracy.item() > 85.00):
                #    show_ca = True
                self.per_test(is_show_ca=show_ca, epoch=epoch)
                self.io.print_log('Done.')

            # save model and weights
            if self.accuracy_updated:
                if self.epoch_info['mean_loss'] < self.best_loss:
                    self.best_loss = self.epoch_info['mean_loss']
                    self.best_epoch = epoch
                    if (self.best_accuracy.item() > 73.00):
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
