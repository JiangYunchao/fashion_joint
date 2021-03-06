"""Class Solver."""
import os
import sys
import torch
import numpy as np
from numpy import linalg

from time import time
from torch import nn
from torch.nn.parallel import data_parallel
from torch.autograd import Variable
from torch.autograd import gradcheck
from transfer import Smooth1D, MovingAverage
from check_utils import check_dirs
import metrics
import torch.nn.functional as F


# TODO: To support more types of schedulars
class FashionNetSolver(object):
    """Base class for Model."""

    def __init__(self, param):
        """Use param to initialize a Solver instance.

        Parameters
        ----------
        param: parameter to initialize the solver class
            param['net']: training net
            param['triplet']: whether to use triplet loss
            param['env']: the environment for visdom
            param['visdom_title']: title for figure

        """
        import visdom
        #import torch.backends.cudnn as cudnn
        #cudnn.benchmark = True
        # param
        self.param = param
        # set network
        self.net = param.get('net')
        self.gpus = param.get('gpus')
        self.gamma = param.get('gamma', 0.02)
        # set visualizer
        self.vis = visdom.Visdom(env=param.get('env', 'main'))
        self.title = param.get('visdom_title', 'Fashion Net')
        # to display trianing log
        self.display = param.get('display', 1)
        self.save_dir = param.get('save_dir', './')
        check_dirs(self.save_dir, action='mkdir')
        # set random seed manually
        # self.random_seed = param.get('random_seed', random.randint(1, 1e4))
        # random.seed(self.random_seed)
        # torch.cuda.manual_seed_all(self.random_seed)
        self.iter = 0
        self.last_epoch = -1
        self.parallel = True if len(self.gpus) > 1 else False
        # loss function
        self.rank_loss = nn.SoftMarginLoss().cuda(device=self.gpus[0])
        #self.vse_loss = nn.MarginRankingLoss(margin=0.2).cuda(device=self.gpus[0])
        self.InitOptim()
        self.InitSchedular()
        self.InitMeter()
        self.InitHistory()

    def loss(self, posi, nega, text, img, epoch, backward=False, size_average=True):
        """Compute the loss.

        Parameters
        ----------
        posi, nega: positive, negative
        text: tuple of text vector
        img: tuple of img embedding vector
        target: target value
        backward: if do backward
        size_average: if return averaged loss

        """
        self.rank_loss.size_average = size_average
        #self.vse_loss.size_average = False
        target = torch.ones_like(posi)
        if epoch >= 0:
            lamda = 0.001
            margin = 0.2
            #text, img: Ncate * Batch * Dim
            loss_vse = torch.tensor(0.).cuda(device=self.gpus[0])
            for ipt, t in enumerate(text):
                for ipl, i in enumerate(img):
                    if ipt == ipl:    # == means compute loss with same category while != means compute loss with different category
                        num = torch.matmul(t, i.t())
                        denom_t = torch.unsqueeze(torch.norm(t, 2, 1), 1)
                        denom_i = torch.unsqueeze(torch.norm(i, 2, 1), 0)
                        denom = torch.matmul(denom_t, denom_i)
                        cos = torch.div(num, denom)
                        #print("cos:{}".format(cos))
                        #print("t:{}\ni:{}".format(t, i))

                        [rows, cols] = cos.shape  #rows==cols==batch_size
                        print(cos)
                        loss_vse += (rows * (cols-1) * margin - 2 * cos.trace() + cos.sum()) / torch.tensor(cols).float().cuda(device=self.gpus[0])
            # for i in range(rows-1):
            #     #x1 = torch.tensor([cos[i][i]]).repeat(1, rows-1)
            #     loss_item = 0
            #     for j in range(cols-1):
            #         if i == j:
            #             continue
            #         else:
            #             y = torch.tensor([1]).cuda(device=self.gpus[0])
            #             x1 = torch.unsqueeze(torch.tensor([cos[i][i]]), 0).cuda(device=self.gpus[0])
            #             x2 = torch.unsqueeze(torch.tensor([cos[i][j]]), 0).cuda(device=self.gpus[0])
            #             loss_item += self.vse_loss(x1, x2, y)
            #
            #     loss_vse += loss_item

            loss_score = self.rank_loss(posi - nega, target)
            print("loss:{:.4f},{:.4f}".format(torch.tensor(1.).cuda(device=self.gpus[0])- (lamda * loss_vse), loss_score.item()))
            loss = torch.tensor(1.).cuda(device=self.gpus[0])- (lamda * loss_vse) + loss_score
        else:
            loss = self.rank_loss(posi - nega, target)
        if backward:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.nn.utils.clip_grad_norm(self.net.parameters(), 1)
        return loss.item()



    def accuracy(self, output, size_average=True):
        """Accuracy function.

        Parameters:
        output: Type of torcu.*Tensor or cuda.*Tensor

        """
        correct = torch.gt(output.data, 0).sum()
        if size_average:
            return float(correct) / float(output.numel())
        else:
            return correct / 1.0

    def InitOptim(self):
        """Init Optimizer."""
        # set the optimizer
        lr_param = self.param['optim_param']
        optimizer = lr_param['type']
        defaults = lr_param['defaults']
        groups = lr_param['groups']
        num_gropus = len(list(self.net.children()))
        if len(groups) == 1:
            groups = groups * num_gropus
        else:
            # num of groups should match the network
            assert (num_gropus == len(groups)), '''number of groups {},
            while size of children is {}'''.format(len(groups), num_gropus)
        param_groups = []
        for child, param in zip(self.net.children(), groups):
            param_group = {'params': child.parameters()}
            param_group.update(param)
            param_group.update(defaults)
            param_groups.append(param_group)
        self.optimizer = optimizer(param_groups, **defaults)

    def set_loader(self, loader):
        """Init Loader."""
        self.dataloader = loader
        self.total_iters = len(loader)

    def InitSchedular(self):
        """Initialize LR Schedular."""
        # set learning rate policy
        lr_param = self.param['lr_param']
        lr_policy = lr_param['policy']
        param = lr_param['param']
        self.lr_scheduler = lr_policy(self.optimizer, **param)

    def InitMeter(self):
        """Initialize Meters."""
        self.meter_accuracy = MovingAverage()
        self.meter_binary = MovingAverage()
        self.meter_loss = MovingAverage()

    def InitHistory(self):
        """Initialize history."""
        self.binary_history = dict(
            train=dict(X=[], Y=[]),  # training history for binary accuracy
            test=dict(X=[], Y=[])  # test history for binary accuracy
        )
        self.accuracy_history = dict(
            train=dict(X=[], Y=[]),  # training history accuracy
            test=dict(X=[], Y=[])  # test history for accuracy
        )
        self.loss_history = dict(
            train=dict(X=[], Y=[]),  # training history for loss
            test=dict(X=[], Y=[])  # test history for loss
        )

    def one_hot(self, uidx):
        """Return one hot coding.

        Parameters
        ----------
        uidx: user ids

        Return
        ------
        udixv: one hot coding, shape of (batch_size, num_users)

        """
        num_users = self.net.num_users
        uidxv = uidx.view(-1, 1)
        uidxv = torch.zeros(uidxv.shape[0], num_users).scatter_(1, uidxv, 1.0)
        return uidxv

    def convert_to_variable(self, inputs):
        """Conver inputs to Variable.

        Parameter
        ---------
        triplet: type of inputs
        """
        device = self.gpus[0]
        num_users = self.net.num_users
        posi_text, nega_text,posi_img, nega_img, uidx = inputs
        uidxv = uidx.view(-1, 1)
        uidxv = torch.zeros(uidxv.shape[0], num_users).scatter_(1, uidxv ,1.0)

        if self.parallel:
            posi_text = tuple(Variable(v).float() for v in posi_text)
            nega_text = tuple(Variable(v).float() for v in nega_text)
            posi_img = tuple(Variable(v) for v in posi_img)
            nega_img = tuple(Variable(v) for v in nega_img)
            uidxv = Variable(uidxv)
        else:
            posi_text = tuple(Variable(v.cuda(device=device)).float() for v in posi_text)
            nega_text = tuple(Variable(v.cuda(device=device)).float() for v in nega_text)
            posi_img = tuple(Variable(v.cuda(device=device)) for v in posi_img)
            nega_img = tuple(Variable(v.cuda(device=device)) for v in nega_img)
            uidxv = Variable(uidxv.cuda(device=device))
        inputv = (posi_text, nega_text,posi_img,nega_img, uidxv)
        return inputv, uidx.view(-1)

    def forward(self, inputv):
        """Compute one batch."""
        if self.parallel:
            # model parallel
            output = data_parallel(
                self.net, inputv, self.gpus, output_device=self.gpus[0])
        else:
            output = self.net(*inputv)
        return output

    def StepEpoch(self, epoch=None):
        """Run one epoch for training net."""
        # update the epoch
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.net.train()
        latest_time = time()
        ratio = 2
        for idx, inputs in enumerate(self.dataloader):
            data_time = time() - latest_time
            posi_text, nega_text,posi_img, nega_img, uidx = inputs
            posi_text = posi_text * ratio
            posi_img = posi_img * ratio
            for i in range(ratio):
                input = posi_text[i], nega_text[i],posi_img[i],nega_img[i], uidx
                inputv, uindex = self.convert_to_variable(input)
                pscore, nscore, text, img = self.forward(inputv)
                loss = self.loss(pscore, nscore, text, img, epoch, backward=True)
                # compute accuracy
                accuracy = self.accuracy(pscore - nscore)
                binary_accuracy = 0 #self.accuracy(bpscore - bnscore)

                # record the loss and accuracy
                self.meter_loss.update(loss)
                self.meter_accuracy.update(accuracy)
                self.meter_binary.update(binary_accuracy)

                # append history
                self.append_history('train', loss, accuracy, binary_accuracy)
                #for param in self.net.parameters():
                #    print("parameter:{}\nTensorSize:".format(param.grad, param.grad.size()))
            # show results
            batch_time = time() - latest_time
            latest_time = time()
            if (self.display > 0) and (self.iter % self.display == 0):
                self.plot_accuracy(
                    self.iter, {
                        'train': self.meter_accuracy.avg,
                        'train(binary)': self.meter_binary.avg
                    })
                self.plot_loss(self.iter, {'train': self.meter_loss.avg})
                sys.stdout.write(
                    'Epoch[{}]({}): [{}]/[{}] '
                    'Batch Time {:.3f} Data Time {:.3f} '
                    'Loss {} Accuracy {} Accuracy(Binary) {} \n'.format(
                        epoch, self.iter, idx, self.total_iters, batch_time,
                        data_time, self.meter_loss, self.meter_accuracy,
                        self.meter_binary))
                sys.stdout.flush()
                try:
                    w = self.net.ulambda.weight.data[0][0]
                    print('lambda:{:.3f}'.format(w))
                except Exception as e:
                    pass
                try:
                    w = self.net.user_embdding.embdding.weight.data[0][0]
                    s = self.net.user_embdding.scale[0]
                    print('user:{:.3f} = {:.3f}*{:.3f}'.format(w * s, s, w))
                except Exception as e:
                    pass

            self.iter += 1

        self.plot_accuracy(
            self.iter, {
                'train': self.meter_accuracy.avg,
                'train(binary)': self.meter_binary.avg
            })
        self.plot_loss(self.iter, {'train': self.meter_loss.avg})
        # update scale
        scale = pow(self.net.user_embdding.scale[0]**2 + self.gamma, 0.5)
        self.net.set_scale(scale)

    def TestEpoch(self, epoch=None, verbose=False):
        """Run test epoch.

        Parameter
        ---------
        epoch: test on the result of epoch-th. epoch = -1 means the test is
            done after net initialization.
        """
        if epoch is None:
            epoch = self.last_epoch
        # test only use pair outfits
        self.net.eval()
        accuracy = loss = binary = 0.0
        latest_time = time()
        num_users = self.net.num_users
        g = lambda: [[] for u in range(num_users)]
        posi_scores, posi_binary = g(), g()
        nega_scores, nega_binary = g(), g()
        for idx, inputs in enumerate(self.dataloader):
            # compute output and loss
            inputv, uindex = self.convert_to_variable(inputs)
            data_time = time() - latest_time
            #pscore, nscore, bpscore, bnscore = self.forward(inputv)
            pscore, nscore = self.forward(inputv)
            for n, u in enumerate(uindex.view(-1)):
                posi_scores[u].append(pscore[n].data[0])
                nega_scores[u].append(nscore[n].data[0])
                #posi_binary[u].append(bpscore[n].data[0])
                #nega_binary[u].append(bnscore[n].data[0])
            loss_ = self.loss(
                pscore, nscore, backward=False, size_average=False)
            accuracy_ = self.accuracy(pscore - nscore, size_average=False)
            binary_ = 0 #self.accuracy(bpscore - bnscore, size_average=False)
            batch_time = time() - latest_time
            latest_time = time()
            if verbose:
                bsize = uindex.size(0)
                sys.stdout.write(
                    'Epoch[{}]: [{}]/[{}] '
                    'Batch Time {:.3f} Data Time {:.3f} '
                    'Loss {:.4f} Accuracy {:.3f} Accuracy(Binary) {:.3f} \n'.
                    format(epoch, idx, self.total_iters, batch_time, data_time,
                           loss_ / bsize, accuracy_ / bsize, binary_ / bsize))
                sys.stdout.flush()
            loss += loss_
            accuracy += accuracy_
            binary += binary_
            # set the target
        count = len(self.dataloader.dataset)
        accuracy /= count
        binary /= count
        loss /= count
        mean_ndcg, _ = metrics.NDCG(posi_scores, nega_scores)
        ndcg = mean_ndcg.mean()
        mean_ndcg_bianry, _ = metrics.NDCG(posi_binary, nega_binary)
        ndcg_bianry = mean_ndcg_bianry.mean()
        self.append_history('test', loss, accuracy, binary)
        sys.stdout.write(
            'Test: Epoch [{}] '
            'Loss {:.4f} Accuracy {:.4f} Accuracy(Binary) {:.4f}\n'.format(
                epoch, loss, accuracy, binary))
        sys.stdout.flush()
        self.plot_accuracy(
            self.iter, {
                'test': accuracy,
                'test(binary)': binary,
                'ndcg': ndcg,
                'ndcg(binary)': ndcg_bianry,
            })
        self.plot_loss(self.iter, {'test': loss})
        return loss, accuracy, binary

    def append_history(self, k, loss, accuracy, binary):
        """Append history."""
        # loss history
        self.loss_history[k]['X'].append(self.iter)
        self.loss_history[k]['Y'].append(loss)
        # accuracy history
        self.accuracy_history[k]['X'].append(self.iter)
        self.accuracy_history[k]['Y'].append(accuracy)
        # binary accuracy history
        self.binary_history[k]['X'].append(self.iter)
        self.binary_history[k]['Y'].append(binary)

    def save(self, label=None, suffix=''):
        """Save network and optimizer."""
        if label is None:
            label = str(self.last_epoch)
        self.save_solver(label, suffix)
        self.save_net(label, suffix)
        self.save_optim(label, suffix)

    def load(self, label, suffix=''):
        """Load the state of solver."""
        # load net state
        self.load_net_state(label, suffix)
        # load optimizer state
        self.load_optim_state(label, suffix)
        # load solver state
        self.load_solver_state(label, suffix)

    def resume(self, label, suffix=''):
        """Resume from the latest state."""
        self.load(label, suffix)
        # new figure
        self.init_plot()
        win_size = MovingAverage.win_size
        # train loss
        self.update_trace(
            Smooth1D(self.loss_history['train']['X'], win_size),
            Smooth1D(self.loss_history['train']['Y'], win_size), 'train',
            self.loss_win)
        # test loss
        self.update_trace(
            np.array(self.loss_history['test']['X']),
            np.array(self.loss_history['test']['Y']), 'test', self.loss_win)
        # train accuracy
        self.update_trace(
            Smooth1D(self.accuracy_history['train']['X'], win_size),
            Smooth1D(self.accuracy_history['train']['Y'], win_size), 'train',
            self.acc_win)
        # test accuracy
        self.update_trace(
            np.array(self.accuracy_history['test']['X']),
            np.array(self.accuracy_history['test']['Y']), 'test', self.acc_win)
        # train binary accuracy
        self.update_trace(
            Smooth1D(self.binary_history['train']['X'], win_size),
            Smooth1D(self.binary_history['train']['Y'], win_size),
            'train(binary)', self.acc_win)
        # test binary accuracy
        self.update_trace(
            np.array(self.binary_history['test']['X']),
            np.array(self.binary_history['test']['Y']), 'test(binary)',
            self.acc_win)

    def save_solver(self, label, suffix=''):
        """Save the state of solver.

        State of solver:

                state = {
                    'param': param,
                    'solver_state': solver_state,
                    'schedular_state': schedular_state,
                }

        """
        param = {
            k: v
            for k, v in self.param.items() if k not in ['net', 'loader']
        }
        solver_state = dict(
            iter=self.iter,
            last_epoch=self.last_epoch,
            # random_seed=self.random_seed,
            save_dir=self.save_dir,
            meter_loss=self.meter_loss,
            meter_accuracy=self.meter_accuracy,
            meter_binary=self.meter_binary,
            loss_history=self.loss_history,
            accuracy_history=self.accuracy_history,
            binary_history=self.binary_history)
        if hasattr(self, 'loss_win') and hasattr(self, 'acc_win'):
            vis_state = dict(loss_win=self.loss_win, acc_win=self.acc_win)
            solver_state.update(vis_state)
        schedular_state = {
            k: v
            for k, v in self.lr_scheduler.__dict__.items()
            if k not in ['is_better', 'optimizer'] and not callable(v)
        }
        state = {
            'param': param,
            'solver_state': solver_state,
            'schedular_state': schedular_state,
        }
        solver_path = self.format_filepath('solver', label, suffix)
        torch.save(state, solver_path)

    def save_net(self, label, suffix=''):
        """Save the net's state."""
        model_path = self.format_filepath('net', label, suffix)
        torch.save(self.net.state_dict(), model_path)

    def save_optim(self, label, suffix=''):
        """Save the state of optimizer."""
        optim_path = self.format_filepath('optim', label, suffix)
        torch.save(self.optimizer.state_dict(), optim_path)

    def load_solver_state(self, label, suffix=''):
        """Load the state of solver."""
        solver_path = self.format_filepath('solver', label, suffix)
        state = torch.load(solver_path)
        self.__dict__.update(state['solver_state'])
        for k, v in state['param'].items():
            if self.param[k] != v:
                print('{} is different to previous setting:\n'
                      'Now: {} \nPrevious:{}'.format(k, self.param[k], v))
        # self.param.update(state['param'])
        print('Drop the previous scheduler setting:'
              'which is \n{}'.format(state['schedular_state']))
        # self.lr_scheduler.__dict__.update(state['schedular_state'])

    def load_net_state(self, label, suffix=''):
        """Load the state of net."""
        model_path = self.format_filepath('net', label, suffix)
        self.net.load_state_dict(torch.load(model_path))

    def load_optim_state(self, label, suffix=''):
        """Load the state of optimizer."""
        optim_path = self.format_filepath('optim', label, suffix)
        self.optimizer.load_state_dict(torch.load(optim_path))

    def format_filepath(self, name, label, suffix=''):
        """Return filepath.

        The path format of checkpoint is 'name_label_suffix.pth' or
        'name_label.pth'.
        Parameters
        ----------
        name: Name of state like 'net', 'optim' etc.
        label: To control the checkpoint file name. Typical chioces are
            epoch, 'best', 'latest'.
        suffix: Extra string to control the checkpoint file name.

        """
        suffix = '_' + suffix if suffix else ''
        filename = '%s_%s%s.pth' % (name, label, suffix)
        filepath = os.path.join(self.save_dir, filename)
        return filepath

    def init_plot(self):
        """Initialize plots for accuracy and loss."""
        legend = [
            'train',
            'test',
        ]
        self.loss_win = self.vis.line(
            X=np.zeros(1),
            Y=np.ones((1, len(legend))) * np.nan,
            opts={
                'title': self.title + ' (Loss)',
                'ylabel': 'loss',
                'xlabel': 'iterations',
                'legend': legend,
            })
        legend = [
            'train',
            'test',
            'train(binary)',
            'test(binary)',
            'ndcg',
            'ndcg(binary)',
        ]
        self.acc_win = self.vis.line(
            X=np.zeros(1),
            Y=np.ones((1, len(legend))) * np.nan,
            opts={
                'title': self.title + ' (Accuracy)',
                'ylabel': 'accuracy',
                'xlabel': 'iterations',
                'legend': legend,
            })

    def plot_accuracy(self, x, data):
        """Append accuracy is visdom."""
        if not hasattr(self, 'loss_win') or not hasattr(self, 'acc_win'):
            self.init_plot()
        for k, y in data.items():
            self.update_trace(np.array([x]), np.array([y]), k, self.acc_win)

    def plot_loss(self, x, data):
        """Append loss history in visdom."""
        if not hasattr(self, 'loss_win') or not hasattr(self, 'acc_win'):
            self.init_plot()
        for k, y in data.items():
            self.update_trace(np.array([x]), np.array([y]), k, self.loss_win)

    def update_trace(self, x, y, name, win):
        """Update single trace."""
        if self.vis.check_connection():
            self.vis.line(
                X=x,
                Y=y,
                update='append',
                name=name,
                win=win,
                opts={
                    'showlegend': True
                })
