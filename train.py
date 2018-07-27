#!/usr/bin/env python
"""Fashion Net Training Script."""
import os
import torch
import torchvision
import numpy as np

import options
import progress
import models
import metrics
from fashionset import PolyvoreDataLoader
from torch.nn.parallel import data_parallel
from torch.autograd import Variable
from progress import ProgressBar


def show_dict(opt, num=0):
    """Show hierarchal options."""
    indent = '  ' * num
    for k, v in opt.items():
        if isinstance(v, dict):
            print('{}{} : {{'.format(indent, k))
            show_dict(v, num + 1)
            print('{}}}'.format(indent))
        else:
            print('{}{} : {},'.format(indent, k, v))


def show_options(param, args):
    """Print log hearder."""
    _, columns = map(int, os.popen('stty size', 'r').read().split())
    dots = '-' * ((columns - 9) // 2) if columns > 0 else ''
    print(dots + '   Net   ' + dots)
    show_dict(param['net']._modules)
    print(dots + '  Param  ' + dots)
    info = {k: v for k, v in param.items() if k not in ['net']}
    show_dict(info)
    print(dots + ' Options ' + dots)
    show_dict(vars(args))
    print(dots + '   End   ' + dots)


def get_optim_defaults(args):
    """Get the default for given optimier."""
    if args.optim == 'Adadelta':
        defaults = dict(rho=args.rho, eps=args.eps)
    elif args.optim == 'Adagrad':
        defaults = dict(lr_decay=args.lr_decay)
    elif args.optim in ['Adam', 'Adamax']:
        betas = tuple(float(beta) for beta in args.betas.split(','))
        defaults = dict(betas=betas, eps=args.eps)
    elif args.optim == 'SparseAdam':
        betas = [float(beta) for beta in args.betas.split(',')]
        defaults = dict(betas=betas, eps=args.eps)
    elif args.optim == 'ASGD':
        defaults = dict(lambd=args.lambd, alpha=args.alpha, t0=args.t0)
    elif args.optim == 'RMSprop':
        defaults = dict(alpha=args.alpha, eps=args.eps, momentum=args.momentum)
    elif args.optim == 'Rprop':
        etas = [float(eta) for eta in args.etas.split(',')]
        step_sizes = [float(st) for st in args.step_sizes.split(',')]
        defaults = dict(etas=etas, step_sizes=step_sizes)
    elif args.optim == 'SGD':
        defaults = dict(momentum=args.momentum)
    else:
        raise KeyError
    return defaults


def get_dataset(args, phase=None, evaluate=False):
    """Get date loader.

    Parameters
    ----------
    args.image_dir: folder for images
    args.tuple_dir: folder for tuple file
    args.list_dir: folder for image list
    args.hard_mode: enable hard negative examples
    args.match_mode: enable negative example for match mode
    evaluate: use data for evalution
    phase: use corresponding data (only train data support `triplet`)

    """
    variable = args.variable
    triplet = False#args.triplet if phase == 'train' else False
    data = PolyvoreDataLoader(
        args,
        phase=phase,
        variable=variable,
        triplet=triplet,
        evaluate=evaluate)
#    if args.hard_mode:
#        data.loader.dataset.hard_mode_on()
#    elif args.match_mode:
#        data.loader.dataset.match_mode_on()
#    else:
#        data.loader.dataset.random_mode_on()
    return data


def get_optim_param(args):
    """Get the optimizer setting for given optimier."""
    optim = args.optim
    if optim == 'Adadelta':
        defaults = dict(rho=args.rho, eps=args.eps)
    elif optim == 'Adagrad':
        defaults = dict(lr_decay=args.lr_decay)
    elif optim in ['Adam', 'Adamax']:
        betas = tuple(float(beta) for beta in args.betas.split(','))
        defaults = dict(betas=betas, eps=args.eps)
    elif optim == 'SparseAdam':
        betas = [float(beta) for beta in args.betas.split(',')]
        defaults = dict(betas=betas, eps=args.eps)
    elif optim == 'ASGD':
        defaults = dict(lambd=args.lambd, alpha=args.alpha, t0=args.t0)
    elif optim == 'RMSprop':
        defaults = dict(alpha=args.alpha, eps=args.eps, momentum=args.momentum)
    elif optim == 'Rprop':
        etas = [float(eta) for eta in args.etas.split(',')]
        step_sizes = [float(st) for st in args.step_sizes.split(',')]
        defaults = dict(etas=etas, step_sizes=step_sizes)
    elif optim == 'SGD':
        defaults = dict(momentum=args.momentum)
    else:
        raise KeyError
    # learning rates and weight decay
    num_lrs, lrs = len(args.lrs), args.lrs
    num_wds, wds = len(args.weight_decay), args.weight_decay
    if num_lrs == 1 and num_wds > 1:
        lrs = lrs * num_wds
    elif num_lrs > 1 and num_wds == 1:
        wds = wds * num_lrs
    elif num_lrs > 1 and num_wds > 1:
        assert (num_lrs == num_wds)
    param_groups = [dict(lr=lr, weight_decay=wd) for lr, wd in zip(lrs, wds)]
    optimizer = options.enum_optim[optim]
    return dict(type=optimizer, groups=param_groups, defaults=defaults)


def get_lr_param(args):
    """Get parameters for lr shcedular."""
    lr_param = dict(
        policy=options.enum_lr_policy[args.lr_policy],
        param=dict(
            mode='max',
            cooldown=args.cooldown,
            factor=args.factor,
            patience=args.patience,
            threshold=args.threshold,
            verbose=True))
    return lr_param


def get_net(args, num_users):
    """Get network."""
    # triplet mode only support for training
    triplet = False #if args.evaluate else args.triplet
    config = dict(
        single=args.single,  # whether to use single encoder
        #binary01=args.binary01,  # whether to use {0,1} code
        #triplet=triplet,  # whether to add triplet loss
        #scale_tanh=False #args.scale_tanh,
    )
    arch = options.enum_arch[args.arch]
    net = arch(num_users, args.dim, **config)
    if args.pre_trained:
        net.load_state_dict(torch.load(args.pre_trained))
    net.cuda(device=args.gpus[0])
    return net


def get_solver(args, net):
    """Get data loader."""
    # solver parameters
    param = dict(
        net=net,
        env=args.env,
        gpus=args.gpus,
        gamma=args.gamma,
        #triplet=args.triplet,
        display=args.display,
        save_dir=args.save_dir,
        visdom_title=args.visdom_title,
        lr_param=get_lr_param(args),
        optim_param=get_optim_param(args),
    )
    show_options(param, args)
    solver = models.FashionNetSolver(param)
    if args.resume:
        solver.resume(label=args.resume)
    return solver


def validate(solver, data, best, suffix, verbose=False):
    """Validate."""
    solver.set_loader(data.loader)
    loss, accuracy, binary = solver.TestEpoch(verbose=verbose)
    solver.lr_scheduler.step(accuracy, epoch=solver.last_epoch)
    if accuracy > best:
        best = accuracy
        print('Best model: loss {:.4f}, '
              'accuracy {:.4f}({:.4f})'.format(loss, accuracy, binary))
        solver.save_net('best', suffix=suffix)
    return best


def test(args, ratio):
    """Test fashion net."""
    save_file = os.path.join(args.save_dir, args.suffix + '.npz')
    eval_data = get_dataset(args, phase=args.phase, evaluate=True)
    num_users = eval_data.num_users
    net = get_net(args, num_users).eval()
    #for parameter in net.parameters():
    #    print("param:", parameter.data)
    # test_ndcg(net, eval_data, num_users, ratio, save_file, args.gpus)
    # eval_data = get_dataset(args, phase=args.phase)
    net.zero_uscores = True
    test_accuracy(net, eval_data, num_users, ratio, save_file, args.gpus)
    net.zero_uscores = False
    net.zero_iscores = True
    test_accuracy(net, eval_data, num_users, ratio, save_file, args.gpus)
    net.zero_iscores = False
    test_accuracy(net, eval_data, num_users, ratio, save_file, args.gpus)
    eval_data = get_dataset(args, phase=args.phase, evaluate=True)
    test_ndcg(net, eval_data, num_users, ratio,save_file, args.gpus)

def test_accuracy(net, data, num_users, ratio, save_file=None, gpus=[0]):
    """Show test accuracy."""
    net.eval()
    accuracy = binary = 0.0
    parallel = len(gpus) > 1
    dtype = torch.FloatTensor if parallel else torch.cuda.FloatTensor
    total_iters = len(data.loader)
    for idx, input in enumerate(data.loader):
        # compute output and loss
        posi_text, nega_text, posi_img, nega_img, uidx = input
        # convert to Variable
        posi_text = posi_text * ratio
        posi_img = posi_img *ratio
        accuracy_ = 0
        binary_ = 0
        for i in range(ratio):
            p_text = tuple(Variable(v.type(dtype)) for v in posi_text[i])
            n_text = tuple(Variable(v.type(dtype)) for v in nega_text[i])
            p_img = tuple(Variable(v.type(dtype)) for v in posi_img[i])
            n_img = tuple(Variable(v.type(dtype)) for v in nega_img[i])
            uidx = uidx.view(-1, 1)
            batch_size = uidx.size(0)
            uidxv = torch.zeros(batch_size, num_users).scatter_(1, uidx, 1.0)
            uidxv = Variable(uidxv.type(dtype))
            posiv = (p_text, p_img, uidxv)
            negav = (n_text, n_img, uidxv)
            # compute gradient and do Optimizer step
            if parallel:
                # model parallel
                pscore, bpscore = data_parallel(net, posiv, gpus)
                nscore, bnscore = data_parallel(net, negav, gpus)
            else:
                pscore, bpscore = net(*posiv)
                nscore, bnscore = net(*negav)
            accuracy_ += net.accuracy(pscore - nscore)#, size_average=False)
            binary_ += net.accuracy(bpscore - bnscore)#, size_average=False)
        print('Batch [{}]/[{}] Accuracy {:.3f} Accuracy(Binary) {:.3f} \n'.
              format(idx, total_iters, accuracy_ / ratio,
                     binary_ / (ratio * batch_size)))
        accuracy += accuracy_ / ratio
        binary += binary_ / ratio
    count = len(data.loader.dataset)
    accuracy /= total_iters
    binary /= count
    print('Average accuracy: {}, Binary Accuracy: {}'.format(accuracy, binary))
    # save results
    if net.zero_iscores:
        results = dict(uaccuracy=accuracy, ubinary=binary)
    elif net.zero_uscores:
        results = dict(iaccuracy=accuracy, ibinary=binary)
    else:
        results = dict(accuracy=accuracy, binary=binary)
    if os.path.exists(save_file):
        results.update(np.load(save_file))
    np.savez(save_file, **results)


def test_ndcg(net, data, num_users, ratio, save_file=None, gpus=[0]):
    """Evaluate net."""
    progress = ProgressBar()
    posi_scores = [[] for u in range(num_users)]
    posi_binary = [[] for u in range(num_users)]

    net.eval()
    parallel = len(gpus) > 1
    dtype = torch.FloatTensor if parallel else torch.cuda.FloatTensor
    #data.loader.dataset.set_to_posi()
    progress.reset(len(data.loader), messsage='Computing postiive outfits')
    for idx, inputv in enumerate(data.loader):
        items_text, nega_text, items_img, nega_img, uidx = inputv
        text = tuple(Variable(v.type(dtype)) for v in items_text[0])
        img = tuple(Variable(v.type(dtype)) for v in items_img[0])
        uidx = uidx.view(-1, 1)
        uidxv = torch.zeros(uidx.shape[0], num_users).scatter_(1, uidx, 1.0)
        uidxv = Variable(uidxv.type(dtype))
        inputv = (text, img, uidxv)
        if parallel:
            scores, binary = data_parallel(net, inputv, gpus)
        else:
            scores, binary = net(*inputv)
        for n, u in enumerate(uidx.view(-1)):
            posi_binary[u].append(binary[n].item())
            posi_scores[u].append(scores[n].data[0])
        progress.forward()
    progress.end()
    # compute scores for negative outfits
    nega_scores = [[] for u in range(num_users)]
    nega_binary = [[] for u in range(num_users)]
    #data.loader.dataset.set_to_nega(ratio=6)
    progress.reset(len(data.loader), messsage='Computing negative outfits')
    for idx, inputv in enumerate(data.loader):
        posi_text, items_text, posi_img, items_img, uidx = inputv
        for i in range(ratio):
            text = tuple(Variable(v.type(dtype)) for v in items_text[i])
            img = tuple(Variable(v.type(dtype)) for v in items_img[i])
            uidx = uidx.view(-1, 1)
            uidxv = torch.zeros(uidx.shape[0], num_users).scatter_(1, uidx, 1.0)
            uidxv = Variable(uidxv.type(dtype))
            inputv = (text,img, uidxv)
            if parallel:
                scores, binary = data_parallel(net, inputv, gpus)
            else:
                scores, binary = net(*inputv)
            for n, u in enumerate(uidx.view(-1)):
                nega_binary[u].append(binary[n].data[0])
                nega_scores[u].append(scores[n].data[0])
        progress.forward()
    progress.end()
    mean_ndcg, avg_ndcg = metrics.NDCG(posi_scores, nega_scores)
    mean_ndcg_bianry, avg_ndcg_binary = metrics.NDCG(
        posi_binary, nega_binary)
    aucs, mean_auc = metrics.ROC(posi_scores, nega_scores)
    aucs_binary, mean_auc_binary = metrics.ROC(posi_binary, nega_binary)
    results = dict(
        mean_ndcg=mean_ndcg,
        avg_ndcg=avg_ndcg,
        mean_ndcg_bianry=mean_ndcg_bianry,
        avg_ndcg_binary=avg_ndcg_binary,
        aucs=aucs,
        mean_auc=mean_auc,
        aucs_binary=aucs_binary,
        mean_auc_binary=mean_auc_binary)
    print('avg_mean_ndcg:{} avg_mean_auc:{}'.format(
        mean_ndcg.mean(), mean_auc))
    # save results
    if os.path.exists(save_file):
        results.update(np.load(save_file))
    np.savez(save_file, **results)


def step(solver, data, suffix, epoch=None, coord='all'):
    """Traing one epoch with configuration.

    We will save 'net_latest_suffix.pth'
    Parameters
    ----------
    epoch: start epoch
    coord: whether to to use coordinate descent
    suffix: checkpoint suffix

    """
    suffix = solver.net.name() if suffix is None else suffix
    # coordinate descent
    solver.net.active_all_param()
    if coord == 'user':
        solver.net.freeze_item_param()
        #data.loader.dataset.hard_mode_on()
    elif coord == 'item':
        solver.net.freeze_user_param()
        #data.loader.dataset.random_mode_on()
    else:
        solver.net.active_all_param()
    solver.set_loader(data.loader)
    solver.StepEpoch(epoch)
    solver.save()
    solver.save('latest', suffix=suffix)


def train(args):
    """Train net."""
    best = 0
    suffix = args.suffix
    train_data = get_dataset(args, phase='train', evaluate=False)
    #val_data = get_dataset(args, phase='val', evaluate=False)
    #assert (train_data.num_users == val_data.num_users)
    num_users = train_data.num_users
    net = get_net(args, num_users)
    solver = get_solver(args, net)
    if args.coord:
        while solver.last_epoch < args.epochs * 2:
            step(solver, train_data, suffix, coord='user')
            scale = solver.net.user_embdding.scale[0] ** 2
            scale -= args.gamma
            solver.net.set_scale(scale)
            step(solver, train_data, suffix, coord='item')
            #best = validate(solver, val_data, best, suffix, verbose=True)
    else:
        while solver.last_epoch < args.epochs:
            step(solver, train_data, suffix, coord='all')
            #best = validate(solver, val_data, best, suffix, verbose=True)


if __name__ == '__main__':
    args = options.parse_args()
    #print(args)
    ratio = 6
    if args.evaluate:
        test(args, ratio)
    else:
        train(args)
