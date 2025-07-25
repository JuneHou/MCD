import os
import sys
import json
import argparse
from pprint import pprint
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils.utils as utils
import utils.config as config
from train_MCD import train, evaluate
import modules.base_model_MCD as base_model
from modules.base_model_MCD import Bia_Model
from utils.dataset_MCD import Dictionary, VQAFeatureDataset
from utils.losses import Plain
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of running epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate for adamax')
    parser.add_argument('--loss-fn', type=str, default='Plain',
                        help='chosen loss function')
    parser.add_argument('--num-hid', type=int, default=1024,
                        help='number of dimension in last layer')
    parser.add_argument('--model', type=str, default='baseline_newatt',
                        help='model structure')
    parser.add_argument('--name', type=str, default='exp0.pth',
                        help='saved model name')
    parser.add_argument('--name-new', type=str, default=None,
                        help='combine with fine-tune')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('--fine-tune', action='store_true',
                        help='fine tuning with our loss')
    parser.add_argument('--resume', action='store_true',
                        help='whether resume from checkpoint')
    parser.add_argument('--not-save', action='store_true',
                        help='do not overwrite the old model')
    parser.add_argument('--test', dest='test_only', action='store_true',
                        help='test one time')
    parser.add_argument('--eval-only', action='store_true',
                        help='evaluate on the val set one time')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    parser.add_argument("--wandb", action='store_true',
                        help='use wandb for logging')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse args early to set GPU before any CUDA operations
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0', help='gpu card ID')
    args, unknown = parser.parse_known_args()
    
    # Set GPU before importing torch
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    args = parse_args()
    print(args)
    print_keys = ['cp_data', 'version', 'train_set',
                  'loss_type', 'use_cos', 'entropy', 'scale']
    print_dict = {key: getattr(config, key) for key in print_keys}
    pprint(print_dict, width=150)

    if args.wandb:
        wandb.init(project="MCD", name=args.name, config=print_dict)
        wandb.config.update(args)
        print(f"Initialized wandb project: MCD, run name: {args.name}")

    cudnn.benchmark = True
    
    print(f"Using GPU: {args.gpu} (will appear as cuda:0 due to CUDA_VISIBLE_DEVICES)")

    seed = 1111
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    if 'log' not in args.name:
        args.name = '/data/wang/junh/githubs/MCD/MCD/logs/' + args.name
    if args.test_only or args.fine_tune or args.eval_only:
        args.resume = True
    if args.resume and not args.name:
        raise ValueError("Resuming requires folder name!")
    if args.resume:
        logs = torch.load(args.name)
        print("loading logs from {}".format(args.name))

    # ------------------------DATASET CREATION--------------------
    if config.version=='v2':
        dictionary = Dictionary.load_from_file(config.dict_path)
    else:
        dictionary = Dictionary.load_from_file('../dictionary_v1.pkl')
    if args.test_only:
        eval_dset = VQAFeatureDataset('test', dictionary)
    else:
        train_dset = VQAFeatureDataset('train', dictionary)
        eval_dset = VQAFeatureDataset('val', dictionary)
    if config.train_set == 'train+val' and not args.test_only:
        train_dset = train_dset + eval_dset
        eval_dset = VQAFeatureDataset('test', dictionary)
    if args.eval_only:
        eval_dset = VQAFeatureDataset('val', dictionary)

    tb_count = 0

    if not config.train_set == 'train+val' and 'LM' in args.loss_fn:
        utils.append_bias(train_dset, eval_dset, len(eval_dset.label2ans))

    # ------------------------MODEL CREATION------------------------
    constructor = 'build_{}'.format(args.model)
    model, metric_fc = getattr(base_model, constructor)(eval_dset, args.num_hid)
    model = model.cuda()
    metric_fc = metric_fc.cuda()
    bias_model = Bia_Model(num_hid=1024, dataset=train_dset).cuda()
    if config.version=='v2':
        model.w_emb.init_embedding('/data/wang/junh/datasets/vqa-cp-v2/glove6b_init_300d.npy')
        bias_model.w_emb.init_embedding('/data/wang/junh/datasets/vqa-cp-v2/glove6b_init_300d.npy')
    else:
        model.w_emb.init_embedding('../glove6b_init_300d_v1.npy')
        bias_model.w_emb.init_embedding('../glove6b_init_300d_v1.npy')

    # model = nn.DataParallel(model).cuda()
    optim = torch.optim.Adamax([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr)
    optim_G = torch.optim.Adamax(filter(lambda p: p.requires_grad, bias_model.parameters()), lr=0.001)

    if args.loss_fn == 'Plain':
        loss_fn = Plain()
    else:
        raise RuntimeError('not implement for {}'.format(args.loss_fn))

    # ------------------------STATE CREATION------------------------
    eval_score, best_val_score, start_epoch, best_epoch = 0.0, 0.0, 0, 0
    tracker = utils.Tracker()
    if args.resume:
        model.load_state_dict(logs['model_state'])
        metric_fc.load_state_dict(logs['margin_model_state'])
        optim.load_state_dict(logs['optim_state'])
        if 'loss_state' in logs:
            loss_fn.load_state_dict(logs['loss_state'])
        start_epoch = logs['epoch']
        best_epoch = logs['epoch']
        best_val_score = logs['best_val_score']
        if args.fine_tune:
            print('best accuracy is {:.2f} in baseline'.format(100 * best_val_score))
            args.epochs = start_epoch + 10 # 10 more epochs
            for params in optim.param_groups:
                params['lr'] = config.ft_lr

            # if you want save your model with a new name
            if args.name_new:
                if 'log' not in args.name_new:
                    args.name = 'logs/' + args.name_new
                else:
                    args.name = args.name_new

    eval_loader = DataLoader(eval_dset,
                    args.batch_size, shuffle=False, num_workers=4)
    if args.test_only or args.eval_only:
        model.eval()
        metric_fc.eval()
        evaluate(model, metric_fc, eval_loader, write=True)
    else:
        train_loader = DataLoader(
            train_dset, args.batch_size, shuffle=True, num_workers=4)
        training_start_time = time.time()
        for epoch in range(start_epoch, args.epochs):
            print("training epoch {:03d}".format(epoch))
            tb_count = train(model, metric_fc, bias_model, optim, optim_G, train_loader, loss_fn, tracker, tb_count, epoch, args)

            if not (config.train_set == 'train+val' and epoch in range(args.epochs - 3)):
                # save for the last three epochs
                write = True if config.train_set == 'train+val' else False
                print("validating after epoch {:03d}".format(epoch))
                model.train(False)
                metric_fc.train(False)
                if config.cp_data:
                    if config.version=='v2':
                        with open('/data/wang/junh/githubs/MCD/MCD/util/qid2type_cpv2.json', 'r') as f:
                            qid2type = json.load(f)
                    else:
                        with open('../util/qid2type_cpv1.json', 'r') as f:
                            qid2type = json.load(f)
                else:
                    with open('../util/qid2type_v2.json', 'r') as f:
                        qid2type = json.load(f)
                eval_score, score_yesno, score_other, score_number = evaluate(model, metric_fc, eval_loader, qid2type, epoch, write=write)
                model.train(True)
                metric_fc.train(True)
                print("eval score: {:.2f} ".format(100 * eval_score))
                print("yn score: {:.2f} ".format(100 * float(score_yesno)))
                print("num score: {:.2f} ".format(100 * float(score_number)))
                print("other score: {:.2f} ".format(100 * float(score_other)))

                if args.wandb:
                    wandb.log({
                        "eval_score": eval_score * 100,
                        "yn_score": score_yesno * 100,
                        "num_score": score_number * 100,
                        "other_score": score_other * 100
                    })

                        

            if eval_score > best_val_score:
                best_val_score = eval_score
                best_epoch = epoch
                results = {
                    'epoch': epoch + 1,
                    'best_val_score': best_val_score,
                    'model_state': model.state_dict(),
                    'optim_state': optim.state_dict(),
                    'loss_state': loss_fn.state_dict(),
                    'margin_model_state': metric_fc.state_dict(),
                    'results': {
                        'eval_score': eval_score,
                        'yn_score': score_yesno,
                        'num_score': score_number,
                        'other_score': score_other
                    }
                }
                print("best accuracy {:.2f} on epoch {:03d}".format(
                    100 * best_val_score, best_epoch))
            if not args.not_save:
                torch.save(results, args.name)
        
        # Training completed
        training_time = time.time() - training_start_time
        print("Training completed!")
        print("best accuracy {:.2f} on epoch {:03d}".format(100 * best_val_score, best_epoch))
        print(f"Total training time: {training_time/3600:.2f} hours")

        # print("best accuracy {:.2f} on epoch {:03d}".format(
        #     100 * best_val_score, best_epoch))
