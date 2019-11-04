import argparse


def obtain_search_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes', 'kd'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--autodeeplab', type=str, default='search',
                        choices=['search', 'train'])
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--load-parallel', type=int, default=0)
    parser.add_argument('--clean-module', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=321,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=321,
                        help='crop image size')
    parser.add_argument('--resize', type=int, default=512,
                        help='resize image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--block_multiplier', type=int, default=1)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--alpha_epoch', type=int, default=20,
                        metavar='N', help='epoch to start training alphas')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--arch-lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate for alpha and beta in architect searching process')

    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3,
                        metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # This flag controls the model parallel
    parser.add_argument('--model-parallel', type=bool, default=False,
                        help='whether to use model parallel, it can help match the parameter in paper use two smaller gpus with 8G-12G memory')
    parser.add_argument('--use-ABN', type=bool,
                        default=True,
                        help='whether to use Activated Batch Normalization, it can save about 25 percent gpu memory in retrain')
    parser.add_argument('--use-distribute', type=bool, default=False,
                        help='whether to use distributed in pytorch, it can help multi-gpus workload more balancing')
    parser.add_argument('--steps', type=int, default=5, help='number of nodes in a cell')
    parser.add_argument('--filter-multiplier', type=int, default=32)
    parser.add_argument('--encoder-layer', type=int, default=12)
    parser.add_argument('--decoder-layer', type=int, default=6)
    parser.add_argument('--Output-Focal', type=bool, default=False)
    parser.add_argument('--Model-Parallel', type=bool, default=False)
    parser.add_argument('--Full-autoencoder', type=bool, default=False)
    parser.add_argument('--pre-train', type=bool, default=False)
    parser.add_argument('--local_rank', dest='local_rank', type=int, default=-1)
    parser.add_argument('--manualSeed', type=int, default=10, )
    parser.add_argument('--dist', type=bool, default=True )
    parser.add_argument('--save-pth', type=str, default='log')
    args = parser.parse_args()
    return args


def obtain_retrain_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--resize', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # training hyper params
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--filter_multiplier', type=int, default=20)
    parser.add_argument('--block_multiplier', type=int, default=1)
    parser.add_argument('--autodeeplab', type=str, default='train',
                        choices=['search', 'train'])
    parser.add_argument('--load-parallel', type=int, default=0)
    # TODO: CHECK THAT THEY EVEN DO THIS FOR THE MODEL IN THE PAPER
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--use-ABN', type=bool,
                        default=False,
                        help='whether to use Activated Batch Normalization, it can save about 25 percent gpu memory in retrain')
    parser.add_argument('--use-distribute', type=bool, default=False,
                        help='whether to use distributed in pytorch, it can help multi-gpus workload more balancing')

    parser.add_argument('--encoder-layer', type=int, default=12)
    parser.add_argument('--train', action='store_true', default=True,
                        help='training mode')
    parser.add_argument('--exp', type=str, default='bnlr7e-3',
                        help='name of experiment')
    parser.add_argument('--gpu', type=int, default=0,
                        help='test time gpu device id')
    parser.add_argument('--backbone', type=str, default='autodeeplab', help='resnet101')
    parser.add_argument('--groups', type=int, default=None,
                        help='num of groups for group normalization')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--base_lr', type=float, default=0.05,
                        help='base learning rate')
    parser.add_argument('--last_mult', type=float, default=1.0,
                        help='learning rate multiplier for last layers')
    parser.add_argument('--scratch', action='store_true', default=False,
                        help='train from scratch')
    parser.add_argument('--freeze_bn', action='store_true', default=False,
                        help='freeze batch normalization parameters')
    parser.add_argument('--weight_std', action='store_true', default=False,
                        help='weight standardization')
    parser.add_argument('--beta', action='store_true', default=False,
                        help='resnet101 beta')
    parser.add_argument('--crop_size', type=int, default=769,
                        help='image crop size')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--local_rank', dest='local_rank', type=int, default=-1, )
    parser.add_argument('--manualSeed', type=int, default=10, )
    args = parser.parse_args()
    return args
