import argparse

parser = argparse.ArgumentParser(description='BDT project arguments')

# Training settings
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--min-lr', type=float, default=1e-6,
                    help='minimal learning rate (default: 1.0)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='MT',
                    help='momentum (default: 0.9)')
parser.add_argument('--num-workers', type=int, default=-1, metavar='NW',
                    help='number of workers (default: 8)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--step-size', type=float, default=10,
                    help='Step size to update learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--adam', action='store_true', default=False,
                    help='using adam optim')
parser.add_argument('--adamW', action='store_true', default=False,
                    help='using adamW optim')
parser.add_argument('--sgd', action='store_true', default=False,
                    help='using SGD optim')
parser.add_argument('--cos', action='store_true', default=False,
                    help='using cosine scheduler')
parser.add_argument('--step', action='store_true', default=False,
                    help='using step scheduler')
parser.add_argument('--multi-step', action='store_true', default=False,
                    help='using multi-step scheduler')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--loss-weight', action='store_true', default=False,
                    help='weight the cross entropy loss')
parser.add_argument('--progressive-sampler', action='store_true', default=False,
                    help='use progressive sampler')
parser.add_argument('--sampling-interpolate', type=str, default='linear', 
                    help='which kind of interpolate method to use')
parser.add_argument('--accum-steps', type=int, default=1,
                    help='use batch gradient accumulation to train on larger batch')
parser.add_argument('--ddp', action='store_true', default=False,
                    help='use DDP training')
parser.add_argument('--world-size', type=int, default=1,
                    help='world size for ddp training')
parser.add_argument('--dist-url', type=str, default='tcp://localhost:10001',
                    help='communication url for ddp training')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use AMP training')
parser.add_argument('--scaler', action='store_true', default=False,
                    help='use gradient scaler training')
parser.add_argument('--contrastive', action='store_true', default=False,
                    help='use contrastive training')
parser.add_argument('--resume', type=str, default=None,
                    help='resume from given log dir')
parser.add_argument('--cur-ep', type=int, default=None,
                    help='staring ep of the resumed model')
parser.add_argument('--linear-probe', action='store_true', default=False,
                    help='linear probe training')
parser.add_argument('--dev', action='store_true', default=False,
                    help='train with debugging options')
parser.add_argument('--dropout-p', type=float, default=-1.0,
                    help='drop out layer probability')
parser.add_argument('--lars', action='store_true', default=False,
                    help='train with LARS optimizer')
parser.add_argument('--disc-transfer', action='store_true', default=False,
                    help='transfer learning with discriminative lr')
parser.add_argument('--lr-decay', type=float, default=2.8,
                    help='lr decay rate for disc transfer learning')
parser.add_argument('--extract-feat', action='store_true', default=False,
                    help='extract features with pretrained model')
parser.add_argument('--divide', action='store_true', default=False,
                    help='divide the input data to reduce GPU memory usage')


# Model settings
parser.add_argument('--model-type', type=str, default='resnet50', 
                    help='which kind of model to use')
parser.add_argument('--pretrain', action='store_true', default=False,
                    help='use pre-trained image')
parser.add_argument('--load-model', type=str, default=None,
                    help='load pre-trained model')
parser.add_argument('--cudnn', action='store_true', default=False,
                    help='use cudnn benchmark to speed up the training')
parser.add_argument('--load-moco', action='store_true', default=False,
                    help='load moco pre-trained model')
parser.add_argument('--load-best', action='store_true', default=False,
                    help='load best model')
parser.add_argument('--use-flash-attn', action='store_true', default=False,
                    help='use flash attention')
parser.add_argument('--bf16', action='store_true', default=False,
                    help='use bfloat16 for training')


# Log settings
parser.add_argument('--log', action='store_true', default=False,
                    help='log output and train images')
parser.add_argument('--log-interval', type=int, default=2000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=5,
                    help='how many epochs to wait before save the model')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For saving the current Model')
parser.add_argument('--save-rec', action='store_true', default=False,
                    help='For saving the current Model recurrently')
parser.add_argument('--save-best', action='store_true', default=False,
                    help='For saving the current best Model')
parser.add_argument('--log-grad-norm', action='store_true', default=False,
                    help='log model gradient norm')
parser.add_argument('--exp', type=str, default='task',
                    help='name of the experiments')


# Dataset settings
parser.add_argument('--load-from-raw', action='store_true', default=False,
                    help='load from raw dcm file')
parser.add_argument('--use-otsu', action='store_true', default=False,
                    help='use otsu segmented image')
parser.add_argument('--binary', action='store_true', default=False,
                    help='use binary label for classification')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='only use cancer & benign group')
parser.add_argument('--ignore-abnormal', action='store_true', default=False,
                    help='only use normal group')
parser.add_argument('--abnormal-only', action='store_true', default=False,
                    help='only use abnormal group')
parser.add_argument('--balance-data', action='store_true', default=False,
                    help='balance training data')
parser.add_argument('--binary-balance', action='store_true', default=False,
                    help='balance training data with binary label')
parser.add_argument('--subset', action='store_true', default=False,
                    help='train on a balanced subset')
parser.add_argument('--subset-ratio', type=float, default=1.0,
                    help='normal volume ratio on the subset')
parser.add_argument('--mid-slice', action='store_true', default=False,
                    help='train train with only the middle slice')
parser.add_argument('--target-H', type=int, default=512,
                    help='target input image height')
parser.add_argument('--target-W', type=int, default=512,
                    help='target input image width')
parser.add_argument('--num-slice', type=int, default=1,
                    help='#slices used as input from the same volume')
parser.add_argument('--sampling-gap', type=int, default=None,
                    help='gap between each sampled slices')
parser.add_argument('--fix-gap', action='store_true', default=False,
                    help='fixed gap between sampled slices to span the whole z-stack')
parser.add_argument('--patchify', action='store_true', default=False,
                    help='patchify input image into sub-patches')
parser.add_argument('--patch-size', type=int, default=128,
                    help='target input patch size')
parser.add_argument('--patch-cnt', type=int, default=1,
                    help='number of patches to feed into the input')
parser.add_argument('--patch-eps', type=float, default=1e-2,
                    help='ratio of mean value of valid patches')
parser.add_argument('--pick-mass-slice', action='store_true', default=False,
                    help='always pick the slice contain the mass')
parser.add_argument('--patch-lv', action='store_true', default=False,
                    help='use patch level dataset')
parser.add_argument('--positive-range', type=int, default=4,
                    help='range of slices that be considered as positive')
parser.add_argument('--mass-eps', type=float, default=0.2,
                    help='threshold of overlapping mass ratio')
parser.add_argument('--load-from-npz', action='store_true', default=False,
                    help='load images from compressed npz files')
parser.add_argument('--test-patch', action='store_true', default=False,
                    help='run test on the patch level images')

parser.add_argument('--uniform-norm', action='store_true', default=False,
                    help='normalize the data uniformly')


# Augmentation settings
parser.add_argument('--affine-prob', type=float, default=0.0,
                    help='probability to apply affine transforms')
parser.add_argument('--rotate-degree', type=int, default=30,
                    help='degree to randomly rotate the input')
parser.add_argument('--translate-ratio', type=float, default=0.2,
                    help='ratio to translate the image')
parser.add_argument('--moco-aug', action='store_true', default=False,
                    help='using MoCo augmentation during training')
parser.add_argument('--cj-strength', type=float, default=0.5,
                    help='strength to apply color jittering to the image')
parser.add_argument('--inter-slice', action='store_true', default=False,
                    help='treat inter slice as positive')
parser.add_argument('--inter-view', action='store_true', default=False,
                    help='treat inter-views as positive')
parser.add_argument('--inter-patient', action='store_true', default=False,
                    help='treat inter-patients as positive')
parser.add_argument('--inter-study', action='store_true', default=False,
                    help='treat inter-studies as positive')

# Test settings
parser.add_argument('--max-patch', action='store_true', default=False,
                    help='take max over patches')
parser.add_argument('--plot-logits', action='store_true', default=False,
                    help='plot prediction logits')
parser.add_argument('--train-data', action='store_true', default=False,
                    help='test on the training data')


# extra data generation args
parser.add_argument('--method', type=str, default='interpolate', 
                    choices=['interpolate', 'patching'],
                    help='indicate the method to use when generating fake images')

def get_opt():
    args = parser.parse_args()
    return args