# quant_train.py
# Adapted from github.com/zkkli/I-ViT
# Extended by Lionnus Kesting (lkesting@ethz.ch)

import argparse
import os
import time
import math
import logging
import uuid
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy
from timm.utils.clip_grad import dispatch_clip_grad

from models import *
from utils import *

import wandb

parser = argparse.ArgumentParser(description="I-ViT")

parser.add_argument("--model", default='deit_tiny',
                    choices=['deit_tiny', 'deit_small', 'deit_base', 
                             'swin_tiny', 'swin_small', 'swin_base'],
                    help="model")
parser.add_argument('--data', metavar='DIR', default='/dataset/imagenet/',
                    help='path to dataset')
parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET'],
                    type=str, help='Image Net dataset path')
parser.add_argument("--nb-classes", default=1000, type=int, help="number of classes")
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument("--device", default="cuda", type=str, help="device")
parser.add_argument("--print-freq", default=1000,
                    type=int, help="print frequency")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument('--output-dir', type=str, default='results/',
                    help='path to save log and quantized model')

parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--eff-batch-size', default=None, type=int,
                    help='Effective batch size using gradient accumulation. Must be a multiple of --batch-size')
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--pin-mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                    help='')
parser.set_defaults(pin_mem=True)

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

parser.add_argument('--model-ema', action='store_true')
parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 1e-6)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-7)')

parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                           "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

parser.add_argument('--best-acc1', type=float, default=0, help='best_acc1')

# Quantization control params
parser.add_argument(
    '--bitwidth',
    default=None,
    type=str,                    # accept "8"  OR  "8,8,8,8,8,8,8,8"
    help='Single bit-width or comma-separated list of 8 bit-widths for the respective locations: \
    patchembed, pos enc, attention out, softmax out, mlp out, norm 2 in, attention block out.'
)
parser.add_argument('--gelu',      default='ibert', 
                    help='GELU implementation to use')
parser.add_argument('--softmax',   default='ibert',
                    help='Softmax implementation to use')
parser.add_argument('--layernorm', default='ibert',
                    help='LayerNorm implementation to use')
# Alternative to set all three at once
parser.add_argument(
    '--layer_type',
    choices=['ivit','ibert'],
    default=None,
    help='If set, use this implementation for GELU, Softmax, and LayerNorm. Overrides --gelu, --softmax, and --layernorm.'
)

# Calibration parameters
parser.add_argument('--calibration-batches', type=int, default=100,
                    help='Number of batches to use for calibration (default: 5)')
parser.add_argument('--calibration-epochs', type=int, default=0,
                    help='If 0 no calibration, else the number of epochs after which the model is unfixed.')

# Weights & Biases logging
parser.add_argument('--wandb-project', type=str, default='i-vit',
                    help='Weights & Biases project name')
parser.add_argument('--wandb-entity', type=str, default=None,
                    help='Weights & Biases entity (username or team)')
parser.add_argument('--wandb-run-name', type=str, default=None,
                    help='Custom run name for Weights & Biases')
parser.add_argument('--no-wandb', action='store_true',
                    help='Disable Weights & Biases logging')

def str2model(name):
    d = {'deit_tiny': deit_tiny_patch16_224,
         'deit_small': deit_small_patch16_224,
         'deit_base': deit_base_patch16_224,
        #  'swin_tiny': swin_tiny_patch4_window7_224,
        #  'swin_small': swin_small_patch4_window7_224,
        #  'swin_base': swin_base_patch4_window7_224,
         }
    print('Model: %s' % d[name].__name__)
    return d[name]

def calibrate_model(model, train_loader, num_batches=100, device='cuda'):
    """
    Calibrate quantization parameters by running inference on training data.
    This helps establish proper scaling factors before training begins.
    """
    model.eval()
    logging.info(f"Starting calibration with {num_batches} batches...")
    
    # Collect initial statistics
    initial_stats = {}
    for name, module in model.named_modules():
        if hasattr(module, 'act_scaling_factor'):
            initial_stats[name] = {
                'scale': module.act_scaling_factor.clone()
            }
            # QuantAct has x_min/x_max, QuantMatMul doesn't
            if hasattr(module, 'x_min'):
                initial_stats[name]['x_min'] = module.x_min.clone()
                initial_stats[name]['x_max'] = module.x_max.clone()
    
    with torch.no_grad():
        for i, (data, _) in enumerate(train_loader):
            if i >= num_batches:
                break
            data = data.to(device, non_blocking=True)
            _ = model(data)
            
            if (i + 1) % 20 == 0:
                logging.info(f"Calibration progress: {i + 1}/{num_batches} batches")
    
    # Log the changes in scaling factors
    logging.info("Calibration completed. Scaling factor changes:")
    for name, module in model.named_modules():
        if hasattr(module, 'act_scaling_factor') and name in initial_stats:
            old_scale = initial_stats[name]['scale'].item()
            new_scale = module.act_scaling_factor.item()
            if abs(old_scale - new_scale) > 1e-6:
                logging.info(f"{name}: scale {old_scale:.6f} -> {new_scale:.6f}")
    
    # Fix ranges for first few steps/epochs
    for module in model.modules():
        if hasattr(module, 'fix'):
            module.fix()
    
    # Unfix after few epochs in the main training loop
    return model

def main():
    args = parser.parse_args()

    # generate a unique identifier for this run
    run_id = uuid.uuid4().hex
    args.run_id = run_id

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    import warnings
    warnings.filterwarnings('ignore')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # log to a unique file per run
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename=os.path.join(args.output_dir, f'log_{run_id}.log'))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f'Run ID: {run_id}')
    logging.info(args)

    device = torch.device(args.device)

    # Dataset
    train_loader, val_loader = dataloader(args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # Unpack model config for bitwidths
    # Default bitwidths
    default_bitwidths = [8, 8, 8, 8, 8, 8, 8, 8]  # patch_embed, pos_encoding, block_input, attention_out, softmax_out, mlp_out, norm2_in, att_block_out
    
    if args.bitwidth is not None:
        # turn "8,8,8,8,8,8,8,8" into [8,8,8,8,8,8,8,8]
        bw_list = [int(x) for x in args.bitwidth.replace(',', ' ').split()]
        if len(bw_list) == 1:
            bw_list *= 8
        if len(bw_list) != 8:
            raise ValueError('--bitwidth must be 1 or 8 values')
        args.quant_bitwidths = bw_list
    else:
        args.quant_bitwidths = default_bitwidths
        
    (
        patch_embed_bw,
        pos_encoding_bw,
        block_input_bw,
        attention_out_bw,
        softmax_bw,
        mlp_out_bw,
        norm2_in_bw,
        att_block_out_bw
    ) = args.quant_bitwidths
    
    # if the helper is set, override all three
    if args.layer_type is not None:
        args.gelu      = args.layer_type
        args.softmax   = args.layer_type
        args.layernorm = args.layer_type

    # Model
    model = str2model(args.model)(
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        patch_embed_bw=patch_embed_bw,
        pos_encoding_bw=pos_encoding_bw,
        block_input_bw=block_input_bw,
        attention_out_bw=attention_out_bw,
        softmax_bw=softmax_bw,
        mlp_out_bw=mlp_out_bw,
        norm2_in_bw=norm2_in_bw,
        att_block_out_bw=att_block_out_bw,
        gelu_type=args.gelu,
        softmax_type=args.softmax,
        layernorm_type=args.layernorm
    )

    model.to(device)
    
    # Perform calibration before setting up training
    if args.calibration_epochs > 0:
        model = calibrate_model(model, train_loader, 
                              num_batches=args.calibration_batches, 
                              device=device)
        # Weights & Biases init
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args)
        )
        # Explicitly store bit-widths and layer types for easy filtering
        wandb.config.update({
            'patch_embed_bw': patch_embed_bw,
            'pos_encoding_bw': pos_encoding_bw,
            'block_input_bw': block_input_bw,
            'attention_out_bw': attention_out_bw,
            'softmax_bw': softmax_bw,
            'mlp_out_bw': mlp_out_bw,
            'norm2_in_bw': norm2_in_bw,
            'att_block_out_bw': att_block_out_bw,
            'gelu_type': args.layer_type or args.gelu,
            'softmax_type': args.layer_type or args.softmax,
            'layernorm_type': args.layer_type or args.layernorm,
            'calibration_batches': args.calibration_batches,
            'effective_batch_size': args.eff_batch_size or args.batch_size
        }, allow_val_change=True)
        wandb.watch(model, log='gradients', log_freq=100)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        
    args.min_lr = args.lr / 15
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion_v = nn.CrossEntropyLoss()

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)

    print(f"Start training for {args.epochs} epochs")
    best_epoch = 0
    
    # Initialize timing variables
    epoch_times = []
    
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Unfix quantization parameters after first epoch
        if epoch == args.calibration_epochs:
            logging.info(f"Unfixing quantization parameters at epoch {epoch}")
            for module in model.modules():
                if hasattr(module, 'unfix'):
                    module.unfix()
        
        # train for one epoch
        train_loss = train(args, train_loader, model, criterion, optimizer, epoch,
              loss_scaler, args.clip_grad, model_ema, mixup_fn, device)
        lr_scheduler.step(epoch)

        # if args.output_dir:  # this is for resume training
        #     checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
        #     torch.save({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'model_ema': get_state_dict(model_ema),
        #         'scaler': loss_scaler.state_dict(),
        #         'args': args,
        #     }, checkpoint_path)

        acc1 = validate(args, val_loader, model, criterion_v, device)
        
        # Calculate epoch time and ETA
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times[-10:])  # Average of last 10 epochs
        remaining_epochs = args.epochs - epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # Log timing info
        logging.info(f'Epoch {epoch} completed in {timedelta(seconds=int(epoch_time))} - ETA: {eta_str}')

        #  WandB logging per epoch
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss_epoch': train_loss,
                'val_acc1': acc1,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'eta_seconds': eta_seconds
            })

        # remember best acc@1 and save checkpoint
        is_best = acc1 > args.best_acc1
        args.best_acc1 = max(acc1, args.best_acc1)
        if is_best:
            # record the best epoch
            best_epoch = epoch
            # save checkpoint with unique run_id
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_{run_id}.pth.tar')
            
            # Create model config to save with checkpoint
            model_config = {
                'model_name': args.model,
                'num_classes': args.nb_classes,
                'drop_rate': args.drop,
                'drop_path_rate': args.drop_path,
                'quant_bitwidths': args.quant_bitwidths,
                'patch_embed_bw': patch_embed_bw,
                'pos_encoding_bw': pos_encoding_bw,
                'block_input_bw': block_input_bw,
                'attention_out_bw': attention_out_bw,
                'softmax_bw': softmax_bw,
                'mlp_out_bw': mlp_out_bw,
                'norm2_in_bw': norm2_in_bw,
                'att_block_out_bw': att_block_out_bw,
                'gelu_type': args.gelu,
                'softmax_type': args.softmax,
                'layernorm_type': args.layernorm
            }
            
            # Save checkpoint with model config
            torch.save({
                'model': model.state_dict(),
                'model_config': model_config,
                'epoch': epoch,
                'best_acc1': args.best_acc1,
                'args': vars(args)  # Save all args for reference
            }, ckpt_path)
            
        logging.info(f'Acc at epoch {epoch}: {acc1}')
        logging.info(f'Best acc at epoch {best_epoch}: {args.best_acc1}')


def train(args, train_loader, model, criterion, optimizer, epoch, loss_scaler, max_norm, model_ema, mixup_fn, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # Calculate gradient accumulation steps
    accumulation_steps = 1
    if args.eff_batch_size is not None:
        if args.eff_batch_size % args.batch_size != 0:
            raise ValueError(f"eff_batch_size ({args.eff_batch_size}) must be a multiple of batch_size ({args.batch_size})")
        accumulation_steps = args.eff_batch_size // args.batch_size
        logging.info(f"Using gradient accumulation: {accumulation_steps} steps for effective batch size {args.eff_batch_size}")

    # switch to train mode
    model.train()
    unfreeze_model(model)

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        output = model(data)
        loss = criterion(output, target)
        # measure accuracy and record loss (use unscaled loss for logging)
        losses.update(loss.item(), data.size(0))
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass - accumulate gradients
        loss_scaler._scaler.scale(loss).backward()

        # Only update weights after accumulating gradients
        if (i + 1) % accumulation_steps == 0:
            # Unscale gradients and clip if needed
            if max_norm is not None:
                loss_scaler._scaler.unscale_(optimizer)
                dispatch_clip_grad(model.parameters(), max_norm, mode='norm')
            
            # Step optimizer and update scaler
            loss_scaler._scaler.step(optimizer)
            loss_scaler._scaler.update()
            
            # Zero gradients for next accumulation
            optimizer.zero_grad()
            
            # Update model EMA
            if model_ema is not None:
                model_ema.update(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            # WandB logging per iteration
            if not args.no_wandb:
                wandb.log({
                    'iter': i + epoch * len(train_loader),
                    'train_loss': losses.val,
                    'epoch': epoch,
                    'effective_batch_size': args.eff_batch_size or args.batch_size
                })
    
    # Handle any remaining gradients at the end of epoch
    if len(train_loader) % accumulation_steps != 0:
        if max_norm is not None:
            loss_scaler._scaler.unscale_(optimizer)
            dispatch_clip_grad(model.parameters(), max_norm, mode='norm')
        loss_scaler._scaler.step(optimizer)
        loss_scaler._scaler.update()
        optimizer.zero_grad()
                
    # return avg loss for epoch 
    return losses.avg

def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    freeze_model(model)

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



if __name__ == "__main__":
    main()