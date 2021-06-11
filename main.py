import os
import torch
import time
import argparse
from torchvision import transforms
from utils import AverageMeter, accuracy, _pil_interp
from logger import create_logger
from imagenet import Imagenet_Dataset
from config import get_config
from torch.utils.data import DataLoader,  SequentialSampler

def parse_option():
    parser = argparse.ArgumentParser('Simple Imagenet Testing Pipeline', add_help=False)

    # easy config modification
    parser.add_argument('--dataset', type=str, default='imagenet', help='just a part of testing tag')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default='dataset', help='path to dataset')
    parser.add_argument('--gpu', type=str, default='0', help="gpu choose, e.g. '0,1,2, ...'")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def build_dataset(config, transform):
    test_dataset = Imagenet_Dataset(root_dir=config.TEST.DATA_PATH,
                               label_file=config.TEST.LABEL_PATH,
                               transform=transform)
    return test_dataset

def build_transform(config):
    resize_im = config.TEST.IMG_SIZE > 224
    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.TEST.INTERPOLATION))
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.TEST.IMG_SIZE, config.TEST.IMG_SIZE),
                                  interpolation=_pil_interp(config.TEST.INTERPOLATION))
            )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(config.TEST.MEAN, config.TEST.STD))
    return transforms.Compose(t)

def build_loader(config, dataset):
    sampler_val = SequentialSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(
        dataset, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return data_loader_val

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

def main(config, model):
    dataset_val = build_dataset(config)
    data_loader_val = build_loader(config, dataset_val)
    model.cuda()
    logger.info(str(model))
    acc1, acc5, loss = validate(config, data_loader_val, model)

if __name__ == "__main__":
    _, config = parse_option()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.TAG}")
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    # use pretrained resnext101 as an example
    from torchvision.models import resnext101_32x8d
    test_model = resnext101_32x8d(pretrained=True)
    main(config, test_model)
