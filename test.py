from imagenet import Imagenet_Dataset
import os
import sys
import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

testset = Imagenet_Dataset(root_dir = '/nvme/data/imagenet/original_val/val/',
                                    label_file='/media/disk1/code/ViT-pytorch/utils/val.txt',
                                    transform = transform_test)
test_sampler = SequentialSampler(testset)
test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=128,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None


def test_engine(weight_path, test_loader):
    # define the network
    net = se_resnet50(num_classes=1000)
    net.load_state_dict(torch.load(weight_path), strict=False)
    net = net.cuda()
    net.eval()

    # define the statistic params
    correct_1 = 0.0
    correct_5 = 0.0
    with torch.no_grad():
        for step, (images, labels) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(step + 1, len(test_loader)))
            images = images.cuda()
            labels = labels.cuda()

            test_outputs = net(images)
            _, pred = test_outputs.topk(5, 1, largest=True, sorted=True)
            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()

            # compute Top-5 Accuracy
            correct_5 += correct[:, :5].sum()

            # compute Top-1 Accuracy
            correct_1 += correct[:, :1].sum()

        print()
        print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
        print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))

test_engine('./seresnet50-weight.pkl', test_loader)