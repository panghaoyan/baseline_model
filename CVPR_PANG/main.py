import timm.scheduler.step_lr
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import ViT
import os
from dataloader import *
from Sewer_metrics import *
from CODEBRIM_metrics import *
from argparse import ArgumentParser
# from visdom import Visdom
import time
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
# import matplotlib
from swin_transformer_models.swin_transformer import SwinTransformer
from swin_transformer_models.swin_mlp import SwinMLP
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True



def show_example(img):
    image = img
    image = make_grid(image, padding=0)
    npimg = image.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()


def choose_model(name):
    if name == 'swin':

        model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=args.num_classes,
                                embed_dim=96,
                                depths=[2, 2, 10, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=False,
                                drop_rate=0,
                                drop_path_rate=0.3,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
    elif name == 'swin_mlp':
        model = SwinMLP(img_size=256,
                        patch_size=4,
                        in_chans=3,
                        num_classes=args.num_classes,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[ 4, 8, 16, 32 ],
                        window_size=8,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=False,
                        drop_rate=0,
                        drop_path_rate=0.2,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False)
    else:
        raise NotImplementedError("Unkown model")

    return model


def choose_dataloader(name):
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])
    Sewer_train_set = MultiLabelDataset(annRoot="/home/ouc/水下缺陷检测数据集/The_Sewer-ML_Dataset",
                                        imgRoot="/home/ouc/水下缺陷检测数据集/The_Sewer-ML_Dataset/data_all",
                                        split="Train", transform=train_transform)
    # Sewer_val_set = MultiLabelDataset(annRoot="/home/ouc/水下缺陷检测数据集/The_Sewer-ML_Dataset",
    #                                   imgRoot="/home/ouc/水下缺陷检测数据集/The_Sewer-ML_Dataset/data_all",
    #                                   split="Val", transform=val_transform)

    # Sewer_test_set = MultiLabelDataset(annRoot="/home/ouc/水下缺陷检测数据集/The_Sewer-ML_Dataset",
    #                                     imgRoot="/home/ouc/水下缺陷检测数据集/The_Sewer-ML_Dataset/data_all",
    #                                     split="Test", transform=transform)
    CODEBRIM_datasets = CODEBRIM(is_gpu=torch.cuda.is_available(), path=args.dataset_path
                                 , img_size=args.img_size, batch_size=args.batch_size, workers=args.workers)

    if name == 'sewer':
        train_loader = torch.utils.data.DataLoader(Sewer_train_set, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers)
        # val_loader = torch.utils.data.DataLoader(Sewer_val_set, batch_size=args.batch_size,
        #                                          shuffle=False, num_workers=args.workers)
        # test_loader = torch.utils.data.DataLoader(Sewer_test_set, batch_size=batch_size,
        #                                                  shuffle=True, num_workers=0)
    else:
        train_loader, val_loader, test_loader = CODEBRIM_datasets.get_dataset_loader(
            batch_size=args.batch_size,
            workers=args.batch_size,
            is_gpu=torch.cuda.is_available()

        )

    return train_loader


def choose_evaluation(name, outputs, target, LabelWeights):
    if name == 'sewer':
        evaluation = Sewer_evaluation(outputs, target, LabelWeights)
    else:
        evaluation = CODEBRIM_evaluation(outputs, target, LabelWeights)
    return evaluation


def train():
    print("current model: {0}".format(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MultiLabelSoftMarginLoss()
    # loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

    model = choose_model(args.model).to(device)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1, eta_min=5e-6)
    train_loader = choose_dataloader(args.dataset)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0005, eps=1e-8, betas=(0.9, 0.999))
    lr_scheduler = timm.scheduler.step_lr.StepLRScheduler(optimizer, decay_rate=0.1, decay_t=len(train_loader))
    model.train()
    count = 0
    num_steps = len(train_loader)
    for epoch in range(args.epoch):
        total_loss = 0
        num = 0
        for img, target in train_loader:
            # show_example(img)
            img = img.to(device)
            target = target.to(device)
            # labels = target
            # labels = train_loader.dataset.labels_batch
            # print(target)
            # print(labels)
            optimizer.zero_grad()
            outputs = model(img)  # (32, 6)

            loss = loss_fn(outputs, target)
            main = choose_evaluation(args.dataset, outputs, target, LabelWeights)
            loss.backward()

            optimizer.step()
            lr_scheduler.step(epoch=epoch)

            tmp_loss = loss.item()
            total_loss += loss.item()
            num += args.batch_size
            print('Epoch: [{0}/{1}] Batch: [{2}/{3}] loss: [{4}]\t'
                  .format(epoch, args.epoch, num, len(choose_dataloader(args.dataset).dataset), tmp_loss))
            print('OP: {OP:.8f}\t'
                  'OR: {OR:.8f}\t'
                  'OF1: {OF1:.8f}\t'
                  'CP: {CP:.8f}\t'
                  'CR: {CR:.8f}\t'
                  'CF1: {CF1:.8f}\t'
                  'mAP {mAP:.8f}'
                  .format(OP=main['OP'], OR=main['OR'], OF1=main['OF1'], CP=main['CP'], CR=main['CR'], CF1=main['CF1'],
                          mAP=main['mAP']))
            # visdom.line([loss.item()], [count], win='train_loss', update='append')
            count += 1
            train_loader.dataset.labels_batch = []
        if epoch % 10 == 0:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            pth_name = str(args.model) + '_' + 'epoch' + str(epoch) + '_' + 'loss' + str(total_loss) + '.pth'
            path = args.ckpt_path + '/' + pth_name
            torch.save(state, path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=r'/home/ouc/水下缺陷检测数据集/CODEBRIM/classification_dataset_balanced')
    parser.add_argument('--batch_size', type=int, default=32, help="Size of the batch per GPU")
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16,
                        help="size of patch in patch_embbeding of transformer, image_size must be divisible by patch_size,")
    parser.add_argument('--num_classes', type=int, default=6, help="Sewer 18, if CODEBRIM, then num_class should be 6")
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--dim', type=int, default=512,
                        help="Last dimension of output tensor after linear transformation nn.Linear(..., dim)")
    parser.add_argument('--depth', type=int, default=6, help="Number of Transformer blocks")
    parser.add_argument('--heads', type=int, default=16, help="Number of heads in Multi-head Attention layer")
    parser.add_argument('--mlp_dim', type=int, default=2048, help="Dimension of the MLP (FeedForward) layer")
    parser.add_argument('--dropout', type=int, default=0.1, help="Dropout rate")
    parser.add_argument('--emb_dropout', type=int, default=0.1, help="Embedding dropout rate")
    parser.add_argument('--dataset', type=str, default='codebrim', help="choose model, 'sewer" or 'codebrim')
    parser.add_argument('--model', type=str, default='swin', help="choose model, 'swin" or 'swin_mlp')
    parser.add_argument('--ckpt_path', type=str, default=r'/home/ouc/CVPR_PANG/checkpoint',
                        help='the path to save the latest checkpoint')
    args = parser.parse_args()

    # visdom = Visdom()
    # visdom.line([0.0], [0], win='train_loss', opts=dict(title='train_loss'))
    setup_seed(0)

    train()
