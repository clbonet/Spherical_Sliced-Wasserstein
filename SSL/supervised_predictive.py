"""
Modified version of linear_eval.py to perform predictive supervised learning.
"""
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F
import torchvision


from dataparser import dataparser, from_args, Field
from encoder import ResNet, SmallAlexNet
from util import AverageMeter


@dataparser
class Options:
    "Supervised predictive training"

    results_folder: Path = Path("./results_supervised")

    # Meta settings
    dataset: str = Field(default="cifar10", choices=["stl10", "cifar10"])
    data_folder: Path = Path("./data")

    encoder: str = Field(choices=["alexnet", "resnet"])
    feat_dim: int = 3
    layer_index: int = -2

    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    lr_decay_rate: float = 0.2
    lr_decay_epochs: str = "60,80"

    num_workers: int = 6
    log_interval: int = 40
    gpu: int = 0
    seed: int = 0


def prepare_loader(opt: Options):
    get_train_transform = lambda mean, std, crop_size, crop_scale: torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(crop_size, scale=crop_scale),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std,),
        ]
    )
    get_val_transform = lambda mean, std, resize, crop_size: torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std,),
        ]
    )
    if opt.dataset == "stl10":
        train_transform = get_train_transform(
            mean=(0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            std=(0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
            crop_size=64,
            crop_scale=(0.08, 1),
        )
        train_dataset = torchvision.datasets.STL10(
            opt.data_folder, "train", download=True, transform=train_transform,
        )
        val_transform = get_val_transform(
            mean=(0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            std=(0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
            resize=70,
            crop_size=64,
        )
        val_dataset = torchvision.datasets.STL10(
            opt.data_folder, "test", transform=val_transform,
        )
    elif opt.dataset == "cifar10":
        train_transform = get_train_transform(
            mean=(0.4915, 0.4822, 0.4466),
            std=(0.2470, 0.2435, 0.2616),
            crop_size=32,
            crop_scale=(0.2, 1.0),
        )
        train_dataset = torchvision.datasets.CIFAR10(
            opt.data_folder, train=True, download=True, transform=train_transform,
        )
        val_transform = get_val_transform(
            mean=(0.4915, 0.4822, 0.4466),
            std=(0.2470, 0.2435, 0.2616),
            crop_size=32,
            resize=32,
        )
        val_dataset = torchvision.datasets.CIFAR10(
            opt.data_folder, train=False, transform=val_transform,
        )
    else:
        raise NotImplementedError(f"dataset {opt.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    return train_loader, val_loader


def validate(opt, encoder, classifier, val_loader):
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            pred = classifier(
                encoder(images.to(opt.gpu), layer_index=opt.layer_index).flatten(1)
            ).argmax(dim=1)
            correct += (pred.cpu() == labels).sum().item()
    return correct / len(val_loader.dataset)


def linear_eval(opt: Options):
    torch.manual_seed(opt.seed)
    torch.cuda.set_device(opt.gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if opt.encoder == "alexnet":
        encoder = SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpu)
    else:
        encoder = ResNet(feat_dim=opt.feat_dim).to(opt.gpu)
    encoder.train()
    train_loader, val_loader = prepare_loader(opt)

    stats_file = open(
        opt.results_folder / "linear_eval.txt", "a", buffering=1
    )
    def print_stats(msg):
        print(msg)
        print(msg, file=stats_file)

    with torch.no_grad():
        sample, _ = train_loader.dataset[0]
        eval_numel = encoder(
            sample.unsqueeze(0).to(opt.gpu), layer_index=opt.layer_index
        ).numel()

    classifier = nn.Linear(eval_numel, 10).to(opt.gpu)

    optim = torch.optim.Adam(list(classifier.parameters()) + list(encoder.parameters()),
        lr=opt.lr, betas=(0.5, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, opt.epochs)

    loss_meter = AverageMeter("loss")
    it_time_meter = AverageMeter("iter_time")
    best_val_acc = 0.0
    for epoch in range(opt.epochs):
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, (images, labels) in enumerate(train_loader):
            optim.zero_grad()
            feats = encoder(
                images.to(opt.gpu), layer_index=opt.layer_index,
            ).flatten(1)
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels.to(opt.gpu))
            loss_meter.update(loss, images.shape[0])
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                print_stats(
                    f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(train_loader)}\t{loss_meter}\t{it_time_meter}"
                )
            t0 = time.time()
        scheduler.step()

        encoder.eval()
        val_acc = validate(opt, encoder, classifier, val_loader)
        encoder.train()

        best_val_acc = max(val_acc, best_val_acc)
        print_stats(f"Epoch {epoch}/{opt.epochs}\tval_acc {val_acc*100:.4g}%")

    encoder.eval()
    val_acc_last = validate(opt, encoder, classifier, val_loader)

    torch.save(encoder.state_dict(), opt.results_folder / "encoder.pth")
    print_stats(f"top1 = {val_acc_last}")
    with open("all_results.txt", "a") as f:
        identifier = "supervised_predictive"
        print(f"{identifier},{val_acc_last}", file=f)


def main():
    opt = from_args(Options)
    linear_eval(opt)


if __name__ == "__main__":
    main()
