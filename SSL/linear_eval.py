from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F
import torchvision


from dataparser import dataparser, from_args, Field
from encoder import ResNet
from util import AverageMeter


@dataparser
class Options:
    "Linear evaluation"

    encoder_checkpoint: Path = Field(positional=True)

    # Meta settings
    data_folder: Path = Path("./data")

    feat_dim: int = 128
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

    encoder = ResNet(feat_dim=opt.feat_dim).to(opt.gpu)
    encoder.eval()
    train_loader, val_loader = prepare_loader(opt)

    stats_file = open(
        opt.encoder_checkpoint.parent / "linear_eval.txt", "a", buffering=1
    )
    def print_stats(msg):
        print(msg)
        print(msg, file=stats_file)

    encoder.load_state_dict(
        torch.load(opt.encoder_checkpoint, map_location=torch.device(opt.gpu))
    )
    print_stats(f"Loaded checkpoint from {opt.encoder_checkpoint}")

    with torch.no_grad():
        sample, _ = train_loader.dataset[0]
        eval_numel = encoder(
            sample.unsqueeze(0).to(opt.gpu), layer_index=opt.layer_index
        ).numel()

    classifier = nn.Linear(eval_numel, 10).to(opt.gpu)

    optim = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, gamma=opt.lr_decay_rate, milestones=opt.lr_decay_epochs
    )
    loss_meter = AverageMeter("loss")
    it_time_meter = AverageMeter("iter_time")
    best_val_acc = 0.0
    for epoch in range(opt.epochs):
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, (images, labels) in enumerate(train_loader):
            optim.zero_grad()
            with torch.no_grad():
                feats = encoder(
                    images.to(opt.gpu), layer_index=opt.layer_index
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
        val_acc = validate(opt, encoder, classifier, val_loader)
        best_val_acc = max(val_acc, best_val_acc)
        print_stats(f"Epoch {epoch}/{opt.epochs}\tval_acc {val_acc*100:.4g}%")
    print_stats(f"Best top1 = {best_val_acc}")
    with open("all_results.txt", "a") as f:
        identifier = opt.encoder_checkpoint.parent.name
        print(f"{identifier},{best_val_acc}", file=f)
    return best_val_acc


def main():
    opt = from_args(Options)
    linear_eval(opt)


if __name__ == "__main__":
    main()
