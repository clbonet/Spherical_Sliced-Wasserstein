from typing import List
from pathlib import Path
import json
import time
from datetime import datetime

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import STL10, CIFAR10

from dataparser import dataparser, Field, from_args
from util import TwoAugUnsupervisedDataset, AverageMeter, strfdelta
from encoder import ResNet
from sw_sphere import sliced_wasserstein_sphere_uniform


@dataparser
class Options:
    "Pre-training script"

    # Storage settings
    data_folder: Path = Path("./data/")
    result_folder: Path = Path("./results/")

    # Meta settings
    method: str = Field(choices=["ssw", "simclr", "hypersphere"],
                        help="Choose one of ssw, simclr, hypersphere")

    # Methods hyper-parameters
    unif_w: float = 1.0
    align_w: float = 1.0
    align_alpha: float = 2.0
    unif_t: float = 2.0
    num_projections: int = 256

    # Run parameters
    lr: float = 0.05
    cosine_schedule: bool = Field(action="store_true")
    epochs: int = 200
    batch_size: int = 768
    momentum: float = 0.9
    weight_decay: float = 1e-3
    feat_dim: int = 128

    num_workers: int = 6
    log_interval: int = 40
    gpus: List[int] = Field(nargs="*", default=[0])
    identifier: str = Field(default=None)
    seed: int = 0


def prepare_loader(opt: Options) -> DataLoader:
    get_transform = lambda mean, std, crop_size, crop_scale: torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(crop_size, scale=crop_scale),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std,),
        ]
    )

    transform = get_transform(
        mean=(0.4915, 0.4822, 0.4466),
        std=(0.2470, 0.2435, 0.2616),
        crop_size=32,
        crop_scale=(0.2, 1.0),
    )
    dataset = CIFAR10(opt.data_folder, train=True, download=True)
    dataset = TwoAugUnsupervisedDataset(dataset, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=True,
    )


def align_loss(x: Tensor, y: Tensor, alpha: float) -> Tensor:
    return (x - y).norm(dim=1).pow(alpha).mean()


def uniform_loss(x: Tensor, t: float) -> Tensor:
    sq_dist = torch.pdist(x, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().log()


def pretrain(opt: Options):
    opt.result_folder.mkdir(exist_ok=True)
    opt.data_folder.mkdir(exist_ok=True)

    torch.manual_seed(opt.seed)

    identifier = opt.identifier if opt.identifier is not None else str(int(time.time()))
    save_folder = opt.result_folder / identifier
    save_folder.mkdir(exist_ok=True)

    start_time = datetime.now()
    stats_file = open(save_folder / "stats.txt", "a", buffering=1)
    def print_stats(msg):
        print(msg)
        print(msg, file=stats_file)

    start_time_fmt = start_time.strftime("%d/%m/%y %H:%M")
    print_stats(f"Starting training at {start_time_fmt}")

    print_stats(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in opt.__dict__.items()}
        ),
    )

    if opt.method != "simclr":
        method_name = f"loss_uniform(t={opt.unif_t:g})" if opt.method == "hypersphere" else "(ssw(x) + ssw(y)) / 2"
        print_stats(
            f"Optimize: {opt.align_w:g} * loss_align(alpha={opt.align_alpha:g}) + {opt.unif_w:g} * {method_name}"
        )
    elif opt.method == "simclr":
        print_stats("Optimize: simclr(tau=0.2)")

    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    encoder = ResNet(feat_dim=opt.feat_dim).to(opt.gpus[0])
    encoder = nn.DataParallel(encoder, opt.gpus)

    optim = torch.optim.SGD(
        encoder.parameters(),
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, opt.epochs)

    loader = prepare_loader(opt)

    def align_uniform_loss(x, y):
        align_loss_val = align_loss(x, y, alpha=opt.align_alpha)
        unif_loss_val = (
            uniform_loss(x, t=opt.unif_t) + uniform_loss(y, t=opt.unif_t)
        ) / 2
        return align_loss_val, unif_loss_val

    def ssw_loss(x, y):
        align_loss_val = align_loss(x, y, alpha=opt.align_alpha)
        unif_loss_val = (
            sliced_wasserstein_sphere_uniform(x, opt.num_projections)
            + sliced_wasserstein_sphere_uniform(y, opt.num_projections)
        ) / 2
        return align_loss_val, unif_loss_val

    def simclr_loss(x, y):
        b = x.size(0)
        z = torch.cat((x, y))
        sims = (z @ z.T) / 0.2  # tau
        sims.diagonal().sub_(1e9)
        labels = torch.cat(
            (torch.arange(b, 2 * b, device=z.device), torch.arange(b, device=z.device))
        )
        return F.cross_entropy(sims, labels), torch.tensor(0.0)

    loss_func = {
        "hypersphere": align_uniform_loss,
        "ssw": ssw_loss,
        "simclr": simclr_loss,
    }[opt.method]

    align_meter = AverageMeter("align_loss")
    unif_meter = AverageMeter("uniform_loss")
    loss_meter = AverageMeter("total_loss")
    it_time_meter = AverageMeter("iter_time")

    for epoch in range(opt.epochs):
        align_meter.reset()
        unif_meter.reset()
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()

        for ii, (im_x, im_y) in enumerate(loader):
            optim.zero_grad()
            x = encoder(im_x.to(opt.gpus[0]))
            y = encoder(im_y.to(opt.gpus[0]))

            align_loss_val, unif_loss_val = loss_func(x, y)
            loss = align_loss_val * opt.align_w + unif_loss_val * opt.unif_w

            align_meter.update(align_loss_val)
            unif_meter.update(unif_loss_val)
            loss_meter.update(loss, x.shape[0])
            loss.backward()

            optim.step()

            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                print_stats(
                    f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t"
                    + f"{align_meter}\t{unif_meter}\t{loss_meter}\t{it_time_meter}"
                )
            t0 = time.time()

        stats = dict(
            epoch=epoch,
            time=it_time_meter.sum,
            loss=float(loss_meter.sum),
            unif_loss=float(unif_meter.sum),
            align_loss=float(align_meter.sum),
        )
        print_stats(json.dumps(stats))
        scheduler.step()

        if epoch % 100 == 0:
            checkpoint_file = save_folder / f"encoder_{epoch}.pth"
            torch.save(encoder.module.state_dict(), checkpoint_file)

    checkpoint_file = save_folder / "encoder.pth"
    torch.save(encoder.module.state_dict(), checkpoint_file)
    print_stats(f"Saved to {checkpoint_file}")
    end_time = datetime.now()
    run_time = strfdelta(end_time - start_time, "{hours}:{minutes}:{seconds}")
    end_time = end_time.strftime("%d/%m/%y %H:%M")
    print_stats(f"Training done at {end_time} in {run_time}")


def main():
    opt = from_args(Options)
    pretrain(opt)


if __name__ == "__main__":
    main()
