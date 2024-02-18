import torch
from torch.utils import benchmark

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sw_sphere import sliced_wasserstein_sphere_uniform
from main import uniform_loss


def do_benchmarks():
    D = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_threads = 1
    results = []
    min_run_time = 1

    print(f"D = {D}")
    print(f"device = {device}, num_threads = {num_threads}")

    setup_forward = """
    x = torch.randn(N, D, device=device)
    y = torch.randn(N, D, device=device)
    """

    setup_backward = """
    x = torch.randn(N, D, device=device, requires_grad=True)
    y = torch.randn(N, D, device=device, requires_grad=True)
    """

    for N in range(256, 1025, 256):
        for num_projections in [2**i for i in range(5, 11)]:
            t0 = benchmark.Timer(
                stmt="loss(x, num_projections) + loss(y, num_projections)",
                setup=setup_forward,
                globals={
                    "device": device,
                    "N": N,
                    "D": D,
                    "loss": sliced_wasserstein_sphere_uniform,
                    "num_projections": num_projections,
                },
                num_threads=num_threads,
                label="Forward",
                sub_label=str(N),
                description=f"SSW L={num_projections}",
            ).blocked_autorange(min_run_time=min_run_time)

            results.append(t0)

            t0 = benchmark.Timer(
                stmt="(loss(x, num_projections) + loss(y, num_projections)).backward()",
                setup=setup_backward,
                globals={
                    "device": device,
                    "N": N,
                    "D": D,
                    "loss": sliced_wasserstein_sphere_uniform,
                    "num_projections": num_projections,
                },
                num_threads=num_threads,
                label="Forward + Backward",
                sub_label=str(N),
                description=f"SSW L={num_projections}",
            ).blocked_autorange(min_run_time=min_run_time)

            results.append(t0)

        t1 = benchmark.Timer(
            stmt="loss(x, 2.) + loss(y, 2.)",
            setup=setup_forward,
            globals={"device": device, "N": N, "D": D, "loss": uniform_loss},
            num_threads=num_threads,
            label="Forward",
            sub_label=str(N),
            description="Uniform",
        ).blocked_autorange(min_run_time=min_run_time)

        results.append(t1)

        t1 = benchmark.Timer(
            stmt="(loss(x, 2.) + loss(y, 2.)).backward()",
            setup=setup_backward,
            globals={"device": device, "N": N, "D": D, "loss": uniform_loss},
            num_threads=num_threads,
            label="Forward + Backward",
            sub_label=str(N),
            description="Uniform",
        ).blocked_autorange(min_run_time=min_run_time)

        results.append(t1)
    return results


def compare_to_df(results) -> pd.DataFrame:
    obj_results = []
    for measurement in results:
        for t in measurement._sorted_times:
            obj_results.append({
                "title": measurement.title,
                "label": measurement.label,
                "sub_label": int(measurement.sub_label),
                "description": measurement.description,
                "time": t,
            })
    return pd.DataFrame(obj_results)


def do_plots(results):
    df = compare_to_df(results)
    print(df.head())

    for label in df.label.unique():
        subdf = df[df.label == label]
        plt.rc('font', size=15)
        plt.title(label)
        plt.ylabel("Seconds")
        plt.xlabel("Batch size")
        sns.lineplot(
            data=subdf,
            x="sub_label",
            y="time",
            hue="description",
        )
        plt.grid()
        plt.legend(loc="upper left")
        plt.savefig(f"./benchmark_{label.lower()}.pdf")
        plt.close('all')


def main():
    results = do_benchmarks()
    # compare = benchmark.Compare(results)
    # compare.colorize(rowwise=True)
    # compare.print()

    print(results)

    do_plots(results)


if __name__ == "__main__":
    main()

