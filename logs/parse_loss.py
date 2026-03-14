import re
import csv
import argparse
import matplotlib.pyplot as plt


pattern = re.compile(r"epoch=(\d+)\s+step=(\d+)\s+loss=([0-9.eE+-]+)")


def parse_log(log_file):
    data = {}
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                step = int(m.group(2))
                loss = float(m.group(3))
                data[step] = (epoch, loss)
    return data


def determine_max_step(data1, data2):

    max1 = max(data1.keys()) if data1 else 0
    max2 = max(data2.keys()) if data2 else 0

    if max1 >= 1000 or max2 >= 1000:
        return 1000
    else:
        return min(max1, max2)


def main(log1, log2, output):

    data1 = parse_log(log1)
    data2 = parse_log(log2)

    max_step = determine_max_step(data1, data2)

    steps = []
    loss1 = []
    loss2 = []
    rel_err = []

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch_log1", "loss_log1", "epoch_log2", "loss_log2"])

        for step in range(max_step + 1):

            e1, l1 = data1.get(step, ("", ""))
            e2, l2 = data2.get(step, ("", ""))

            writer.writerow([step, e1, l1, e2, l2])

            if l1 != "" and l2 != "":
                steps.append(step)
                loss1.append(l1)
                loss2.append(l2)

                if l1 != 0:
                    rel_err.append(abs(l1 - l2) / abs(l1))
                else:
                    rel_err.append(0)

    print(f"Saved to {output}")

    # -----------------------
    # 绘制 loss 曲线
    # -----------------------

    plt.figure()

    plt.plot(steps, loss1, color="green", label=f"{log1}")
    plt.plot(steps, loss2, color="orange", label=f"{log2}")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison")
    plt.legend()

    plt.grid(True)

    plt.savefig("loss_curve.png", dpi=400)
    print("Saved loss_curve.png")

    # -----------------------
    # 绘制相对误差
    # -----------------------

    plt.figure()

    plt.plot(steps, rel_err)

    avg_err = sum(rel_err) / len(rel_err) if rel_err else 0

    plt.axhline(avg_err, linestyle="--", label=f"avg={avg_err:.4%}")

    plt.xlabel("Step")
    plt.ylabel("Relative Error")
    plt.title("Relative Error Between Logs")

    plt.legend()
    plt.grid(True)

    plt.savefig("relative_error.png", dpi=200)
    print("Saved relative_error.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--log1", required=True)
    parser.add_argument("--log2", required=True)
    parser.add_argument("--output", default="loss_compare.csv")

    args = parser.parse_args()

    main(args.log1, args.log2, args.output)