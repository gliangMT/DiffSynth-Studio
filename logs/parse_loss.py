import re
import csv
import argparse


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


def main(log1, log2, output):

    data1 = parse_log(log1)
    data2 = parse_log(log2)

    all_steps = sorted(set(data1.keys()) | set(data2.keys()))

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch_log1", "loss_log1", "epoch_log2", "loss_log2"])

        for step in all_steps:
            e1, l1 = data1.get(step, ("", ""))
            e2, l2 = data2.get(step, ("", ""))
            writer.writerow([step, e1, l1, e2, l2])

    print(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log1", required=True)
    parser.add_argument("--log2", required=True)
    parser.add_argument("--output", default="loss_compare.csv")
    args = parser.parse_args()

    main(args.log1, args.log2, args.output)