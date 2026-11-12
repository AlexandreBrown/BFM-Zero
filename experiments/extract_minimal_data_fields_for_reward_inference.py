"""Extract only the replay fields needed for the reward-comparison experiments.

The original reward_inference.py already loads and samples its replay buffer.
This helper applies the same sampling rules without loading the full buffer into
RAM, so compact task-specific samples can be transferred from cluster.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from humanoidverse.agents.buffers.trajectory import find_start_stop_traj, get_idxs

TASKS = [
    "move-ego-0-0",
    "move-ego-0-0.7",
    "move-ego-90-0.3",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    "raisearms-l-l",
    "raisearms-m-m",
    "move-arms-0-0.7-m-m",
    "crouch-0.25",
    "sitonground",
]


def gather(dataset, time_index, env_index):
    out = np.empty((len(time_index), *dataset.shape[2:]), dtype=dataset.dtype)
    order = np.argsort(time_index, kind="stable")
    sorted_times = time_index[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(sorted_times)) + 1, len(order)]
    for left, right in zip(cuts[:-1], cuts[1:]):
        positions = order[left:right]
        unique, inverse = np.unique(env_index[positions], return_inverse=True)
        out[positions] = dataset[int(sorted_times[left]), unique][inverse]
    return out


def gather_bulk(dataset, time_index, env_index):
    out = np.empty((len(time_index), *dataset.shape[2:]), dtype=dataset.dtype)
    order = np.argsort(time_index, kind="stable")
    times = time_index[order]
    for start in range(0, dataset.shape[0], 32):
        stop = min(start + 32, dataset.shape[0])
        left, right = np.searchsorted(times, [start, stop])
        if left == right:
            continue
        positions = order[left:right]
        block = dataset[start:stop]
        out[positions] = block[time_index[positions] - start, env_index[positions]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument(
        "--bulk",
        action="store_true",
        help="Gather all task samples in one sequential pass per field",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((args.buffer / "config.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        h5py.File(args.buffer / "buffer.hdf5", "r") as source,
        h5py.File(args.output, "w") as out,
    ):
        end = torch.from_numpy(source[config["end_key"]][:]).squeeze(-1)
        length = config["capacity"] if config["_is_full"] else config["_idx"]
        starts, stops, lengths = find_start_stop_traj(
            end[:length],
            at_capacity=config["_is_full"],
            cursor=config["_idx"] - 1 if config["_is_full"] else None,
        )
        explicit = "next-observation-state" in source
        offset = 0 if explicit else 1
        eligible = lengths >= 1 + offset
        if args.bulk:
            all_indices = []
            for i, task in enumerate(TASKS):
                torch.manual_seed(args.seed + i)
                index = (
                    get_idxs(
                        1,
                        args.samples,
                        lengths[eligible],
                        starts[eligible],
                        config["capacity"],
                        offset=offset,
                    )
                    .long()
                    .numpy()
                )
                all_indices.append(index)
                out.create_group(task).create_dataset("indices", data=index)
            index = np.concatenate(all_indices)
            t, e = index[:, 0], index[:, 1]
            tn = t if explicit else (t + 1) % config["capacity"]
            fields = {
                "state": "observation-state",
                "privileged_state": "observation-privileged_state",
                "qpos": "qpos",
                "qvel": "qvel",
                "action": "action",
            }
            for name, field in fields.items():
                key = ("next-" + field) if explicit and name != "action" else field
                values = gather_bulk(source[key], t if name == "action" else tn, e)
                for i, task in enumerate(TASKS):
                    out[task].create_dataset(
                        name,
                        data=values[i * args.samples : (i + 1) * args.samples],
                        compression="gzip",
                        compression_opts=1,
                    )
                del values
                out.flush()
                print(f"Extracted all tasks: {name}", flush=True)
            out.attrs["source_config"] = json.dumps(config)
            out.attrs["seed"] = args.seed
            out.attrs["samples_per_task"] = args.samples
            out.attrs["sampling"] = "Native get_idxs; bulk HDF access only"
            return
        for i, task in enumerate(TASKS):
            torch.manual_seed(args.seed + i)
            index = (
                get_idxs(
                    1,
                    args.samples,
                    lengths[eligible],
                    starts[eligible],
                    config["capacity"],
                    offset=offset,
                )
                .long()
                .numpy()
            )
            t, e = index[:, 0], index[:, 1]
            tn = t if explicit else (t + 1) % config["capacity"]
            fields = {
                "state": "observation-state",
                "privileged_state": "observation-privileged_state",
                "qpos": "qpos",
                "qvel": "qvel",
            }
            group = out.create_group(task)
            group.create_dataset("indices", data=index)
            for name, field in fields.items():
                key = "next-" + field if explicit else field
                group.create_dataset(
                    name,
                    data=gather(source[key], tn, e),
                    compression="gzip",
                    compression_opts=1,
                )
            group.create_dataset(
                "action",
                data=gather(source["action"], t, e),
                compression="gzip",
                compression_opts=1,
            )
            out.flush()
            print(
                f"Extracted {task}: {args.samples} native trajectory samples",
                flush=True,
            )
        out.attrs["source_config"] = json.dumps(config)
        out.attrs["seed"] = args.seed
        out.attrs["samples_per_task"] = args.samples
        out.attrs["sampling"] = (
            "Native get_idxs: uniform eligible trajectory, uniform valid start; explicit/implicit successor preserved"
        )


if __name__ == "__main__":
    main()
