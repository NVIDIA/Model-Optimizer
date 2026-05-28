# Expert pruning for Nemotron3 Nano 30B-A3B

`nemotron_nano_v3_pruneexp.yaml` holds a configuration for heterogeneous pruning number of experts
from 2944 (23 MoE layers * 128 experts per layer) down to 1472.

Run

```bash
torchrun --nproc_per_node 2 examples/puzzletron/main.py --config examples/puzzletron/configs/nemotron-nano-30b-A3b-v3/nemotron_nano_v3_pruneexp.yaml 2>&1 | tee ./log.txt | grep "Puzzletron Progress"
```
