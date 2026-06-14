# StableToolBench

This directory contains the StableToolBench implementation, extended with dynamic tool retrieval strategies.

**For usage instructions, see the [root README](../README.md).**

## Origin

StableToolBench is a stable benchmarking framework for tool learning in LLMs, built on [ToolBench](https://github.com/OpenBMB/ToolBench). See the [original paper](https://arxiv.org/abs/2403.07714) and [project page](https://zhichengg.github.io/stb.github.io/) for details on the benchmark design, virtual API server, solvable queries, and evaluation methodology.

## Virtual API Server

The virtual server replays cached tool responses and falls back to LLM-generated responses for unavailable APIs. See `server/config.yml` for configuration.

```bash
# Download the cache from HuggingFace:
# https://huggingface.co/datasets/stabletoolbench/Cache
# Unzip into server/tool_response_cache/ and server/tools/

# Start the server:
bash scripts/run_server.sh
```

## Citation

```bibtex
@misc{guo2024stabletoolbench,
    title={StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models},
    author={Zhicheng Guo and Sijie Cheng and Hao Wang and Shihao Liang and Yujia Qin and Peng Li and Zhiyuan Liu and Maosong Sun and Yang Liu},
    year={2024},
    eprint={2403.07714},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
