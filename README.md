# SELF-EDIT

## Setup

```bash
uv sync --python /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13
```

## Harness check

This uses a tiny map, with no download. The norm oracle must undo the planted column. A random editor with the same budget of three proposals must accept at most 6 of 40 seeds.

```bash
.venv/bin/python -m selfedit selftest
```

## M3 on a small model

Default checkpoint is `Qwen/Qwen2.5-0.5B-Instruct`,. The same command takes any causal LM that has `mlp.down_proj.weight`.

```bash
.venv/bin/python -m selfedit m3 --rounds 1 --conditions live,oracle,random,blind,no_edit
.venv/bin/python -m selfedit m3
```

