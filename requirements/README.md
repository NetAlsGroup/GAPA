# Requirements

Dependency files are split by platform. The current experiment baseline is
Python 3.10. Do not treat `full.txt` as a cross-platform reproducible
environment lock.

Linux server testing records only versions that have been verified in the
target environment. Start with:

```bash
pip install -r requirements/linux-cu118-torch.txt
pip install -r requirements/linux.txt
pip install -e .
```

`requirements/linux.txt` is now the validated Python dependency entry for the
current Linux CUDA 11.8 experiment environment. It records the packages that
were actually needed to run the server agent, remote workflow, and algorithm
smoke tests on the server.

The Linux file is not a strict lockfile. It is a validated top-level
dependency list plus the required DGL wheel source for `torch 2.6 + cu118`.

macOS and Windows have separate files because PyTorch wheels and GPU runtimes
are platform-specific.
