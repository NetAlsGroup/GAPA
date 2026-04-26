# Requirements

Dependency files are split by platform. The current experiment baseline is
Python 3.10. Do not treat `full.txt` as a cross-platform reproducible
environment lock.

Linux server testing records only versions that have been verified in the
target environment. Start with:

```bash
pip install -r requirements/linux-cu118-torch.txt
```

Then add non-Torch dependencies to `requirements/linux.txt` only after they are
confirmed to install and run in the server environment.

macOS and Windows have separate files because PyTorch wheels and GPU runtimes
are platform-specific.
