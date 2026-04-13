#!/usr/bin/env bash
set -euo pipefail

pip install -e /home/pavel/diplom/PyMlSec

pymlsec scan \
  --mode env \
  --python "$(which python)" \
  --output pymlsec-out \
  --fail-on-severity high \
  --insecure-bdu
