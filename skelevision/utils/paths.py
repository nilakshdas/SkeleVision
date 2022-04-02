from pathlib import Path
from typing import Union

from skelevision.constants import ROOT_DIR


def resolve_path(path: Union[str, Path]) -> Path:
    PREFIX = "skelevision://"
    if isinstance(path, str) and path.startswith(PREFIX):
        path = ROOT_DIR / path[len(PREFIX) :]
    return Path(path).resolve()
