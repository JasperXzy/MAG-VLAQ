from pathlib import Path
import sys


def _setup_paths():
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent]
    for path in reversed(candidates):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_setup_paths()

from lit.cli import SCALightningCLI


def cli_main():
    SCALightningCLI(
        model_class="lit.module:SCAModule",
        datamodule_class="lit.datamodule:SCADataModule",
    )


if __name__ == "__main__":
    cli_main()
