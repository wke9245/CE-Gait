import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cegait.cli import main


def run(name):
    main(["experiment", "--name", name, "--output", f"outputs/{name}", *sys.argv[1:]])
