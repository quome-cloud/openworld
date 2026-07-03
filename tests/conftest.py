import sys, pathlib
# Let tests import the e119 subpackage: put experiments/ on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments"))
