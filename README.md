# egrep.py

A Python clone of the UNIX `egrep` command.

## Get the code

```bash
git clone https://github.com/divcamillediv/egrep.py.git
cd egrep.py
```

The test text `56667-0.txt` (*A History of Babylon*, Project Gutenberg) is in the repository.

## Run without Docker

Requires Python 3.10 or later. No packages to install.

```bash
python egrep.py "S(a|g|r)+on" 56667-0.txt
```

Add `--stats` to see which method was used (KMP or automaton) and the timings:

```bash
python egrep.py --stats "S(a|g|r)+on" 56667-0.txt
```

Tests and benchmarks:

```bash
pip install -r requirements-dev.txt
python -m unittest test_egrep
python bench_egrep.py          # charts and CSV files in bench_results/
```

## Run with Docker

Build the image, then run it:

```bash
docker build -t egrep-py .
docker run --rm egrep-py "S(a|g|r)+on" 56667-0.txt
```

To search your own file, mount the current folder:

```bash
docker run --rm -v "$PWD":/data egrep-py "Babylon" /data/my_file.txt
```

Tests and benchmarks:

```bash
docker build -f Dockerfile.bench -t egrep-py-bench .
docker run --rm egrep-py-bench
docker run --rm -v "$PWD/bench_results":/app/bench_results egrep-py-bench python bench_egrep.py
```