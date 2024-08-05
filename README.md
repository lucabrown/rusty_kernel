# rusty_kernel

This library is used to create an instance of the Wasserstein Neighbourhood Hash kernel. You can import the `py_graph_kernel` class into a Python environment and use it in conjunction with `scikit_learn` to train it on machine learning datasets. 

This repository contains the source code for the MSc Advanced Computing personal project.

## Installation

Clone the repository:

```bash
git clone https://github.com/lucabrown/rusty_kernel.git

cd rusty_kernel
```

Install the Python environment manager [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and use it to create an environment `env` by specifying the python version:

```bash
conda create --name env python=3.11
``` 

Activate the Python environment:

```bash
conda activate env
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements for the project:

```bash
pip install -r requirements.txt
```

Compile the `rusty_kernel` crate as a Python module:

```bash
maturin develop -r
```

## Usage

Download datasets [here](https://chrsmrrs.github.io/datasets/) and place in the `TEST` folder (MUTAG, BZR_MD, ENZYMES included for convenience). Then run the following to compare the CSNH kernel Rust implementation to the `GraKel` implementation: 

```bash
python3 src/benchmark_rust.py
```

Alternatively, to compare the CSNH kernel to the WNH kernel, run:

```
python3 src/benchmark_wnh.py
```

The Python scripts might take 20/30 extra seconds to execute when running for the first time.