# RAIL-BENCH

This is the official toolbox for [RAIL-BENCH](https://www.mrt.kit.edu/railbench/), the world's first perception benchmark suite for the railway. 

It includes evaluation scripts for all five RAIL-BENCH challenges:

- **RAIL-BENCH Rail**: Rail Track Detection
- **RAIL-BENCH Object**: Object Detection
- **RAIL-BENCH Vegetation**: Vegetation Segmentation
- **RAIL-BENCH Tracking**: Multi Object Tracking *(coming soon)*
- **RAIL-BENCH Odometry**: Monocular Visual Odometry *(coming soon)*


## Getting Started

### Requirements

- Python >= 3.12

### Setting up the Python environment

It is recommended to use a virtual environment to keep dependencies isolated.

**1. Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install the benchmark suite and all dependencies:**

```bash
pip install -e .
```

This installs the package in editable mode along with all required dependencies:
[`shapely`](https://shapely.readthedocs.io/), [`numpy`](https://numpy.org/), [`opencv-python`](https://pypi.org/project/opencv-python/), [`scikit-learn`](https://scikit-learn.org/), [`scipy`](https://scipy.org/), [`networkx`](https://networkx.org/), [`matplotlib`](https://matplotlib.org/), [`tqdm`](https://tqdm.github.io/).

# Running an Evaluation

## RAIL-BENCH Rail

**1. Prepare evaluation**

- Create a project folder inside `Benchmarks/RAILBENCH_Rail/data/` (e.g. `my_project`) with two subfolders: `annotations/` and `detectors/`.
- Put your ground-truth file in the `annotations/` folder and name it `annotations_[split].json` (e.g. `annotations_val.json`).
- For each model or detector you want to evaluate, create a subfolder inside `detectors/` and add that model's prediction file named `pred_[split].json` (e.g. `pred_val.json`).
- Make sure all ground-truth and prediction files follow the formats described in `Benchmarks/RAILBENCH_Rail/format/gt_format.md` and `Benchmarks/RAILBENCH_Rail/format/pred_format.md`, respectively. 

The `Benchmarks/RAILBENCH_Rail/data/` could then look like this: 

```text
Benchmarks/RAILBENCH_Rail/data/
└── my_project/
	├── annotations/
	│   └── annotations_val.json         # ground-truth file for the "val" split
	└── detectors/
		├── ModelA/
		│   └── pred_val.json            # ModelA predictions for the "val" split
		└── ModelB/
		    ├── pred_val.json            # ModelB predictions for the "val" split
		    └── pred_test.json           # optional: predictions for another split
```

**2. Run evaluation**

```bash
cd Benchmarks/RAILBENCH_Rail
python run_rail_eval.py [-h] [--metric {ChamferAP,LineAP}] [--project PROJECT] [--overwrite]
```

The results are placed in a new folder `results` in your project folder. 

## RAIL-BENCH Object

**1. Prepare evaluation**

Follow the first step of **RAIL-BENCH Rail** to set up the same data structure inside `Benchmarks/RAILBENCH_Object/data/`, but with the object detection annotations and predictions. 

**2. Run evaluation**

```bash
cd Benchmarks/RAILBENCH_Object
python run_object_eval.py [-h] [--project PROJECT] [--overwrite]
```

The results are placed in a new folder `results` in your project folder. 

## RAIL-BENCH Vegetation

For evaluation you need to specify the path to your ground truth masks (`gt_path`) and predicted masks (`pred_path`) as well as which `split` you are evaluating on. Note, that the evaluation script assumes to find a folder named as the specific `split` (e.g. `val`) under `gt_path` and `pred_path`. 

With `expected_num_gt_files` you have the option to perform a sanity check on the number of gt files that you expect to evaluate. The argument is not required. 

```bash
cd Benchmarks/RAILBENCH_Vegetation
python run_veg_eval.py [-h] [--split SPLIT] [--pred_path PRED_PATH] [--gt_path GT_PATH] [--expected_num_gt_files EXPECTED_NUM_GT_FILES] [--project_name PROJECT_NAME] [--overwrite]
```

