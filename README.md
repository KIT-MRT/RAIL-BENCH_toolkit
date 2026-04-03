# RAIL-BENCH

This is the official toolbox for [RAIL-BENCH](https://www.mrt.kit.edu/railbench/), the world's first perception benchmark suite for the railway. 

It includes evaluation scripts for all five RAIL-BENCH challenges:

- **RAIL-BENCH Rail**: Rail Track Detection
- **RAIL-BENCH Object**: Object Detection
- **RAIL-BENCH Vegetation**: Vegetation Segmentation
- **RAIL-BENCH Tracking**: Multi Object Tracking *(coming soon)*
- **RAIL-BENCH Odometry**: Monocular Visual Odometry *(coming soon)*


# 1 Getting Started

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

# 2 Format checks

We provide format checks to ensure that your prediction files are correctly formated. The functions both check the general formatting, but also provide specific checks for the respective RAIL-BENCH challenges. 

> **💡 Tip:** If you want to participate in a challenge, we advise you to check if your submission file is correctly formatted using the respective `check_formatting.py` function before submission. For some of the challenges you can find a folder `format` with a guide on the correct formatting. 

Currently, we only provide a check function for the RAIL-BENCH Rail and Object challenge, but more will follow. 

## 2.1 RAIL-BENCH Rail and RAIL-BENCH Object

Use the function `check_formatting.py` to check whether you submission file is correctly formated. 
 - with `PRED_FILE` you specify the path to your json file. 
 - optionally: set the `--is_railbench_test` flag if you have a submission to the RAIL-BENCH challenge, this will initate additional checks.

```bash
cd Benchmarks/RAILBENCH_Rail 
# or 
cd Benchmarks/RAILBENCH_Object 
```

```bash
check_formatting.py [-h] [--pred_file PRED_FILE] [--is_railbench_test]
```


# 3 Running an Evaluation

## 3.1 RAIL-BENCH Rail

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

## 3.2 RAIL-BENCH Object

**1. Prepare evaluation**

Follow the first step of **RAIL-BENCH Rail** to set up the same data structure inside `Benchmarks/RAILBENCH_Object/data/`, but with the object detection annotations and predictions. 

**2. Run evaluation**

```bash
cd Benchmarks/RAILBENCH_Object
python run_object_eval.py [-h] [--project PROJECT] [--overwrite]
```

The results are placed in a new folder `results` in your project folder. 

## 3.3 RAIL-BENCH Vegetation

For evaluation you need to specify the path to your ground truth masks (`gt_path`) and predicted masks (`pred_path`) as well as which `split` you are evaluating on. Note, that the evaluation script assumes to find a folder named as the specific `split` (e.g. `val`) under `gt_path` and `pred_path`. 

With `expected_num_gt_files` you have the option to perform a sanity check on the number of gt files that you expect to evaluate. The argument is not required. 

```bash
cd Benchmarks/RAILBENCH_Vegetation
python run_veg_eval.py [-h] [--split SPLIT] [--pred_path PRED_PATH] [--gt_path GT_PATH] [--expected_num_gt_files EXPECTED_NUM_GT_FILES] [--project_name PROJECT_NAME] [--overwrite]
```

# 4 Citation

If you use this software, please cite our work:

```bibtex
@article{baetz2026railbench,
  title   = {Railway Artificial Intelligence Learning Benchmark (RAIL-BENCH): A Benchmark Suite for Perception in the Railway Domain},
  author  = {B{\"a}tz, Annika and Klasek, P. and Ham, S.-Y. and Neumaier, P. and K{\"o}ppel, M. and Lauer, M.},
  note    = {Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Under review.},
  year    = {2026}
}
```

See [`CITATION.cff`](CITATION.cff) for a machine-readable citation file.

Additionally, the **RAIL-BENCH Object** evaluation (and the AP computation in **RAIL-BENCH Rail**) builds on code from Rafael Padilla's [`review_object_detection_metrics`](https://github.com/rafaelpadilla/review_object_detection_metrics). If you publish results produced with these parts of the toolkit, please also cite:

```bibtex
@article{padilla2021comparative,
  title   = {A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit},
  author  = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B.},
  journal = {Electronics},
  volume  = {10},
  number  = {3},
  pages   = {279},
  year    = {2021},
  doi     = {10.3390/electronics10030279}
}
```

See the [`NOTICE`](NOTICE) file for full details on this obligation.

## License & Acknowledgements

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.

Portions of this software are derived from third-party open-source projects.
See the [`NOTICE`](NOTICE) file for full attribution and their license terms.

