# RAIL-BENCH

This is the official toolbox for [RAIL-BENCH](https://www.mrt.kit.edu/railbench/), the world's first perception benchmark suite for the railway. 

It includes evaluation scripts for all five RAIL-BENCH challenges:

- **RAIL-BENCH Rail**: Rail Track Detection
- **RAIL-BENCH Object**: Object Detection
- **RAIL-BENCH Vegetation**: Vegetation Segmentation
- **RAIL-BENCH Tracking**: Multi Object Tracking
- **RAIL-BENCH Odometry**: Monocular Visual Odometry *(coming soon)*

-----
This readme is structured as follows:

1. [Getting Started](#1-getting-started): how to set up your python environment
2. [Visualize Annotations](#2-visualize-annotations): guide for visualizing annotations 
3. [Format Checks](#3-format-checks): guide for checking the format of your predictions before submission
4. [Running an Evaluation](#4-running-an-evaluation): run evaluation with the official RAIL-BENCH evaluation metrics
5. [Citations](#5-citation)


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

or using a conda environment

```bash
conda create -n railbench_toolkit python=3.12 -y
conda activate railbench_toolkit
```

**2. Install the benchmark suite and all dependencies:**

```bash
pip install -e .
```

This installs the package in editable mode along with all required dependencies:
[`shapely`](https://shapely.readthedocs.io/), [`numpy`](https://numpy.org/), [`opencv-python`](https://pypi.org/project/opencv-python/), [`scikit-learn`](https://scikit-learn.org/), [`scipy`](https://scipy.org/), [`networkx`](https://networkx.org/), [`matplotlib`](https://matplotlib.org/), [`tqdm`](https://tqdm.github.io/).

# 2 Visualize Annotations

To visualize the annotations use the functions in the folder annotation_visualizer. 

How to visualize rails: 

```bash
python annotation_visualizer/visualize_rails.py 
  --annotations your_annotation_path/annotations_train.json 
  --image_dir your_image_path/val
```

Visualization tools for the other benchmarks will follow soon ...


# 3 Format checks

We provide format checks to ensure that your prediction files are correctly formated. Depending on the challenge, the functions check the general formatting and/or provide specific checks for the respective RAIL-BENCH challenges. 

> **💡 Tip:** If you want to participate in a challenge, we advise you to check if your predictions are correctly formatted using the respective `check_formatting.py` function before submission. For all challenges you can find a folder `format` with a guide on the correct formatting. 

Currently, a check function is missing for the RAIL-BENCH odometry is missing, but will follow soon.  

## 3.1 RAIL-BENCH Rail and RAIL-BENCH Object

### Preparation

Save your predictions in a JSON file named  `pred_test.json` following the formatting rules in the folders `Benchmarks/RAILBENCH_Rail/format` or `Benchmarks/RAILBENCH_Object/format`, respectively. For performing a format check, this JSON file can be placed anywhere, you only need to provide the path to the file.  

### Run the check

```bash
cd Benchmarks/RAILBENCH_Rail 
# or 
cd Benchmarks/RAILBENCH_Object 
```

Use the respective `check_formatting.py` function to check whether your JSON file is correctly formated. 
 - with `pred_file` you specify the path to your json file. 
 - optionally: set the `--is_railbench_test` flag if you have a submission to the RAIL-BENCH challenge, this will initate additional checks.


```bash
python check_formatting.py [-h] [--pred_file PRED_FILE] [--is_railbench_test]
```

## 3.2 RAIL-BENCH Vegetation

### Preparation

Save your predicted masks in a single folder. Follow the rules in `Benchmarks/RAILBENCH_Vegetation/format/format_rules.md`.

### Run the check

```bash
cd Benchmarks/RAILBENCH_Vegetation
```

Use the function `check_formatting.py` to check whether you submission file is correctly formated. 
With `pred_path` you specify the path to the folder with your predicted masks. 

```bash
python check_formatting.py [-h] [--pred_path PRED_PATH]
```

## 3.3 RAIL-BENCH Tracking

### Preparation

Save all predictions for each sequence in an individual JSON file and place all JSON files in a single folder. Follow the rules in `Benchmarks/RAILBENCH_Tracking/format/pred_format.md`.

If you submission is for the RAIL-BENCH tracking challenge, name the files: 'scene_31_test.json', 'scene_32_test.json', 'scene_33_test.json', and 'scene_34_test.json'. 

### Run the check

```bash
cd Benchmarks/RAILBENCH_Tracking
```

Use the function `check_formatting.py` to check whether you submission file is correctly formated. 
- with `pred_path` you specify the path to the folder with your JSON files. 
- optionally: set the `--is_railbench_test` flag if you have a submission to the RAIL-BENCH challenge, this will initate additional checks.

```bash
python check_formatting.py [-h] [--pred_path PRED_PATH] [--is_railbench_test]
```

# 4 Running an Evaluation

## 4.1 RAIL-BENCH Rail

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

## 4.2 RAIL-BENCH Object

**1. Prepare evaluation**

Follow the first step of **RAIL-BENCH Rail** to set up the same data structure inside `Benchmarks/RAILBENCH_Object/data/`, but with the object detection annotations and predictions. 

**2. Run evaluation**

```bash
cd Benchmarks/RAILBENCH_Object
python run_object_eval.py [-h] [--project PROJECT] [--overwrite]
```

The results are placed in a new folder `results` in your project folder. 

## 4.3 RAIL-BENCH Vegetation

For evaluation you need to specify the path to your ground truth masks (`gt_path`) and predicted masks (`pred_path`) as well as which `split` you are evaluating on. Note, that the evaluation script assumes to find a folder named as the specific `split` (e.g. `val`) under `gt_path` and `pred_path`. 

With `expected_num_gt_files` you have the option to perform a sanity check on the number of gt files that you expect to evaluate. The argument is not required. 

```bash
cd Benchmarks/RAILBENCH_Vegetation
python run_veg_eval.py [-h] [--split SPLIT] [--pred_path PRED_PATH] [--gt_path GT_PATH] [--expected_num_gt_files EXPECTED_NUM_GT_FILES] [--project_name PROJECT_NAME] [--overwrite]
```

## 4.4 RAIL-BENCH Tracking

**1. Prepare evaluation**

- Create a project folder inside `Benchmarks/RAILBENCH_Tracking/data/` (e.g. `my_project`) with two subfolders: `gt/` and `trackers/`.
- For each video sequence, put your corresponing ground-truth file in the `gt/` folder.
- For each tracker that you want to evaluate (e.g. `TrackerA`), create the subfolders `trackers/TrackerA/data` and add that tracker's prediction files inside this new subfolder `data`.
- Naming convention: All files (gt and predictions) that correspond to the same sequence must have the same name. 
- Make sure all ground-truth and prediction files follow the formats described in `Benchmarks/RAILBENCH_Tracking/format/gt_format.md` and `Benchmarks/RAILBENCH_Tracking/format/pred_format.md`, respectively. 

The `Benchmarks/RAILBENCH_Tracking/data/` folder could then look like this: 

```text
Benchmarks/RAILBENCH_Tracking/data/
└── my_project/
    ├── gt/
    │   ├── scene_31_test.json         # ground-truth file for sequence 31
    │   ├── scene_32_test.json         # ground-truth file for sequence 32
    │   └── ...
    └── trackers/
        ├── TrackerA/
        │   └── data/                    
        │     ├── scene_31_test.json         # predictions by TrackerA for sequence 31
        │     ├── scene_32_test.json         # predictions by TrackerA for sequence 32
        │     └── ...
        ├── TrackerB/
        │     ├── scene_31_test.json         # predictions by TrackerB for sequence 31
        │     ├── scene_32_test.json         # predictions by TrackerB for sequence 32
        │     └── ...
      ...
```

**2. Run evaluation**

```bash
cd Benchmarks/RAILBENCH_Tracking
python run_tracking_eval.py [-h] [--project my_project]
```

You can find the results within each trackers' subfolder as well as combined in `Benchmarks/RAILBENCH_Tracking/data/my_project/`. 


# 5 Citation

If you use this software, please cite our work:

```bibtex
@misc{bätz2026railwayartificialintelligencelearning,
      title={Railway Artificial Intelligence Learning Benchmark (RAIL-BENCH): A Benchmark Suite for Perception in the Railway Domain}, 
      author={Annika Bätz and Pavel Klasek and Seo-Young Ham and Philipp Neumaier and Martin Köppel and Martin Lauer},
      year={2026},
      eprint={2604.22507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.22507}, 
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

The **RAIL-BENCH Tracking** evaluation builds on code from Jonathon Luiten's [`TrackEval`](https://github.com/JonathonLuiten/TrackEval). If you publish results produced with this part of the toolkit, please also cite:

```bibtex
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}

@article{luiten2020IJCV,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and Osep, Aljosa and Dendorfer, Patrick and Torr, Philip and Geiger, Andreas and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  journal={International Journal of Computer Vision},
  pages={1--31},
  year={2020},
  publisher={Springer}
}
```

See the [`NOTICE`](NOTICE) file for full details on this obligation.

## License & Acknowledgements

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.

Portions of this software are derived from third-party open-source projects.
See the [`NOTICE`](NOTICE) file for full attribution and their license terms.

