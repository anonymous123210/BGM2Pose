# BGM2Pose: Active 3D Human Pose Estimation with Non-Stationary Sounds

This document outlines the steps to set up the environment, download the necessary data, and run the pose estimation experiment.

## 1. Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.10:** This project requires Python version 3.10. You can check your version using `python --version` or `python3 --version`.
* **uv:** The command-line tool used for environment and package management in this project. If you haven't installed it yet, please follow the official `uv` installation guide: [https://astral.sh/docs/uv#installation](https://astral.sh/docs/uv#installation)


## 2. Download Dataset

1.  Download the `dataset_spec_2400.zip` file from the following URL:
    [URL](https://www.kaggle.com/datasets/anonymous54321/acoustic-music-based-pose-learning-dataset-ampl)
2.  Unzip the downloaded file (dataset_spec_2400) into the `pose_estimation/` directory within your project structure.

    ```bash
    # Example assuming the zip file is downloaded in the project root
    unzip dataset_spec_2400.zip -d pose_estimation/ 
    ```

### Dataset Contents Description

After unzipping, the `pose_estimation/` directory will contain the necessary data. Key components include:

* `music_with_intensity/`: This directory stores the preprocessed data for each time step, specifically the Mel spectrogram combined with the Intensity Vector.
* `music_intensity.csv`: This CSV file provides the mapping between the ground truth human pose (obtained via motion capture) at each time step and the path to the corresponding feature file located within the `music_with_intensity/` directory.

## 3. Set Up Virtual Environment

This project uses `uv` for environment management. Ensure you have `uv` installed.

1.  **Sync Dependencies:** Create the virtual environment and install the required packages listed in your project's configuration file (e.g., `pyproject.toml` or `requirements.txt`).
    ```bash
    uv sync
    ```

2.  **Activate Environment:** Activate the newly created virtual environment.
    ```bash
    source .venv/bin/activate
    ```
    Your command prompt should now indicate that you are inside the virtual environment.

## 4. Create Training/Validation Data & Run Evaluation

Once the dataset is in place and the virtual environment is active:

1.  **Navigate to Directory:** Change your current directory to `pose_estimation/`.
    ```bash
    cd pose_estimation
    ```

2.  **Run Experiment Script:** Execute the experiment script. This script is expected to handle the creation of training/validation data splits and perform the accuracy evaluation using k-fold cross-validation for each 
    ```bash
    sh scripts/experiment_music2pose_ambient_kfold.sh
    ```

This will start the process defined within the script. Monitor the output for progress and results.

3.  **Verify Output:** After the script completes successfully, verify that the qualitative result video `test.mp4` has been generated. Check for the file in the following directory:
    `pose_estimation/result/exp-title/images/`

    *(Note: The `exp-title` part of the directory path is likely determined by the experiment script, representing experimental configuration (hyperparameters). Please check the actual directory name generated.)*

## 5. Important Notes on Subject IDs

Please be aware that due to some missing or corrupted motion capture (mocap) data, there is a mismatch between the subject IDs used in the **currently available dataset files** and the subject IDs referenced in the **accompanying paper**.

We plan to correct this inconsistency in a future update. For now, the correct mapping between the raw data subject IDs (found in the dataset files) and the paper's subject IDs is as follows:

| Raw Data Subject ID | Paper Subject ID     |
|---------------------|----------------------|
| 0                   | 1                    |
| 1                   | 9                    |
| 2                   | 6                    |
| 3                   | *not used (broken)* |
| 4                   | 7                    |
| 5                   | 8                    |
| 6                   | 2                    |
| 7                   | 3                    |
| 8                   | 4                    |
| 9                   | 5                    |

Please use this mapping when comparing results or data points between the current dataset and the paper.

## 6. Other Information

* **Music Combinations (`music_cv_ver`):** To understand which combination of music corresponds to the `music_cv_ver` setting used in various experiments, please refer to the following file:
    ```
    libs/music_setting.py
    ```
    This file contains the definitions or mappings for different `music_cv_ver` values.
