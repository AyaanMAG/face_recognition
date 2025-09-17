# LBPH Face Recognition Project

This project uses the Local Binary Patterns Histograms (LBPH) algorithm for face recognition. It includes scripts for training a model and running face recognition on images.

## Project Structure

- `train_and_run_lbph.py`: Main script for training and running LBPH face recognition.
- `faces/`: Directory containing subfolders for each person, with their face images.
- `names.npy`: Numpy file storing names/labels.
- `lbph_face.yml`: Trained LBPH model file.

## Getting Started

1. Place face images in the `faces/` directory, organized by person name.
2. Run `train_and_run_lbph.py` to train the model and perform recognition.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

Install dependencies with:

```bash
pip install opencv-python numpy
```

## Usage

```bash
python train_and_run_lbph.py
```

## License

Specify your license here (e.g., MIT, Apache 2.0).
