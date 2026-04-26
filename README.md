This project evaluates and compares five edge detection pipelines on satellite imagery across three different environments

## Input Images

All source images in `assets/`:

| File | Description |
|---|---|
| `assets/RuralSJ.jpg` | Rural San Jose satellite view |
| `assets/SuburbanSJ.jpg` | Suburban San Jose satellite view |
| `assets/UrbanSF.jpg` | Urban San Francisco satellite view |

## Pipeline Files

| File | Method | Description |
|---|---|---|
| `canny.py` | Simple Canny | Gaussian blur (7×7) followed by OpenCV Canny with fixed thresholds (100/200). Baseline pipeline. |
| `complex_canny.py` | Complex Canny | Bilateral filter (edge-preserving) + Canny + connected-component noise removal to discard small spurious edges. |
| `ridge.py` | Frangi Ridge Filter | Frangi vesselness filter (scikit-image) tuned to detect thin, elongated ridge structures (building edges) in satellite images. |
| `accumulator.py` | Canny Accumulator | Runs 16 Canny passes across 4 blur sizes and 4 threshold pairs; pixels that appear in ≥3 passes are kept, reducing noise via voting. |
| `complex_accumulator.py` | Complex Accumulator | Combines the 16-pass Canny accumulator with a Frangi ridge detection layer and a final connected-components cleanup step for the most robust output. |

`main.py` is the entry point — it reads every image from `assets/`, runs all five pipelines, and writes results to the corresponding output folder.

## Results

Each pipeline writes its output images to a dedicated folder, mirroring the filenames in `assets/`:

| Folder | Pipeline |
|---|---|
| `cannyResults/` | Simple Canny |
| `complexCannyResults/` | Complex Canny |
| `ridgeResults/` | Frangi Ridge |
| `accumulatorResults/` | Canny Accumulator |
| `complexAccumulatorResults/` | Complex Accumulator |

## Running the Code

```bash
python main.py
```

Requires `opencv-python`, `numpy`, and `scikit-image`.

```bash
pip install opencv-python numpy scikit-image
```
