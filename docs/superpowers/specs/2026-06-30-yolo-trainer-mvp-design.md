# YOLO Trainer Visual Frontend MVP Design

Date: 2026-06-30

## Goal

Build a local single-user YOLO training workbench that closes the loop from raw drone image datasets to annotation, dataset quality review, model training, experiment analysis, failure mining, annotation correction, retraining, prediction review, and export.

The MVP is not a CVAT clone or a full research framework. It is a focused local tool for one operator to construct high-quality YOLO datasets and run repeatable YOLO experiments with strong visibility into failure cases.

## Confirmed Product Decisions

- Runtime shape: local single-user app.
- Frontend/backend: React/Vite frontend plus FastAPI backend.
- State: SQLite plus a managed file workspace.
- Dataset import: copy or unzip input data into the app workspace.
- Current input example: `image_dataset.zip` contains raw images, `vtol` and `iris` collection folders, and `meta.jsonl`; it has no YOLO labels or `data.yaml`.
- Label model: collection metadata and training labels are separate.
- Class model: project-level class library; each dataset version selects a subset and freezes its class mapping.
- Annotation: built-in lightweight box editor.
- Temporal support: optional lightweight `track_id` metadata on boxes.
- Edge cases: manual semantic tags plus automatic analysis tags.
- Dataset versions: create a lightweight frozen snapshot before each training run.
- Training: dual-layer mode. Stable Ultralytics YOLO path by default, with a constrained experimental settings panel.
- Models: built-in presets plus custom local `.pt` weights.
- Hardware: auto-detect `cuda`, `mps`, or `cpu`; show advanced acceleration only when available.
- Task model: one active training task at a time, with run history.
- Experiment dashboard: include training curves, logs, confusion matrix, PR/F1, class metrics, threshold scanning, and failure samples. The deepest MVP loop is failure mining back into the annotator.
- Prediction analysis: run predictions on arbitrary filtered image sets.
- Export: save `.pt` weights, support ONNX export, and conditionally expose TensorRT export when the environment supports it.

## Scope

### In Scope

- Import zip or folder datasets into a managed workspace.
- Parse image files and `meta.jsonl` collection metadata when present.
- Maintain projects, datasets, class library, image metadata, annotations, edge-case tags, dataset versions, training runs, metrics, prediction outputs, and exports.
- Provide a lightweight annotator for bounding boxes, classes, deletion, editing, previous-frame copy, optional `track_id`, and edge-case tagging.
- Generate YOLO-compatible labels and `data.yaml` from frozen dataset versions.
- Run Ultralytics YOLO training from the backend.
- Stream training metrics and logs to the frontend.
- Store run artifacts, including config, weights, metrics, plots, prediction outputs, and failure analysis.
- Batch-predict over filtered image sets.
- Link failed predictions back to the annotator for correction.
- Export ONNX, and expose TensorRT only when the backend detects the needed environment.

### Out of Scope for MVP

- Multi-user collaboration, permissions, and remote server deployment.
- Full CVAT-class annotation workflows.
- Complex automatic tracking and situation-map stitching.
- Guaranteed support for custom YOLO architecture surgery, custom necks, custom heads, or custom loss functions.
- Full DVC/Git-style dataset versioning.
- Parallel training or hyperparameter sweep scheduling.
- Cloud storage and cloud training.

## Architecture

### Frontend

React/Vite provides a full browser UI with these major routes:

- Project dashboard: project selection, dataset status, latest runs, quality warnings.
- Dataset import: import zip/folder, preview parsed structure, confirm copy into workspace.
- Dataset browser: filter by collection platform, height, time, class coverage, label status, edge-case tags, and failure tags.
- Annotator: image canvas, box list, class picker, track ID field, edge-case tags, previous-frame copy, save status.
- Class library: project-level classes, dataset class subset, frozen class-index preview.
- Quality review: empty labels, tiny boxes, out-of-bounds boxes, duplicate boxes, unknown classes, unannotated images.
- Training setup: dataset version creation, model preset/custom `.pt`, core hyperparameters, device, augmentation presets, TTA and threshold scan options.
- Training monitor: active run status, logs, loss and metric curves, current artifacts.
- Experiment dashboard: run history, run detail, class metrics, confusion matrix, PR/F1, threshold scan, failure samples, filtered prediction analysis.
- Export panel: `.pt` artifacts, ONNX export, conditional TensorRT export.

### Backend

FastAPI owns durable state, workspace file operations, annotation persistence, training orchestration, prediction jobs, analysis jobs, and export jobs.

Suggested backend modules:

- `settings`: workspace paths, database URL, device detection, feature flags.
- `db`: SQLAlchemy models, migrations, session lifecycle.
- `workspace`: dataset copy/unzip, path normalization, artifact layout.
- `datasets`: import, metadata parsing, split management, quality checks.
- `classes`: project class library and dataset subset mapping.
- `annotations`: box CRUD, YOLO export conversion, track and edge-case metadata.
- `versions`: frozen dataset version snapshots.
- `training`: Ultralytics adapter, run lifecycle, metric parsing, log streaming.
- `prediction`: filtered batch prediction and prediction artifact storage.
- `analysis`: failure mining, threshold scanning, metric summaries.
- `exports`: ONNX and conditional TensorRT export.

### File Workspace

The app manages a workspace directory rather than relying on original input paths.

```text
workspace/
  projects/
    <project_id>/
      datasets/
        <dataset_id>/
          source/
          images/
          meta/
      versions/
        <version_id>/
          images/
          labels/
          data.yaml
          manifest.json
      runs/
        <run_id>/
          config.yaml
          metrics.jsonl
          logs.txt
          weights/
          predictions/
          analysis/
          exports/
```

Images may be copied or hard-linked inside frozen versions depending on filesystem support, but the snapshot manifest must make a run reproducible even if annotations later change.

## Data Model

SQLite should index the file workspace. Minimum entities:

- `Project`: name, workspace path, created time.
- `ClassDef`: project ID, name, color, description, active flag.
- `Dataset`: project ID, name, source type, import status, image count.
- `Image`: dataset ID, relative path, width, height, collection metadata, timestamp, platform, altitude.
- `Annotation`: image ID, class ID, normalized bbox, optional `track_id`, quality flags, created/updated time.
- `EdgeCaseTag`: project ID, tag name, source type `manual` or `auto`.
- `ImageTag` and `AnnotationTag`: many-to-many tag links.
- `DatasetVersion`: project ID, dataset ID, class mapping, split manifest, frozen status, created time.
- `TrainingRun`: project ID, version ID, config, status, device, started/ended time, artifact path.
- `RunMetric`: run ID, epoch/step, metric name, value.
- `Prediction`: run ID, image ID, class ID, bbox, confidence, matched annotation ID, failure type.
- `ExportArtifact`: run ID, format, path, status, metadata.

YOLO labels remain standard `class x_center y_center width height`. `track_id`, collection metadata, and edge-case tags are stored outside YOLO label files.

## Key Workflows

### Import Dataset

1. User selects a zip or folder.
2. Backend copies or extracts it into the workspace.
3. Backend scans images and optional metadata files.
4. Frontend shows detected image counts, collection groups, and missing label status.
5. User confirms dataset creation.

### Annotate and Review

1. User opens dataset browser and filters images.
2. User draws or edits boxes in the annotator.
3. User assigns classes from the project class library.
4. User optionally sets `track_id` for sequence continuity.
5. User applies manual edge-case tags such as occluded, camouflaged, low light, dense, or hard negative.
6. Automated checks add tags such as tiny object, dense objects, empty label, invalid box, low-confidence prediction, false positive, false negative, or class confusion.
7. Quality review lists actionable issues and links back to the annotator.

### Train

1. User opens training setup.
2. User selects dataset, split, class subset, model preset/custom `.pt`, device, epochs, image size, batch size, augmentation settings, and optional TTA/threshold scan. If no split exists, the MVP creates a deterministic default split of 80% train, 10% validation, and 10% test, stratified by class presence when enough labels exist.
3. Backend freezes a dataset version, writes YOLO labels and `data.yaml`, and starts a single training task.
4. Frontend streams logs and metrics.
5. Backend stores weights, metrics, plots, predictions, and analysis under the run directory.

### Analyze and Improve

1. User opens the experiment dashboard.
2. Dashboard shows run curves, class metrics, confusion matrix, PR/F1, threshold scan, and failure samples.
3. User filters failures by class, confidence, edge case, platform, altitude, time range, or failure type.
4. User opens a failure sample in the annotator.
5. User fixes labels or tags the sample.
6. User starts a new run from a new dataset version.

### Predict and Export

1. User selects a trained run and a filtered image set.
2. Backend runs batch prediction and stores results.
3. Frontend overlays predictions and ground truth for review.
4. User exports `.pt`, ONNX, or TensorRT if available.

## Error Handling

- Dataset import must report unsupported files, corrupt images, duplicate names, missing metadata, and empty datasets.
- Annotation saves must validate normalized bbox ranges, positive width/height, known class IDs, and image ownership.
- Training setup must block runs with no labeled images, no class mapping, invalid split, or missing model weights.
- Default split generation must be deterministic for a given dataset version so runs are reproducible.
- Training task status must include queued, preparing, running, completed, failed, and cancelled states, even though MVP runs only one active training task.
- Device detection must degrade gracefully from CUDA to MPS to CPU.
- Export must show unsupported TensorRT conditions instead of failing silently.

## Testing And Verification

- Backend unit tests for import scanning, metadata parsing, bbox validation, YOLO export, class mapping freeze, quality checks, and filtered prediction query construction.
- Backend integration test for creating a dataset version and writing `data.yaml` plus YOLO labels.
- Training adapter smoke test with a tiny dataset fixture and a tiny model when available.
- Frontend component tests for class selection, annotation save state, quality issue links, and run status rendering.
- End-to-end browser smoke test for import, annotate one box, create version, reach training setup, and render dashboard shell.
- Manual verification on the provided `image_dataset.zip`: import should find 1099 images, two collection groups, metadata rows, and no initial YOLO labels.

## MVP Acceptance Criteria

- The app can import the provided raw image zip into a managed workspace.
- The app separates `vtol` and `iris` as collection metadata rather than training classes.
- A user can create project-level classes, choose a dataset subset, annotate boxes, add edge-case tags, and save them.
- Quality review can identify at least empty labels, tiny boxes, invalid boxes, and unknown class references.
- Starting training creates a frozen dataset version with YOLO labels and `data.yaml`.
- One training run can be launched and monitored.
- Run history persists after app restart.
- The dashboard can show run metrics, predictions, class metrics, failure samples, and links back to annotation.
- Batch prediction can run on a filtered image set.
- ONNX export is available for a completed run.
- TensorRT export is shown only when the environment supports it.

## Implementation Order

1. Scaffold React/Vite frontend, FastAPI backend, SQLite models, workspace settings, and dev scripts.
2. Implement dataset import and metadata scanning for the provided zip shape.
3. Implement project class library and dataset class subset.
4. Implement image browser and lightweight annotation save/load.
5. Implement quality review and dataset version snapshot export.
6. Implement Ultralytics training adapter and one-active-run lifecycle.
7. Implement training monitor and run history.
8. Implement prediction analysis and failure mining.
9. Implement experiment dashboard views.
10. Implement ONNX export and conditional TensorRT capability detection.
