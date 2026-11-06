![ImmuVis logo](./imgs/immuvis_logo.png)

# ImmuVis

ImmuVis is a PyTorch-based masked autoencoder for multiplex imaging data. It learns from images with different biological marker panels by combining marker-conditioned hyperkernels with convolutional or transformer encoders and a marker-conditioned decoder.

This repository provides:

- **Masked image reconstruction and virtual staining:** reconstruct observed markers or predict supported markers that are absent from an input panel.
- **Image and cell representations:** extract embeddings for downstream analysis, including cell phenotyping.
- **Training and fine-tuning:** train on multiple panels or extend a trained model to new markers.
- **Tutorial notebooks:** examples for virtual staining and cell typing.

## Choose a model

ImmuVis comes in three model variants:


| Model            | Encoder architecture | Hugging Face repository                                                     |
| ---------------- | -------------------- | --------------------------------------------------------------------------- |
| **ImmuVis Conv** | ConvNeXt v2          | [SzczurekLab/ImmuVis_Conv](https://huggingface.co/SzczurekLab/ImmuVis_Conv) |
| **ImmuVis Swin** | Swin Transformer     | [SzczurekLab/ImmuVis_Swin](https://huggingface.co/SzczurekLab/ImmuVis_Swin) |
| **ImmuVis ViT**  | Vision Transformer   | [SzczurekLab/ImmuVis_ViT](https://huggingface.co/SzczurekLab/ImmuVis_ViT)   |

All three checkpoints use the same [265-marker tokenizer](./configs/markers_tokenizer.yaml), a latent width of **768**, and the same Python API. Choose a variant by changing the Hugging Face repository ID in the loading example below. The saved checkpoint supplies the architecture automatically; no manual model configuration is needed for loading.

## Getting started

### Requirements

- Python **3.12 or newer**.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip.
- Your own multiplex images and, for inference or fine-tuning, a compatible model checkpoint.

### Install from source

Clone the repository and run subsequent commands from its root directory:

```bash
git clone https://github.com/szczurek-lab/ImmuVis.git
cd ImmuVis
uv sync --python 3.12
uv run python -c "from multiplex_model.modules import MultiplexAutoencoder; print('ImmuVis import successful')"
```

`uv sync` creates a local `.venv` and installs the project in editable mode.

Alternatively, use pip in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Run the tutorials

Install the notebook dependencies and register a kernel:

```bash
uv sync --python 3.12 --extra notebooks
uv run python -m ipykernel install --user --name immuvis --display-name "Python (ImmuVis)"
```

For pip users, install `python -m pip install -e ".[notebooks]"`, then run the kernel registration without `uv run`. Select the **Python (ImmuVis)** kernel.


| Tutorial                                                        | What you will need                                                                         | What it demonstrates                                                             |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| [Virtual staining](./notebooks/tutorial_virtual_staining.ipynb) | Multiplex images, panel metadata, and access to a Hugging Face model | Loading panels, reconstruction, marker prediction, and uncertainty visualization |
| [Cell typing](./notebooks/tutorial_celltyping.ipynb)            | Multiplex images, cell-instance masks, access to a Hugging Face model, and labels for the classifier example | Per-cell patch/context embeddings and a simple cell-type classifier |

Provide your own input data and choose `MODEL_REPO_ID` in each notebook before running it. Both tutorials default to ImmuVis Conv and download the model and matching tokenizer automatically; switch to the Swin or ViT repository ID to use another variant. No local checkpoint or tokenizer file is required.

For private or gated Hub repositories, run `uv run hf auth login` with an account that has access before loading.

## Data preparation

Each image must be a channel-first array with shape **`(C, H, W)`**. Within a panel, all images must have the same channel count and marker order. Panels can have different marker sets; `PanelBatchSampler` groups samples from the same panel into batches.

The training scripts expect both `train` and `test` splits, with `test` used for validation during training:

```text
data/
├── train/
│   ├── panel_a/
│   │   └── imgs/
│   │       └── image_001.npy
│   └── panel_b/
│       └── imgs/
│           └── image_002.npy
└── test/
    ├── panel_a/
    │   └── imgs/
    │       └── image_003.npy
    └── panel_b/
        └── imgs/
            └── image_004.npy
```

The loader reads `.npy` with NumPy or `.tiff` with tifffile. Set `data_config.file_extension` to `npy` or `tiff` (without a leading dot).

Replace the dataset-specific contents of [configs/all_panels_config.yaml](./configs/all_panels_config.yaml), or create your own panel configuration and point `panel_config` to it. A minimal example is:

```yaml
paths:
  train: /absolute/path/to/data/train
  test: /absolute/path/to/data/test
datasets:
  - panel_a
  - panel_b
markers:
  panel_a: [CD3, CD20]
  panel_b: [CD3, CD4, CD20]
```

- Each listed panel must have an `imgs/` subdirectory under both split paths.
- Marker names must exactly match [configs/markers_tokenizer.yaml](./configs/markers_tokenizer.yaml), including capitalization and spacing.
- Use the [marker synonyms JSON](./configs/markers_with_synonyms.json) to help map your marker names to tokenizer vocabulary names. Each entry contains a `marker` name and its `synonyms`; review candidate matches against your marker and antibody metadata, then use the exact tokenizer name in your panel configuration without changing channel order. Synonyms are a reference only and are not resolved automatically by the loader.
- Tokenizer values are integer embedding indices, not channel positions within an image. For a new model, use unique contiguous IDs starting at zero. For a checkpoint, preserve its original marker-to-ID mapping.
- Unknown markers raise an error by default. `data_config.unsupported_marker_behavior: drop` removes them and their image channels; fine-tuning is required to learn new markers.

The default preprocessing pipeline applies transforms, arcsinh, Butterworth denoising, and percentile scaling. Review `data_config` for your inputs, especially if they have already been transformed or normalized. Match inference preprocessing to the model's training setup.

## Training

Start with [train_masked_config.yaml](./train_masked_config.yaml). Before launching:

1. Set `panel_config` and `tokenizer_config` to your files.
2. Review the preprocessing, image size, batch size, and masking settings.
3. Keep `device: cuda` for the provided training workflow and `num_workers` greater than zero (the loaders use persistent workers).
4. Replace the example `comet_project` and `tags` values. Set a `run_name` if you want predictable checkpoint names.

Training uses [Comet](https://www.comet.com/) for metrics and validation images. Prefer environment variables for credentials:

```bash
export COMET_API_KEY="your-api-key"
export COMET_WORKSPACE="your-workspace"
uv run python train_masked_model.py train_masked_config.yaml
```

Leave `comet_api_key` and `comet_workspace` as `null` in YAML when using environment variables. Comet logging includes validation reconstructions.

For a pip installation, use `python train_masked_model.py train_masked_config.yaml` from the activated environment.

### Configuration reference


| Settings                                                                                          | Purpose                                                                |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `encoder`, `decoder`                                                                              | Architecture, embedding widths, block counts, and hyperkernel settings |
| `panel_config`, `tokenizer_config`                                                                | YAML paths or inline mappings for panels and marker IDs                |
| `input_image_size`, `batch_size`, `num_workers`, `data_config`                                    | Cropping, batching, file format, and preprocessing                     |
| `lr`, `final_lr`, `weight_decay`, `epochs`, `gradient_accumulation_steps`, `frac_warmup_steps`    | AdamW optimization and warmup/cosine scheduling                        |
| `min_channels_frac`, `fully_masked_channels_max_frac`, `spatial_masking_ratio`, `mask_patch_size` | Channel subset sampling, whole-channel masking, and spatial masking    |
| `beta`                                                                                            | Weighting parameter for the beta-NLL reconstruction loss               |
| `checkpoints_dir`, `from_checkpoint`, `save_checkpoint_freq`                                      | Checkpoint saving and training resumption                              |
| `comet_project`, `comet_workspace`, `comet_api_key`, `tags`, `run_name`                           | Experiment logging and naming                                          |

The encoder's `ma_layers_blocks` / `ma_embedding_dims` configure the marker-agnostic stage; `pm_layers_blocks` / `pm_embedding_dims` configure the pan-marker stage. The decoder uses `decoded_embed_dim` and `num_blocks`. Both stages accept `hyperkernel` settings (`kernel_size`, `stride`, `padding`, `use_bias`).

Encoder and decoder block types are resolved through registries. Alternative training examples are available in [configs/train_swin_config.yaml](./configs/train_swin_config.yaml) and [configs/train_vit_config.yaml](./configs/train_vit_config.yaml); review their data paths and experiment settings as well.

### Checkpoints and resuming

By default, training writes to `checkpoints/`:

- `checkpoint-<run_name>-epoch_<epoch>.pth`: periodic checkpoints (the epoch suffix is zero-based).
- `last_checkpoint-<run_name>.pth`: the most recent epoch, including optimizer and scheduler state.
- `final_model-<run_name>.pth`: final weights and architecture configuration, without optimizer or scheduler state.

To resume training, set `from_checkpoint` to an explicit periodic/latest checkpoint path. Alternatively, set it to `last` and reuse the original `run_name` and `checkpoints_dir`; this looks for that run's `last_checkpoint-`. If it is absent, training starts from scratch. Use a checkpoint with optimizer and scheduler state for resuming, rather than the final-model file.

## Fine-tuning on new panels or markers

Use [finetuning_configs/finetune_masked_config.yaml](./finetuning_configs/finetune_masked_config.yaml) and [finetuning_configs/new_panels_config.yaml](./finetuning_configs/new_panels_config.yaml) as templates:

1. Set `pretrained_checkpoint` to your trained checkpoint.
2. Keep `tokenizer_config` pointing to the tokenizer used by that checkpoint.
3. Set `panel_config` to the new panels, with their exact channel order.
4. Review the architecture, training/preprocessing settings, and Comet values; replace all `...` placeholders.
5. Choose a `freeze_mode`:
   - `freeze_all_but_new_markers`: update new marker embedding rows only; requires at least one new marker.
   - `freeze_marker_agnostic`: freeze the marker-agnostic encoder and shared decoder blocks.
   - `none`: train all parameters.

```bash
uv run python finetune_masked_model.py finetuning_configs/finetune_masked_config.yaml
```

The script extends the tokenizer while preserving existing IDs and saves it to `tokenizer-<run_name>.yaml` in `checkpoints_dir`, unless `updated_tokenizer_path` is supplied. Keep this tokenizer with the fine-tuned checkpoint and use it for subsequent inference. This workflow initializes from `pretrained_checkpoint`.

## Python API

### Load from Hugging Face

Choose any repository ID from the [model-selection table](#choose-a-model) to load that model and the shared marker tokenizer automatically:

```python
from multiplex_model.modules import MultiplexAutoencoder

repo_id = "SzczurekLab/ImmuVis_Conv"
model = MultiplexAutoencoder.from_pretrained(repo_id)
tokenizer = model.marker_tokenizer
```

Use `SzczurekLab/ImmuVis_Swin` or `SzczurekLab/ImmuVis_ViT` instead to load the corresponding transformer variant.

This uses the existing checkpoint loader and returns a CPU model in evaluation mode. Pass `map_location="cuda"` to place it on a GPU. Both required files are fetched from the same Hub snapshot; pass `revision="<commit-hash>"` for reproducibility, `cache_dir="/path/to/cache"` for a custom cache, or `local_files_only=True` for cached/offline loading.

The bundle contains `model.pth` and `markers_tokenizer.yaml` at its root. `from_pretrained("/path/to/local/bundle")` also works without contacting the Hub.

This is a native ImmuVis/PyTorch integration, not a Transformers `AutoModel` or `AutoTokenizer`. The training and fine-tuning scripts accept local checkpoint and tokenizer paths.

### Load a local checkpoint

Load a compatible local checkpoint. The original variants are named `ImmuVis_Conv.pth`, `ImmuVis_Swin.pth`, and `ImmuVis_ViT.pth`; choose the corresponding file:

```python
from multiplex_model.modules import MultiplexAutoencoder

model = MultiplexAutoencoder.load_from_checkpoint(
    "checkpoints/ImmuVis_Conv.pth",
    map_location="cpu",
)
model.eval()
```

For inference, prepare preprocessed float tensors and marker-ID tensors on the same device as the model:


| API                                       | Inputs                                                                                   | Returned value                                                                          |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `model(images, encoded_ids, decoded_ids)` | Images`(B, C_in, H, W)`; input marker IDs `(B, C_in)`; requested output IDs `(B, C_out)` | Dictionary with`output` shaped `(B, C_out, H, W, 2)` for the default two-output decoder |
| `model.encode(images, encoded_ids)`       | Images and their marker IDs                                                              | Dictionary with`output` containing the spatial latent representation                    |
| `model.decode(latents, decoded_ids)`      | Latent representation and requested marker IDs                                           | Raw decoder output tensor                                                               |

Input and output IDs need not describe the same panel, but must be supported by the checkpoint's tokenizer. The two default output values are raw mean logits and log variance: apply `sigmoid` to the mean logits to obtain reconstructed intensities in `[0, 1]`.

See the [virtual staining tutorial](./notebooks/tutorial_virtual_staining.ipynb) for preprocessing, prediction, and visualization examples, and the [cell typing tutorial](./notebooks/tutorial_celltyping.ipynb) for embedding-based analysis.

## Repository layout


| Path                                                   | Contents                                                                                  |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| [multiplex_model/](./multiplex_model/)                 | Model architectures, registries, data loaders, transforms, losses, and training utilities |
| [train_masked_model.py](./train_masked_model.py)       | Multi-panel masked-autoencoder training                                                   |
| [finetune_masked_model.py](./finetune_masked_model.py) | Fine-tuning with tokenizer extension and freezing strategies                              |
| [configs/](./configs/)                                 | Panel/tokenizer metadata and alternative architecture configurations                      |
| [finetuning_configs/](./finetuning_configs/)           | Fine-tuning templates                                                                     |
| [notebooks/](./notebooks/)                             | Virtual staining and cell-typing tutorials                                                |
| [pyproject.toml](./pyproject.toml)                     | Package metadata, dependencies, optional extras, and development-tool settings            |

## Troubleshooting

- **CUDA unavailable:** check `uv run python -c "import torch; print(torch.cuda.is_available())"`. Consult the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for driver/platform compatibility; keep Torch and torchvision versions compatible.
- **Unsupported markers or embedding errors:** check spelling, channel order, and that the tokenizer belongs to the checkpoint. Do not renumber IDs for an existing model.
- **No images found:** check the split paths, panel names, `imgs/` subdirectories, and configured file extension.
- **Crop or validation errors:** images must be large enough for the configured crops. The training validation routine samples four batches for visualization, so provide at least four validation batches.
- **Notebook import errors:** select the registered ImmuVis kernel and verify that the project was installed in that environment.

## License

This project is licensed under the [MIT License](./LICENSE).
