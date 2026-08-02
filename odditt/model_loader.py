"""Model resolution/loading -- extracted from the notebook's Section 4 cell.

Same behavior as the notebook: for any model_id, checks (in order) an attached Kaggle Dataset,
a Google Drive folder (Colab), local disk, and only downloads from the Hugging Face Hub as a last
resort -- saving the download into the right spot for the environment so it isn't repeated next
run. The notebook ran this at cell-execution time with module-level side effects; here it's
wrapped into functions so it can be imported and called explicitly (e.g. once from app.py, or
repeatedly from evals/ to load a second model for comparison).
"""
import glob
import importlib.util
import os
import shutil

from huggingface_hub import snapshot_download
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline as hf_pipeline

KAGGLE_INPUT_DIR = "/kaggle/input"
KAGGLE_WORKING_DIR = "/kaggle/working"
COLAB_DRIVE_MODELS_DIR = "/content/drive/MyDrive/models"
LOCAL_MODELS_DIR = "models"

# Set this to your attached Kaggle Dataset's exact name if auto-detection below ever picks the
# wrong one (e.g. you have more than one dataset attached, or one dataset holds both models under
# subfolders). Leave as None to auto-detect. Auto-detection matches by model slug (e.g. a dataset
# folder or subfolder named "Mistral-7B-Instruct-v0.3" or "all-MiniLM-L6-v2").
KAGGLE_DATASET_NAME = None


def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def _is_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def _is_kaggle() -> bool:
    # A bare /kaggle/working or /kaggle/input path existing isn't a reliable signal on its own --
    # a Colab (or local) session can end up with a stray /kaggle directory too (e.g. from an
    # earlier `kaggle datasets download` or a manually created folder), which previously caused
    # this function to misfire as True on a genuine Colab run and route a model download to
    # /kaggle/working instead of Drive. google.colab being importable is a much stronger, harder
    # to spoof signal, so it's checked first and short-circuits this to False when it's True.
    if _is_colab():
        return False
    return os.path.isdir(KAGGLE_INPUT_DIR) or os.path.isdir(KAGGLE_WORKING_DIR)


def _has_model_files(path: str) -> bool:
    # A folder counts as "has this model" if it (or a subfolder) contains either config.json
    # (transformers-style models, e.g. the LLM) or config_sentence_transformers.json
    # (sentence-transformers-style models, e.g. the embedding model).
    if not os.path.isdir(path):
        return False
    for marker in ("config.json", "config_sentence_transformers.json"):
        if glob.glob(os.path.join(path, "**", marker), recursive=True):
            return True
    return False


def _find_kaggle_input_model(slug: str):
    if not os.path.isdir(KAGGLE_INPUT_DIR):
        return None
    candidates = []
    if KAGGLE_DATASET_NAME:
        candidates.append(os.path.join(KAGGLE_INPUT_DIR, KAGGLE_DATASET_NAME))
    candidates.extend(sorted(glob.glob(os.path.join(KAGGLE_INPUT_DIR, "*"))))
    for path in candidates:
        if not os.path.isdir(path):
            continue
        # Prefer a subfolder matching this model's slug (lets one dataset hold multiple models),
        # but fall back to any config file found anywhere under the dataset.
        slug_matches = glob.glob(os.path.join(path, "**", slug, "**", "config*.json"), recursive=True)
        if slug_matches:
            return os.path.dirname(slug_matches[0])
        any_matches = glob.glob(os.path.join(path, "**", "config*.json"), recursive=True)
        if any_matches:
            return os.path.dirname(any_matches[0])
    return None


def _mount_colab_drive() -> bool:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        return True
    except Exception as e:
        print(f"Could not mount Google Drive ({e}) -- will download instead.")
        return False


def resolve_model_dir(model_id: str):
    """Finds (or downloads) a model's files for this environment, for ANY model_id -- used for
    both the embedding model and the LLM. Returns (path_to_model_files, description)."""
    slug = _model_slug(model_id)

    if _is_kaggle():
        found = _find_kaggle_input_model(slug)
        if found:
            return found, "an attached Kaggle Dataset (no download needed)"

    if _is_colab():
        drive_dir = os.path.join(COLAB_DRIVE_MODELS_DIR, slug)
        if _mount_colab_drive() and _has_model_files(drive_dir):
            return drive_dir, "Google Drive (downloaded on a previous run)"

    local_dir = os.path.join(LOCAL_MODELS_DIR, slug)
    if _has_model_files(local_dir):
        return local_dir, "local disk (downloaded on a previous run)"

    # Nothing found anywhere -- download fresh, into the right spot for this environment so the
    # download is saved and reusable next time instead of repeated.
    if _is_kaggle():
        target_dir = os.path.join(KAGGLE_WORKING_DIR, slug)
        note = (f"downloaded fresh into {target_dir}. After ALL models finish loading: 'Save "
                f"Version' (include Output files), then create a new Kaggle Dataset from that "
                f"output -- that one Dataset will then hold every model this notebook downloaded. "
                f"You can also download that dataset's files afterward and upload them to Google "
                f"Drive if you want a copy there.")
    elif _is_colab():
        target_dir = os.path.join(COLAB_DRIVE_MODELS_DIR, slug)
        note = (f"downloaded fresh into your Google Drive at {target_dir} -- future Colab runs "
                f"will find it there automatically, and it's already backed up in your Drive.")
    else:
        target_dir = os.path.join(LOCAL_MODELS_DIR, slug)
        note = f"downloaded fresh into {target_dir} on local disk -- future runs here will find it automatically."

    # If a previous download attempt into this exact folder crashed partway through (e.g. ran
    # out of disk), a partial/incomplete copy may already be sitting here, silently eating space
    # that a retry would need. Clear it first so retries start clean instead of stacking on top
    # of a broken partial download.
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)

    free_gb = shutil.disk_usage(target_dir).free / (1024 ** 3)
    if free_gb < 18:
        print(f"WARNING: only {free_gb:.1f}GB free near {target_dir}. A 7B-class model's "
              f"safetensors alone are typically ~14-15GB -- this may not fit. If the download "
              f"fails with 'No space left on device', free up space first before rerunning.")

    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        # Many HF repos host the same weights in more than one format (safetensors + a duplicate
        # set of .bin/.pth files, sometimes ONNX/GGUF too). Downloading all of them roughly
        # doubles the disk footprint for no benefit here, since AutoModelForCausalLM only needs
        # the safetensors weights + config/tokenizer files.
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"],
        ignore_patterns=["*.bin", "*.pth", "*.msgpack", "*.h5", "*.gguf", "*.onnx", "original/*"],
    )
    return target_dir, note


def load_embeddings(config: dict) -> HuggingFaceEmbeddings:
    """Resolves and loads the embedding model per CONFIG["embedding_model"]."""
    embedding_dir, note = resolve_model_dir(config["embedding_model"])
    print(f"Loading embedding model from: {embedding_dir}\n({note})")
    return HuggingFaceEmbeddings(model_name=embedding_dir)


def load_llm(model_id: str, config: dict):
    """Resolves and loads any causal LM the same way (4-bit quant, device_map="auto"). Used for
    both the primary LLM (via config["llm_model"]) and, in evals/, a second model to compare
    against -- see evals/model_compare.py.

    Returns (model, tokenizer, gen_pipeline, chat_llm).
    """
    model_dir, note = resolve_model_dir(model_id)
    print(f"Loading LLM '{model_id}' from: {model_dir}\n({note})")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
    )

    # Note: trust_remote_code is deliberately OFF. Phi-4-mini's architecture (like Mistral's
    # before it) is natively built into the transformers library itself. Setting
    # trust_remote_code=True instead downloads and runs the model repo's own bundled modeling
    # code, which can drift out of sync with whatever transformers version is installed. If you
    # swap in a model that genuinely needs its own remote code (check the model card), you'll
    # need to set this to True and re-verify it loads cleanly.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
    )

    gen_pipeline = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config["max_new_tokens"],
        do_sample=config["do_sample"],
        return_full_text=False,
    )

    chat_llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen_pipeline))
    return model, tokenizer, gen_pipeline, chat_llm


def free_llm(*objs) -> None:
    """Best-effort GPU memory cleanup -- drop references and empty the CUDA cache. Useful when
    swapping models within one process (e.g. Phi -> Qwen in evals/model_compare.py)."""
    import gc
    import torch
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
