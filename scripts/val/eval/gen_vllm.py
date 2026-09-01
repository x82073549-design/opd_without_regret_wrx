import os
import json
import hashlib
import random
import re
import argparse
import concurrent.futures
import multiprocessing  # Added for spawn-based worker management
import gc  # Added for explicit resource cleanup
from collections import Counter
import torch  # Added for CUDA cache cleanup
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
# Try to import distributed cleanup helpers for releasing GPU memory.
try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    destroy_model_parallel = None
try:
    from vllm.distributed.parallel_state import destroy_distributed_environment
except ImportError:
    destroy_distributed_environment = None

# --------------------------------------------------------------------------- #
#                   Global constants / variables                              #
# --------------------------------------------------------------------------- #
DATA_DIR = "../data"
# MODEL_FOLDER = "../../model/Qwen3-1.7B-SFT-DAPO-4B-filtered"

def extract_max_number(path):
    """Extract all numbers from a path and return the largest one for sorting."""
    numbers = re.findall(r'\d+', path)
    if numbers:
        return max(int(n) for n in numbers)
    return -1  # If there is no number, keep this entry at the end.

# Optional legacy folder scan. Prefer passing --model for explicit evaluation.
MODEL_FOLDER = os.environ.get("MODEL_FOLDER")
if MODEL_FOLDER:
    try:
        model_files = os.listdir(MODEL_FOLDER)
        MODEL_NAMES_CANDIDATES = [os.path.join(MODEL_FOLDER, f) for f in model_files]
        MODEL_NAMES_CANDIDATES.sort(key=extract_max_number, reverse=True)
    except FileNotFoundError:
        MODEL_NAMES_CANDIDATES = []
else:
    MODEL_NAMES_CANDIDATES = []

# Active model list.
MODEL_NAMES = MODEL_NAMES_CANDIDATES or ["../../model/Qwen3-4B"]

TASKS = [
    {"name": "AIME24", "path": f"{DATA_DIR}/AIME24/test.parquet", "N": 16},
    {"name": "AIME25", "path": f"{DATA_DIR}/AIME25/test.parquet", "N": 16},
    {"name": "AMC23", "path": f"{DATA_DIR}/AMC23/test.parquet", "N": 16},
]

ANSWER_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
PROMPT_TEMPLATE = """{problem} {instruction}"""
MAX_TOKENS  = 31744
TEMPERATURE = 0.7
TOP_P       = 0.95

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
    """Read parquet file and return a list of prompts (no duplication)."""
    df = pd.read_parquet(filepath)
    if "BRUMO25" in filepath or "CMIMC25" in filepath or "HMMT25" in filepath or "HMMT24" in filepath:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "problem"].strip(),
                "answer": df.at[i, "answer"].strip(),
            }
            for i in range(len(df))
        ]
    else:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "prompt"][0]["content"].strip(),
                "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
            }
            for i in range(len(df))
        ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def split_rollout_ids(rollout_ids: list[int], num_workers: int):
    """Round-robin split of rollout IDs into num_workers chunks."""
    chunks = [[] for _ in range(num_workers)]
    for idx, rollout_id in enumerate(rollout_ids):
        chunks[idx % num_workers].append(rollout_id)
    return chunks


def make_request_seed(example_id: int, evaluation_seed: int) -> int:
    """Derive a stable per-question request seed shared across evaluated models."""
    payload = f"{example_id}:{evaluation_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def ensure_answer_instruction(prompt: str) -> str:
    """Append the answer-format instruction exactly once."""
    prompt = prompt.strip()
    if ANSWER_INSTRUCTION in prompt:
        return prompt
    return PROMPT_TEMPLATE.format(problem=prompt, instruction=ANSWER_INSTRUCTION)


# --------------------------------------------------------------------------- #
#              Worker process (one model instance per GPU worker)              #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
    """
    Each worker runs on a single GPU:
    args_tuple = (
        model_name,
        samples,
        evaluation_seed_list,
        gpu_id,
        enable_thinking,
        max_tokens,
        temperature,
        top_p,
    )
    gpu_id: values such as "0" or "3", used for CUDA_VISIBLE_DEVICES
    """
    (
        model_name,
        samples,
        evaluation_seed_list,
        gpu_id,
        enable_thinking,
        max_tokens,
        temperature,
        top_p,
    ) = args_tuple
    
    # CUDA_VISIBLE_DEVICES must be set inside the spawned process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    results = []
    llm = None

    try:
        thinking_description = "model-default" if enable_thinking is None else str(enable_thinking)
        print(
            f"[GPU {gpu_id}] | Model: {model_name} | seeds len={len(evaluation_seed_list)} "
            f"| loading model (TP=1, enable_thinking={thinking_description}, "
            f"max_tokens={max_tokens})...",
            flush=True,
        )
        
        # Initialize a single-GPU, single-instance LLM.
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
        )
        
        # Get the tokenizer. vLLM already stops on the tokenizer/model EOS token.
        # Do not derive stop token IDs from hard-coded strings: a string that is
        # not special for this tokenizer may encode to several ordinary tokens.
        try:
            tokenizer = llm.get_tokenizer()
            print(
                f"[GPU {gpu_id}] tokenizer EOS: token={tokenizer.eos_token!r}, "
                f"id={tokenizer.eos_token_id}; custom stop_token_ids disabled",
                flush=True,
            )
        except Exception as e:
            tokenizer = None
            print(f"[GPU {gpu_id}] Warning: Could not get tokenizer: {e}", flush=True)
        
        for evaluation_seed in evaluation_seed_list:
            if tokenizer is None:
                raise RuntimeError("Tokenizer is required for apply_chat_template, but it could not be loaded.")

            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                chat_template_kwargs["enable_thinking"] = enable_thinking
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": s["prompt"]}],
                    **chat_template_kwargs,
                )
                for s in samples
            ]
            sampling_params = [
                SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=make_request_seed(sample["example_id"], evaluation_seed),
                )
                for sample in samples
            ]
            
            # Disable per-worker tqdm output to keep multi-process logs readable.
            outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=False)
            
            for sample, out in zip(samples, outputs):
                completion = out.outputs[0]
                stop_reason = completion.stop_reason
                if stop_reason is not None and not isinstance(stop_reason, (str, int, float, bool)):
                    stop_reason = str(stop_reason)
                results.append(
                    {
                        "example_id": sample["example_id"],
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "evaluation_seed": evaluation_seed,
                        "request_seed": make_request_seed(sample["example_id"], evaluation_seed),
                        # Retain the legacy key for downstream readers.
                        "seed": evaluation_seed,
                        "response": completion.text,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": stop_reason,
                        "response_token_count": len(completion.token_ids or []),
                        "prompt_token_count": len(out.prompt_token_ids or []),
                    }
                )
    
    except Exception as e:
        raise RuntimeError(f"[GPU {gpu_id}] evaluation worker failed: {e}") from e
    
    finally:
        # Explicitly release vLLM resources.
        # This helps prevent CUDA context deadlocks and zombie processes.
        print(f"[GPU {gpu_id}] Cleaning up resources...", flush=True)
        if llm is not None:
            del llm
        
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass

        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[GPU {gpu_id}] Cleanup done.", flush=True)

    return results


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def parse_tasks(task_specs, default_n):
    tasks = []
    for spec in task_specs:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Bad task spec: {spec}. Expected NAME:PATH[:N].")
        name, task_path = parts[0], parts[1]
        n = int(parts[2]) if len(parts) == 3 else default_n
        tasks.append({"name": name, "path": task_path, "N": n})
    return tasks


def load_task_manifest(manifest_path: str, default_n: int) -> list[dict]:
    """Load evaluation tasks from a JSON manifest."""
    path = Path(manifest_path).resolve()
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if isinstance(manifest, dict) and manifest.get("purpose") not in (None, "validation"):
        raise ValueError("Task manifest purpose must be 'validation'.")
    entries = manifest.get("datasets") if isinstance(manifest, dict) else manifest
    if not isinstance(entries, list) or not entries:
        raise ValueError("Task manifest must contain a non-empty 'datasets' list.")

    tasks = []
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry or "path" not in entry:
            raise ValueError("Each task manifest entry must contain 'name' and 'path'.")
        task_path = Path(entry["path"])
        if not task_path.is_absolute():
            task_path = path.parent / task_path
        task_path = task_path.resolve()
        if not task_path.is_file():
            raise FileNotFoundError(f"Validation dataset does not exist: {task_path}")
        expected_sha256 = entry.get("sha256")
        if expected_sha256 is not None:
            actual_sha256 = sha256_file(str(task_path))
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Validation dataset hash mismatch for {task_path}: "
                    f"expected {expected_sha256}, got {actual_sha256}."
                )
        tasks.append(
            {
                "name": str(entry["name"]),
                "path": str(task_path),
                "N": int(entry.get("n", default_n)),
                "expected_num_questions": entry.get("num_questions"),
            }
        )
    return tasks


def parse_seed_list(seed_spec: Optional[str], default_n: int) -> list[int]:
    """Parse a comma-separated evaluation seed list, or default to range(default_n)."""
    if seed_spec is None:
        return list(range(default_n))
    seeds = [int(value.strip()) for value in seed_spec.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must not contain duplicate values.")
    return seeds


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation rollouts with vLLM.")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking when applying the chat template.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking when applying the chat template.",
    )
    parser.add_argument("--model", action="append", dest="models", help="HF model path. Can be passed multiple times.")
    parser.add_argument("--task", action="append", dest="tasks", help="Task spec NAME:PARQUET[:N]. Can be passed multiple times.")
    parser.add_argument("--task-manifest", default=None, help="JSON manifest containing a datasets list.")
    parser.add_argument("--out-dir", default="justrl_eval_outputs", help="Directory for generated jsonl files.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids, one vLLM worker per GPU.")
    parser.add_argument("--n", type=int, default=16, help="Default number of rollouts per problem.")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated evaluation seeds. The first task N seeds are used for each task.",
    )
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--replace", action="store_true", help="Overwrite existing generation files.")
    parser.set_defaults(enable_thinking=None)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")

    model_names = args.models or MODEL_NAMES
    if args.task_manifest and args.tasks:
        raise ValueError("Use either --task-manifest or --task, not both.")
    if args.task_manifest:
        tasks = load_task_manifest(args.task_manifest, args.n)
    else:
        tasks = parse_tasks(args.tasks, args.n) if args.tasks else TASKS
    evaluation_seeds = parse_seed_list(args.seeds, args.n)
    gpu_workers = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_workers:
        raise ValueError("--gpus must contain at least one GPU id.")

    num_workers = len(gpu_workers)

    print(f"GPU workers (one model per GPU): {gpu_workers}")
    thinking_description = "model-default" if args.enable_thinking is None else str(args.enable_thinking)
    print(f"apply_chat_template enable_thinking={thinking_description}")

    for model_name in model_names:
        print(f"\n{'='*50}\nStarting evaluation for model: {model_name}\n{'='*50}")
        
        OUT_DIR = Path(args.out_dir) / model_name.rstrip("/").split("/")[-1]
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            task_name = task["name"]
            task_path = task["path"]
            N = task["N"]
            if N > len(evaluation_seeds):
                raise ValueError(
                    f"Task {task_name} requests N={N}, but only {len(evaluation_seeds)} evaluation seeds were provided."
                )
            task_seeds = evaluation_seeds[:N]

            print(f"Starting evaluation for task: {task_name} (N={N})")
            
            seed_spec = ",".join(str(seed) for seed in task_seeds)
            seed_hash = hashlib.sha256(seed_spec.encode("utf-8")).hexdigest()[:10]
            out_path = OUT_DIR / (
                f"{task_name.lower()}_t{args.temperature}_p{args.top_p}_n{N}"
                f"-MNT{args.max_tokens}-ES{seed_hash}.jsonl"
            )
            manifest_path = out_path.with_suffix(".manifest.json")

            # --- Repetition Check ---
            if not args.replace and out_path.exists():
                if not manifest_path.exists():
                    raise RuntimeError(
                        f"Result file exists without a generation manifest: {out_path}. "
                        "Use --replace after checking the stale output."
                    )
                with manifest_path.open(encoding="utf-8") as f:
                    existing_manifest = json.load(f)
                expected_existing_values = {
                    "model": str(Path(model_name).resolve()),
                    "task": task_name,
                    "task_path": str(Path(task_path).resolve()),
                    "task_sha256": sha256_file(task_path),
                    "evaluation_seeds": task_seeds,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "enable_thinking": args.enable_thinking,
                    "stop_policy": "model_eos",
                }
                mismatches = {
                    key: {"expected": value, "actual": existing_manifest.get(key)}
                    for key, value in expected_existing_values.items()
                    if existing_manifest.get(key) != value
                }
                if mismatches:
                    raise RuntimeError(
                        f"Existing evaluation output does not match the requested configuration: {mismatches}. "
                        "Use a new run ID or --replace after checking the output."
                    )
                existing_rows = []
                with out_path.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            existing_rows.append(json.loads(line))
                existing_pairs = {
                    (row["example_id"], row.get("evaluation_seed", row.get("seed")))
                    for row in existing_rows
                }
                expected_count = int(existing_manifest.get("expected_generations", -1))
                num_questions = int(existing_manifest.get("num_questions", -1))
                expected_pairs = {
                    (example_id, evaluation_seed)
                    for example_id in range(num_questions)
                    for evaluation_seed in task_seeds
                }
                if (
                    expected_count != num_questions * len(task_seeds)
                    or len(existing_rows) != expected_count
                    or existing_pairs != expected_pairs
                ):
                    raise RuntimeError(
                        f"Existing evaluation output is incomplete: expected {expected_count} rows, "
                        f"got {len(existing_rows)} rows and {len(existing_pairs)} unique pairs; "
                        "the exact question/seed pairs must match the manifest."
                    )
                print(f"Result file already exists at '{out_path}'. Skipping.")
                continue  # Skip to the next task

            # 1. Load original prompts
            samples = load_samples(task_path)
            expected_num_questions = task.get("expected_num_questions")
            if expected_num_questions is not None and len(samples) != int(expected_num_questions):
                raise ValueError(
                    f"Validation dataset size mismatch for {task_name}: "
                    f"expected {expected_num_questions}, got {len(samples)}."
                )

            # Append the suffix only when the dataset prompt does not already
            # contain it. The standard OPD parquet prompts are preformatted.
            for sample in samples:
                sample["prompt"] = ensure_answer_instruction(sample["prompt"])

            if len(samples) > 0:
                print("Example prompt after formatting:")
                print(samples[0]["prompt"])
            
            # 2. Split explicit evaluation seeds across GPUs.
            seed_chunks = split_rollout_ids(task_seeds, num_workers)

            # 3. Launch workers, with each worker using one GPU.
            all_results = []
            args_list = [
                (
                    model_name,
                    samples,
                    seed_chunks[i],
                    gpu_workers[i],
                    args.enable_thinking,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                )
                for i in range(num_workers)
                if seed_chunks[i]
            ]
            
            # Use the spawn start method for worker processes.
            ctx = multiprocessing.get_context("spawn")
            
            worker_failures = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(args_list), mp_context=ctx) as ex:
                futures = [ex.submit(worker_process, tup) for tup in args_list]
                
                # Track overall progress with tqdm.
                for fut in tqdm(concurrent.futures.as_completed(futures),
                                total=len(futures), desc=f"GPU workers ({task_name})"):
                    try:
                        res = fut.result()
                        all_results.extend(res)
                    except Exception as e:
                        worker_failures.append(str(e))

            if worker_failures:
                raise RuntimeError("Evaluation worker failures:\n" + "\n".join(worker_failures))

            print(f"Total generations collected for {task_name}: {len(all_results)}")

            expected_count = len(samples) * len(task_seeds)
            observed_pairs = [(item["example_id"], item["evaluation_seed"]) for item in all_results]
            if len(all_results) != expected_count or len(set(observed_pairs)) != expected_count:
                raise RuntimeError(
                    f"Incomplete evaluation for {task_name}: expected {expected_count} unique "
                    f"(example_id, evaluation_seed) pairs, got {len(all_results)} rows and "
                    f"{len(set(observed_pairs))} unique pairs."
                )

            all_results.sort(key=lambda item: (item["example_id"], item["evaluation_seed"]))
            finish_reason_counts = Counter(str(item["finish_reason"]) for item in all_results)
            stop_reason_counts = Counter(str(item["stop_reason"]) for item in all_results)
            print(f"Finish reasons for {task_name}: {dict(finish_reason_counts)}")
            print(f"Stop reasons for {task_name}: {dict(stop_reason_counts)}")

            # 4. Save to disk
            if all_results:
                with out_path.open("w", encoding="utf-8") as f:
                    for item in all_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                manifest = {
                    "model": str(Path(model_name).resolve()),
                    "task": task_name,
                    "task_path": str(Path(task_path).resolve()),
                    "task_sha256": sha256_file(task_path),
                    "num_questions": len(samples),
                    "evaluation_seeds": task_seeds,
                    "request_seed_scheme": "sha256(example_id:evaluation_seed) first 31 bits",
                    "expected_generations": expected_count,
                    "actual_generations": len(all_results),
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "enable_thinking": args.enable_thinking,
                    "stop_policy": "model_eos",
                    "finish_reason_counts": dict(finish_reason_counts),
                    "stop_reason_counts": dict(stop_reason_counts),
                    "generation_metadata_fields": [
                        "finish_reason",
                        "stop_reason",
                        "response_token_count",
                        "prompt_token_count",
                    ],
                    "output_file": str(out_path.resolve()),
                }
                with manifest_path.open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                print(f"Saved results for {task_name} to {out_path}")
            else:
                print(f"No results collected for {task_name} (Check for errors).")


if __name__ == "__main__":
    # Set the start method to avoid multiprocessing issues in some environments.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
