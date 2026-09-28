import subprocess

subprocess.run(
    [
        "python", "/content/train_scinup_crossencoder.py",
        "--dataset", "/content/data/SciNUP/dataset.jsonl",
        "--qrels", "/content/ground_truth_qrels.trec",
        "--paper-run", "/content/rrf_fused.trec",
        "--output", "/content/results/scinup_finetune_v1",
        "--epochs", "3",
        "--batch-size", "32",
        "--eval-batch-size", "128",
        "--negatives-per-positive", "2",
        "--max-length", "256",
    ],
    check=True,
)
