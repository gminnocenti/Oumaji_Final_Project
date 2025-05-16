import os
import yaml

ROOT = "mlruns"  # adjust if your folder is named differently
for exp in os.listdir(ROOT):
    exp_dir = os.path.join(ROOT, exp)
    if not os.path.isdir(exp_dir):
        continue
    for run in os.listdir(exp_dir):
        run_dir = os.path.join(exp_dir, run)
        meta_path = os.path.join(run_dir, "meta.yaml")
        if not os.path.exists(meta_path):
            continue

        data = yaml.safe_load(open(meta_path))
        if "run_uuid" not in data and "run_id" in data:
            # Copy the old run_id value into run_uuid
            data["run_uuid"] = data["run_id"]
            with open(meta_path, "w") as f:
                yaml.dump(data, f)
            print(f"Fixed {meta_path}")
