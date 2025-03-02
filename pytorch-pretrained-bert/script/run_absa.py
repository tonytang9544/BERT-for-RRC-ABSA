'''
translated by ChatGPT, not tested!
'''

import os
import sys
import subprocess

# Parameters passed as arguments
task = sys.argv[1]
bert = sys.argv[2]
domain = sys.argv[3]
run_dir = sys.argv[4]
runs = int(sys.argv[5])

# Optionally set CUDA_VISIBLE_DEVICES if passed as the 6th argument
if len(sys.argv) > 6:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[6]
    print(f"Using CUDA {os.environ['CUDA_VISIBLE_DEVICES']}")

# Set data directory path
data_dir = f"../data/{task}/{domain}"

# Loop over runs
for run in range(1, runs + 1):
    output_dir = f"../run/{run_dir}/{domain}/{run}"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if valid.json doesn't exist, then run training script
    if not os.path.exists(f"{output_dir}/valid.json"):
        with open(f"{output_dir}/train_log.txt", "w") as log_file:
            subprocess.run([
                "python", f"../src/run_{task}.py",
                "--bert_model", bert, "--do_train", "--do_valid",
                "--max_seq_length", "100", "--train_batch_size", "32", 
                "--learning_rate", "3e-5", "--num_train_epochs", "4",
                "--output_dir", output_dir, "--data_dir", data_dir, "--seed", str(run)
            ], stdout=log_file, stderr=subprocess.STDOUT)

    # Check if predictions.json doesn't exist, then run evaluation script
    if not os.path.exists(f"{output_dir}/predictions.json"):
        with open(f"{output_dir}/test_log.txt", "w") as log_file:
            subprocess.run([
                "python", f"../src/run_{task}.py",
                "--bert_model", bert, "--do_eval", "--max_seq_length", "100",
                "--output_dir", output_dir, "--data_dir", data_dir, "--seed", str(run)
            ], stdout=log_file, stderr=subprocess.STDOUT)

    # If predictions.json and model.pt exist, remove model.pt
    if os.path.exists(f"{output_dir}/predictions.json") and os.path.exists(f"{output_dir}/model.pt"):
        os.remove(f"{output_dir}/model.pt")
