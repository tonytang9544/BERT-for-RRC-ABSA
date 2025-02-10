import os
import sys
import subprocess
import shutil

def main(domain="laptop", dp=5, steps=10000, cuda_device="0"):
    if len(sys.argv) < 4:
        print("Usage: python script.py <domain> <dupe_factor> <steps> [cuda_device]")
        sys.exit(1)

    # domain = sys.argv[1]
    # dp = sys.argv[2]
    # steps = sys.argv[3]
    # cuda_device = sys.argv[4] if len(sys.argv) > 4 else None

    if cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        print(f"Using CUDA device {cuda_device}")

    BERT = "bert-base"
    
    ######################
    # modify this to your local folder for store large files.
    dump_folder = "../../../BERT-for-RRC-ABSA_dependent_files"


    # Generate review pretraining data
    # review_data_path = f"../domain_corpus/{domain}/data.npz"
    review_data_path = os.path.join(dump_folder, "domain_corpus", domain, "data.npz")
    # print(review_data_path)
    if not os.path.exists(review_data_path):
        subprocess.run([
            "python", "../src/gen_pt_review.py",
            "--input_file", f"../domain_corpus/raw/{domain}.txt",
            "--output_file", review_data_path,
            "--bert-model", BERT,
            "--max_seq_length=320",
            "--max_predictions_per_seq=40",
            "--masked_lm_prob=0.15",
            "--random_seed=12345",
            "--dupe_factor", dp
        ], stdout=open(f"../domain_corpus/{domain}/data.log", "w"), stderr=subprocess.STDOUT)

    # Generate SQuAD pretraining data
    squad_data_path = "../squad/data.npz"
    if not os.path.exists(squad_data_path):
        subprocess.run([
            "python", "../src/gen_pt_squad.py",
            "--input_dir", "../squad",
            "--output_dir", "../squad",
            "--bert-model", BERT,
            "--max_seq_length=320",
            "--seed=12345"
        ], stdout=open("../squad/data.log", "w"), stderr=subprocess.STDOUT)

    # Create output directory
    out_dir = f"../pt_model/{domain}_pt"
    os.makedirs(out_dir, exist_ok=True)

    # Run pretraining
    subprocess.run([
        "python", "../src/run_pt.py",
        "--bert_model", BERT,
        "--review_data_dir", f"../domain_corpus/{domain}",
        "--squad_data_dir", "../squad/",
        "--output_dir", out_dir,
        "--train_batch_size", "16",
        "--do_train",
        f"--num_train_steps={steps}",
        "--gradient_accumulation_steps=2",
        # "--fp16",
        "--loss_scale", "2",
        "--save_checkpoints_steps", "10000"
    ], stdout=open(f"{out_dir}/train.log", "w"), stderr=subprocess.STDOUT)

    # # Copy necessary files
    # for file in ["vocab.txt", "bert_config.json"]:
    #     src = f"../pt_model/{BERT}/{file}"
    #     dst = f"{out_dir}/{file}"
    #     print(src)
    #     if os.path.exists(src):
    #         subprocess.run(["cp", src, dst])


    for file in ["vocab.txt", "config.json"]:
        src = f"../pt_model/{BERT}/{file}"
        dst = f"{out_dir}/{file}"
        print(src)
        if os.path.exists(src):
            shutil.copy(src, dst)

if __name__ == "__main__":
    main()
