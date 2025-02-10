import os

dump_folder = "../../../BERT-for-RRC-ABSA_dependent_files"
domain = "laptop"
model_path = os.path.join(os.path.abspath(__file__), dump_folder, "pt_model")
review_data_path = os.path.join(dump_folder, "domain_corpus", domain, "data.npz")
print(review_data_path)
print(os.path.exists(review_data_path))
print(model_path)
print(os.path.exists(model_path))
print(__file__)
