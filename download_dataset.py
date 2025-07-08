# import os
# os.system("modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir ./data")
# # 解压预训练数据集
# os.system("tar -xvf ./data/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2")
# tar -xvf ./mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2
# 下载SFT数据集
# os.system(f'huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_3.5M_CN --local-dir BelleGroup')
import os
# hiyouga/geometry3k
# 下载数据集
os.system("huggingface-cli download --repo-type dataset --resume-download hiyouga/geometry3k --local-dir ./data/geometry3k")

# # 下载模型
# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
# os.system(f'huggingface-cli download --resume-download {model_name} --local-dir ./model_download/qwen2.5_vl_3b')