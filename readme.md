CUDA_VISIBLE_DEVICES=1 python qwen_vl_finetune_refactored.py

朱赫那边也是用VERL的跑这个带有图片的训练

暂时是unsloth还有一个环境问题导致VL-SFT跑不起来-好像这个更新了版本没有了
RuntimeError: Direct module loading failed for Linear_peft_forward: name 'Any' is not defined

deepspeed llm-sft没跑起来-教程是从happy-llm 第七章来的

unsloth llm-SFT和GRPO已完成,可以跑起来，并且速度还不错，就是只能是单卡。 这个是self-llm的教程
估计是升级pytorch带来的问题。



verl-跑不起来：
flash-attn 版本号太高
ImportError: /home/zly/miniconda3/envs/py312/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs

ERROR: Could not find a version that satisfies the requirement flash-attn==2.7.4.post1 (from versions: none)
ERROR: No matching distribution found for flash-attn==2.7.4.post1

pip install flash-attn==2.7.4.post1 --no-build-isolation

[flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl)


