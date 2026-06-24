source .venv/bin/activate
export HF_HOME="/hpcwork/p0021834/workspace_patrick/hf_home"
echo "START LOADING"
vllm serve zai-org/GLM-5.2-FP8

