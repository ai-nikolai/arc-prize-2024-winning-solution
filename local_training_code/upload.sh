ENDING="lora"
ENDING="merged"

N=128
E=141

MODEL_PATH="Mistral-NeMo-Minitron-8B-Base-IKCL-aug-True-tp-1-rt-1-shfl-1-perm-1-n-$N-e-$E-b-2-a-1-$ENDING"
REPO_NAME="arc24/mini-8b-aug-$N-$E-$ENDING"

echo $MODEL_PATH
echo $REPO_NAME

huggingface-cli upload $REPO_NAME pretrained_models/$MODEL_PATH --private
