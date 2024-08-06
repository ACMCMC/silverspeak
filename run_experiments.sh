# Check that the MY_HUGGINGFACE_USER environment variable is set
if [ -z "$MY_HUGGINGFACE_USER" ]; then
    export MY_HUGGINGFACE_USER=$(huggingface-cli whoami)
    echo "MY_HUGGINGFACE_USER is not set. Setting it to $MY_HUGGINGFACE_USER"
fi

python experiments/compute_metrics.py --model arguGPT --dataset reuter 
python experiments/compute_metrics.py --model arguGPT --dataset writing_prompts 
python experiments/compute_metrics.py --model arguGPT --dataset essay 
python experiments/compute_metrics.py --model arguGPT --dataset cheat 
python experiments/compute_metrics.py --model binoculars --dataset reuter 
python experiments/compute_metrics.py --model binoculars --dataset writing_prompts 
python experiments/compute_metrics.py --model binoculars --dataset essay 
python experiments/compute_metrics.py --model binoculars --dataset cheat 
python experiments/compute_metrics.py --model detectGPT --dataset reuter 
python experiments/compute_metrics.py --model detectGPT --dataset writing_prompts 
python experiments/compute_metrics.py --model detectGPT --dataset essay 
python experiments/compute_metrics.py --model detectGPT --dataset cheat 
python experiments/compute_metrics.py --model fastDetectGPT --dataset reuter 
python experiments/compute_metrics.py --model fastDetectGPT --dataset writing_prompts 
python experiments/compute_metrics.py --model fastDetectGPT --dataset essay 
python experiments/compute_metrics.py --model fastDetectGPT --dataset cheat 
python experiments/compute_metrics.py --model ghostbusterAPI --dataset reuter 
python experiments/compute_metrics.py --model ghostbusterAPI --dataset writing_prompts 
python experiments/compute_metrics.py --model ghostbusterAPI --dataset essay 
python experiments/compute_metrics.py --model ghostbusterAPI --dataset cheat 
python experiments/compute_metrics.py --model openAIDetector --dataset reuter 
python experiments/compute_metrics.py --model openAIDetector --dataset writing_prompts 
python experiments/compute_metrics.py --model openAIDetector --dataset essay 
python experiments/compute_metrics.py --model openAIDetector --dataset cheat 
python experiments/compute_metrics.py --model watermark --dataset realnewslike