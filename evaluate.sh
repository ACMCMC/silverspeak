export DATASETS_VERBOSITY=info
export EVALUATE_VERBOSITY=info
export TRANSFORMERS_VERBOSITY=info

workon silverspeak

echo "Evaluating with $num_examples examples and $attack attack"

python -u evaluate.py --num_examples $num_examples --attack $attack --detection_system $detection_system
