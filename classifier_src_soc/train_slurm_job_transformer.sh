#!/bin/bash
#SBATCH --job-name=dgl_gnn_train
#SBATCH --partition=gpu --gres=gpu:h100-96:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/transformer_train_%j.out
#SBATCH --error=logs/transformer_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@comp.nus.edu.sg

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Initialize conda for bash shell if not already done
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda installation not found!"
    exit 1
fi

# Check if aml_project_new environment exists
if conda env list | grep -q "^aml_project_new"; then
    echo "Conda environment 'aml_project_new' already exists. Activating it..."
    conda activate aml_project_new
else
    echo "Creating new conda environment 'aml_project_new'..."
    conda env create -f $HOME/env_new.yml
    conda activate aml_project_new
    pip install --upgrade pip
    pip install kagglehub
    conda install -c dglteam/label/th21_cu121 dgl
fi

# # Set environment variables for optimal performance
# export CUDA_VISIBLE_DEVICES=0,1
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# # Print environment info
# echo "Environment Variables:"
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
# echo "=========================================="

# Navigate to working directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo "=========================================="

# Create output directory
OUTPUT_DIR="./saved_models/"
mkdir -p $OUTPUT_DIR

# Training parameters
NUM_EPOCHS=500
LEARNING_RATE=1e-4

# Run the training script
echo "Starting fine-tuning..."
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "=========================================="

python -u soc_transformer.py \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    2>&1 | tee $OUTPUT_DIR/training_log.txt

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Training failed!"
    echo "Check the error log for details."
    echo "=========================================="
    exit 1
fi

echo "Job completed at: $(date)"
echo "=========================================="