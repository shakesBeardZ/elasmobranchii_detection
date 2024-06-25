# elasmobranchii Detection


## Allocating debug node

```
srun -n 1 --mem=128GB -t 4:00:00 --gres=gpu:a100:2 --pty /bin/bash
```


## Running the code

```
python train.py --gpus 4 --workers 6
```


## Running the code as Slurm Job using the run_job.sh

```
sbatch run_job.sh
```
