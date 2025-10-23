python -m sir.training.train_policy \
        --repo-ids "ankile/square-v1,ankile/square-dagger-v1" \
        --policy diffusion \
        --save-video \
        --env NutAssemblySquare \
        --max-steps 600 \
        --batch-size 128 \
        --use-wandb