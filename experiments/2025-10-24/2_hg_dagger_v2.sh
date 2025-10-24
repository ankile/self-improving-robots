mjpython -m sir.teleoperation.robosuite_dagger \
    --wandb-artifact "self-improving/square-dagger-comparison/act-square-v1-best-step-60000:v0" \
    --env NutAssemblySquare \
    --robot Panda \
    --cameras "agentview,robot0_eye_in_hand" \
    --dataset-name square-dagger-v2 \
    --wandb-project dagger-v2 \
    --auto-save-on-success