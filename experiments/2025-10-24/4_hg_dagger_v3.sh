mjpython -m sir.teleoperation.robosuite_dagger \
    --wandb-artifact "self-improving/square-dagger-comparison/act-square-v1_square-dagger-v2-best-step-15000:v0" \
    --env NutAssemblySquare \
    --robot Panda \
    --cameras "agentview,robot0_eye_in_hand" \
    --dataset-name square-dagger-v3 \
    --wandb-project dagger-v3 \
    --auto-save-on-success