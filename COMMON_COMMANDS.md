Teleop stuff

```
mjpython -m sir.teleoperation.robosuite_teleop \
      --env NutAssemblySquare \
      --robot Panda \
      --cameras "agentview,robot0_eye_in_hand" \
      --save-data \
      --dataset-name square-v1
```

Dataset visualization

```
pip install -e .[viewer]
sir-dataset-viewer --dataset data/square-dagger-v1 --open-browser
```

Legacy Flask viewer

```
pip install -e .[viewer]
sir-legacy-dataset-viewer --repo-id ankile/square-dagger-v1
```
