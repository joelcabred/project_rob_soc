To launch the Deep COACH code : 

```
python coach_mimo.py --config config_selftouch.yml --train_for 1000
```

To launch the Deep COACH +DDPG code :
```
python ddpg_coach_mimo.py --config config_selftouch.yml --train_for 1000 --alpha 0.7 --beta 0.3
```
Default: α=0.7 (DDPG), β=0.3 (COACH)

DDPG learns from intrinsic rewards, COACH learns from human corrections