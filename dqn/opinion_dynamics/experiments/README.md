```
pip install git+https://github.com/mariusdgm/liftoff.git@windows-compatibility#egg=liftoff --upgrade --no-cache-dir
```

```
liftoff-prepare configs --do --runs-no 4
```

```
liftoff training_opinion.py .\results\2025May29-012929_configs --procs-no 8
```

