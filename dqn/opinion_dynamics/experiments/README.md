```
pip install git+https://github.com/mariusdgm/liftoff.git@windows-compatibility#egg=liftoff --upgrade --no-cache-dir
```

```
liftoff-prepare configs --do --runs-no 3
```

```
liftoff training_opinion.py .\results\2025Aug05-003412_configs --procs-no 6
```

