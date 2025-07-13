```
pip install git+https://github.com/mariusdgm/liftoff.git@windows-compatibility#egg=liftoff --upgrade --no-cache-dir
```

```
liftoff-prepare configs --do --runs-no 3
```

```
liftoff training_opinion.py .\results\2025Jul12-011043_configs --procs-no 8
```

