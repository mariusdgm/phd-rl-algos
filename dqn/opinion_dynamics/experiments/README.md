```
pip install git+https://github.com/mariusdgm/liftoff.git@windows-compatibility#egg=liftoff --upgrade --no-cache-dir
```

```
liftoff-prepare configs --do --runs-no 4
```

```
liftoff training_opinion.py .\results\2025May21-215142_configs --procs-no 8
```

