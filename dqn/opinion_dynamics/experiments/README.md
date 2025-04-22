```
pip install git+https://github.com/mariusdgm/liftoff.git@windows-compatibility#egg=liftoff --upgrade --no-cache-dir
```

```
liftoff-prepare configs --do --runs-no 2
```

```
liftoff training_opinion.py .\results\2025Apr21-222011_configs --procs-no 5
```