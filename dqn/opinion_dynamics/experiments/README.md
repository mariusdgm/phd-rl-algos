```
pip install git+https://github.com/mariusdgm/liftoff.git@windows-compatibility#egg=liftoff --upgrade --no-cache-dir
```

```
liftoff-prepare configs --do --runs-no 2
```

```
liftoff training_opinion.py .\results\2025Apr21-222011_configs --procs-no 5
```

beta, nu e u_max ca pana acum, ci limitat la n noduri influente
change reward to use original action