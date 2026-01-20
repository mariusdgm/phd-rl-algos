```mermaid
flowchart LR
  X["x (B, N)"]
  FC["MLP
  Linear(128) → ReLU
  Linear(128→128) → ReLU"]
  Head["Heads
  A_diag (B, L, N)
  b (B, L, N)
  c (B, L)"]

  X --> FC --> Head
```