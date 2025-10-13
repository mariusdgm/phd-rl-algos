```mermaid
flowchart LR
  X["x (B, N)"]
  FC["FC block
  Linear(N→64) → ReLU
  Linear(64→64) → ReLU"]
  Head["Heads
  A_diag (B, J, N)
  b (B, J, N)
  c (B, J)"]

  X --> FC --> Head
```