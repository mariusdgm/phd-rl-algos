```mermaid
flowchart LR
  X["x (B, N)"]
  FC["Feature Extractor
  Linear(128) → ReLU
  Linear(128→128) → ReLU"]
  Head["Heads
  A_diag (B, J, N)
  b (B, J, N)
  c (B, J)"]

  X --> FC --> Head
```