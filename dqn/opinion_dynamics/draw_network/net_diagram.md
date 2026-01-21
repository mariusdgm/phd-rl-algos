```mermaid
flowchart LR
  X["x (B, N)"]
  FC["MLP
  Linear(128) → ReLU
  Linear(128→128) → ReLU"]
  Head["Heads
  A_diag (B, 𝓛, N)
  b (B, 𝓛, N)
  c (B, 𝓛)"]

  X --> FC --> Head
```