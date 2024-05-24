# The Implementation Code for STAR

This repo contains the implementation code for STAR.

/ps/ is for the PS architecure and /all_reduce/ is for the All-Reduce architecture.

The list below shows the main components for each method.
- **Straggler prediction**: code_for_training_data/train.py (class LstmRegr)
- **Heuristic mode determination**: training_sgd*.py
- **PGNS**: gradient_validation/ and pgns.py
- **ML mode determination**: regression.py
- **Resource consideration**: res_dist_managing.py
- **PS assignment**：blossom/
- **Tree**: training_sgd*.py
