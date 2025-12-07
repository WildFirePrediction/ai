"""
A3C V7 LSTM RECALL - LARGE MODEL with RECALL-FIRST SUPERAGGRO Reward Shaping
Scaled-up version of V6 with:
- 3.4x larger model (~6M parameters)
- 512 hidden LSTM (was 256)
- Deeper/wider feature extractor
- Batch normalization
- 12 workers (was 4)
- 25k episodes (was 10k)
- Learning rate schedule
- Higher entropy coefficient

Philosophy: Coverage is EVERYTHING. Better to evacuate an entire mountain than miss one house.
"""
