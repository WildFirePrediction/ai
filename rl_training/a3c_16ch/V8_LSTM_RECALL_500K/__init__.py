"""
A3C V8 LSTM RECALL SMALL - Smaller Model with RECALL-FIRST SUPERAGGRO
Addressing overfitting with reduced capacity:
- ~500K parameters (3.5x smaller than V6, 14x smaller than V7)
- 128 hidden LSTM (was 256 in V6)
- Narrower CNN: 32→64→128 (was 64→128→256)
- 8 workers (balanced)
- 15k episodes

Goal: Better generalization by reducing overfitting
Training/Validation gap in V6: 3.16x (0.4356 vs 0.1379)
Target gap in V8: <2.0x

Philosophy: Coverage is EVERYTHING. Better to evacuate an entire mountain than miss one house.
"""
