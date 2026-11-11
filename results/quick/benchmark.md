| model | config | trainable params | test bpc | test accuracy | train time |
|---|---|---|---|---|---|
| Classic reservoir (adam) | R=750 | 44,309 | 3.114 | 37.9% | 2.9 min |
| Classic reservoir (ridge) | R=750 | 44,309 | 3.772 | 38.9% | 5.2 min |
| Classic reservoir (adam, v0.1 scaling) | R=750 | 44,309 | 3.123 | 37.9% | 2.8 min |
| Attention-enhanced reservoir | R=75, H=13 | 15,464 | 3.056 | 37.6% | 0.2 min |
| Attention-enhanced reservoir | R=75, H=19 | 31,124 | 2.967 | 39.1% | 0.2 min |
| Attention-enhanced reservoir | R=100, H=20 | 45,259 | 2.952 | 39.1% | 0.2 min |
| Attention-enhanced reservoir | R=150, H=25 | 102,809 | 2.920 | 39.6% | 0.4 min |
| Attention-enhanced reservoir | R=160, H=30 | 155,459 | 2.894 | 40.6% | 0.4 min |
| Attention-enhanced reservoir (v0.1 scaling) | R=150, H=25 | 102,809 | 2.894 | 40.7% | 0.3 min |
| Transformer | L=4, heads=4, ffn=64 | 15,067 | 3.122 | 36.3% | 4.1 min |
| Transformer | L=8, heads=8, ffn=72 | 30,299 | 3.056 | 37.5% | 12.6 min |
| Transformer | L=8, heads=8, ffn=128 | 45,083 | 3.115 | 35.9% | 14.2 min |
| Transformer | L=8, heads=8, ffn=356 | 105,275 | 4.410 | 15.9% | 20.4 min |
| Transformer | L=16, heads=16, ffn=256 | 155,803 | 4.410 | 15.9% | 57.1 min |
| Transformer (lr 1e-3) | L=8, heads=8, ffn=128 | 45,083 | 3.253 | 33.8% | 14.2 min |
| Transformer (lr 1e-3) | L=8, heads=8, ffn=356 | 105,275 | 3.276 | 32.9% | 20.8 min |
| Transformer (lr 1e-3) | L=16, heads=16, ffn=256 | 155,803 | 4.409 | 15.9% | 58.0 min |
