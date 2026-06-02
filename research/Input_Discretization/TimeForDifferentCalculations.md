### All weights upfront
============================================================
Batch 13 | Configs 961-1000/1000
============================================================
[Batch init (40 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (40 configs)] elapsed: 15.38s
[Batch prediction (40 configs)] elapsed: 11.72s
[Result extraction (40 configs)] elapsed: 0.11s
Batch time: 27.22s (40 configs)

============================================================
Total combinations processed: 1000/1000
============================================================


Results saved to: results/Chaotic/LorenzLHS\90.0.pkl
Memory released: 0.11 MB
Total time for PP 90.0: 1064.31 seconds
======================================================================

~ 17 minutes total time for a single PP value.
This is a clear step up from the Serial version which took 72 minutes for single PP.


## Batched Weight generation

============================================================
Batch 15 | Configs 897-960/1000
============================================================
[Building batch weights (64 configs)] elapsed: 22.25s
[Batch init (64 configs)] elapsed: 0.17s
Readout training complete.
[Batch training (64 configs)] elapsed: 304.59s
[Batch prediction (64 configs)] elapsed: 412.03s
[Result extraction (64 configs)] elapsed: 4.39s
Batch time: 743.43s (64 configs)

============================================================
Batch 16 | Configs 961-1000/1000
============================================================
[Building batch weights (40 configs)] elapsed: 14.18s
[Batch init (40 configs)] elapsed: 0.07s
Readout training complete.
[Batch training (40 configs)] elapsed: 8.73s
[Batch prediction (40 configs)] elapsed: 11.05s
[Result extraction (40 configs)] elapsed: 0.13s
Batch time: 34.16s (40 configs)

============================================================
Total combinations processed: 1000/1000
============================================================


Results saved to: results/Chaotic/LorenzLHS\95.0.pkl
Memory released: 0.11 MB
Total time for PP 95.0: 2337.18 seconds
======================================================================


This is massive difference(38 Minutes). But the real reason behind the time increase is that the 
memory of the GPU is not enough to hold everything. it does lots of calculation in the
system shared memory. To confirm this I will run again the same code but now with system shared
memory turned off. You may see in batch 15 the time taken in training is 304s and batch prediction also 
took 412 sec.

## Batched Weight Generation with Shared Memory Turned Off

The Code crashed with 64 batch size, rerunning with 50 batch size.

============================================================
Batch 20 | Configs 951-1000/1000
============================================================
[Building batch weights (50 configs)] elapsed: 15.46s
[Batch init (50 configs)] elapsed: 0.07s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.31s
[Batch prediction (50 configs)] elapsed: 14.36s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 41.38s (50 configs)
============================================================
Total combinations processed: 1000/1000
============================================================
Results saved to: results/Chaotic/LorenzLHS\100.0.pkl
Total time for PP 100.0: 922.41 seconds = 15 min.
======================================================================

## With Batching and sparse weights

============================================================
Batch 32 | Configs 993-1000/1000
============================================================
[Building batch weights (8 configs)] elapsed: 2.89s
[Batch init (8 configs)] elapsed: 0.01s
[Batch training (8 configs)] elapsed: 2.32s
[Batch prediction (8 configs)] elapsed: 3.00s
[Result extraction (8 configs)] elapsed: 0.00s
Batch time: 8.22s (8 configs)
============================================================
Total combinations processed: 1000/1000
============================================================
Results saved to: results/Chaotic/MultiChuaLHS\95.0.pkl
Total time for PP 95.0: 815.69 seconds = 13 min.
======================================================================
