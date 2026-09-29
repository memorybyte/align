# Crazyflie 2.1 Deployment Memory Analysis

## Crazyflie 2.1 Hardware: STM32F405RG

| Resource | Total | Available for Application |
|----------|-------|--------------------------|
| **Flash** | 1 MB (1024 KB) | ~256 KB free (firmware uses ~700–750 KB) |
| **RAM** | 192 KB (128 KB SRAM + 64 KB CCM) | ~40–50 KB free (firmware + FreeRTOS overhead) |
| **CPU** | ARM Cortex-M4 @ 168 MHz | Has hardware FPU (single-precision float32) |

> Only the **Actor** network is needed on-board. The Critic is used only during training and is discarded at deployment.

---

## Current Model Architecture

 **Total number of parameters**  **535,614**

 **Total size = (535,614*4 Bytes ~ 2.04 MB)** 

### Runtime RAM Per Inference Step

| Buffer | Size |
|--------|------|
| LSTM hidden state `h` | 256 × 4 bytes = 1,024 B |
| LSTM cell state `c` | 256 × 4 bytes = 1,024 B |
| `fc1` activations | 256 × 4 bytes = 1,024 B |
| LSTM output buffer | 256 × 4 bytes = 1,024 B |
| Input (obs) + output (action) | (27 + 4) × 4 bytes = 124 B |
| Misc. | ~2,048 B |
| **Total RAM** | **~6.2 KB** |

---


## Requirements

                           
Flash (FP32 weights):   2.04 MB              
Flash (FP16 weights):   1.02 MB              
Flash (INT8 weights):   0.52 MB               
RAM at runtime:         ~6.2 KB            

---

## Optimizations we can do

### 1: Reduce hidden_size 256 → 64


| Layer | Params | FP32 | INT8 |
|-------|--------|------|------|
| `obs_norm` (LayerNorm 27) | 54 | 216 B | 54 B |
| `fc1` Linear (27→64) | 1,792 | 7,168 B | 1,792 B |
| `fc1` LayerNorm (64) | 128 | 512 B | 128 B |
| LSTM (input=64, hidden=64) | 33,024 | 132,096 B | 33,024 B |
| `lstm_norm` (LayerNorm 64) | 128 | 512 B | 128 B |
| `action_mean` (64→4) | 260 | 1,040 B | 260 B |
| **TOTAL** | **35,390** | **141,560 B (~138 KB)** | **~35 KB** |

---

### 2: INT8 Post-Training Quantization

We can convert trained FP32 weights to 8-bit integers. With `hidden_size=64`:

- **35 KB** for INT8 weights (vs 138 KB FP32) 
- Runtime still in FP32 or fixed-point arithmetic; dequantize on the fly


