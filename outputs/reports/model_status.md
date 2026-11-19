# Model status

| model | architecture | training_script_exists | checkpoint | calibration | ood_statistics | evaluation_metrics | ready_for_inference |
|---|---|---|---|---|---|---|---|
| model_a | implemented (use --build to instantiate) | True | found (epoch 3, saved 2026-11-19T10:36:51) | found (T=1.048) | missing | found | True |
| model_b | implemented (use --build to instantiate) | True | found (epoch 3, saved 2026-11-03T22:56:06) | found (T=1.249) | missing | found | True |
| ablation_cls_only | implemented (use --build to instantiate) | True | found (epoch 1, saved 2026-11-19T11:21:07) | found (T=1.177) | missing | found | True |
| resnet50 | implemented (use --build to instantiate) | True | found (epoch 2, saved 2026-11-19T14:10:41) | found (T=1.078) | missing | found | True |
| densenet121 | implemented (use --build to instantiate) | True | found (epoch 3, saved 2026-11-19T13:13:59) | found (T=1.283) | missing | found | True |
| efficientnet_b0 | implemented (use --build to instantiate) | True | found (epoch 3, saved 2026-11-19T12:16:24) | found (T=1.845) | missing | found | True |
| vit_small | implemented (use --build to instantiate) | True | found (epoch 6, saved 2026-11-19T15:18:02) | found (T=1.320) | missing | found | True |

Checkpoint locations:

- model_a: `checkpoints/hybrid/model_a_best.pth`
- model_b: `checkpoints/hybrid/model_b_best.pth`
- ablation_cls_only: `checkpoints/hybrid/ablation_cls_only_best.pth`
- resnet50: `checkpoints/baselines/resnet50_best.pth`
- densenet121: `checkpoints/baselines/densenet121_best.pth`
- efficientnet_b0: `checkpoints/baselines/efficientnet_b0_best.pth`
- vit_small: `checkpoints/baselines/vit_small_best.pth`
