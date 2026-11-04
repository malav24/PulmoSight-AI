# Model status

| model | architecture | training_script_exists | checkpoint | calibration | ood_statistics | evaluation_metrics | ready_for_inference |
|---|---|---|---|---|---|---|---|
| model_a | implemented (5.2 M params) | True | missing | missing | missing | missing | False |
| model_b | implemented (5.2 M params) | True | missing | missing | missing | missing | False |
| ablation_cls_only | implemented (5.2 M params) | True | missing | missing | missing | missing | False |
| resnet50 | implemented (23.5 M params) | True | missing | missing | missing | missing | False |
| densenet121 | implemented (7.0 M params) | True | missing | missing | missing | missing | False |
| efficientnet_b0 | implemented (4.0 M params) | True | missing | missing | missing | missing | False |
| vit_small | implemented (21.8 M params) | True | missing | missing | missing | missing | False |

Checkpoint locations:

- model_a: `checkpoints/hybrid/model_a_best.pth`
- model_b: `checkpoints/hybrid/model_b_best.pth`
- ablation_cls_only: `checkpoints/hybrid/ablation_cls_only_best.pth`
- resnet50: `checkpoints/baselines/resnet50_best.pth`
- densenet121: `checkpoints/baselines/densenet121_best.pth`
- efficientnet_b0: `checkpoints/baselines/efficientnet_b0_best.pth`
- vit_small: `checkpoints/baselines/vit_small_best.pth`
