| Session name | Model     | LR     | LR schedule | Optimizer                   | Training data augmentation      | Best epoch | Best valid loss | Best RMSE |
| ------------ | --------- | ------ | ----------- | --------------------------- | ------------------------------- | ---------- | --------------- | --------- |
| Adni_Net16   | ResNet-101| 1e-4   | 3 / 0.75    | As below                    | As below                        | 45         | 0.0086          | 29.18     |
| Adni_Net15   | ResNet-34 | 1e-4   | 3 / 0.75    | As below                    | As below                        | 22         | 0.0378          | 43.53     |
| Adni_Net14   | ResNet-18 | 1e-4   | 3 / 0.75    | As below                    | As below                        |  6         | 0.0408          | 45.71     |
| Adni_Net13   | ResNet-50 | 1e-4   | 3 / 0.75    | As below                    | Normalize R/G/B                 | 60         | 0.0147          | 29.71     |
| Adni_Net12   | ResNet-50 | 1e-4   | 3 / 0.75    | As below                    | center_crop 0.8-1.0, res100x100 | 24         | 0.0771          | 49.91     |
| Adni_Net11   | ResNet-50 | 5e-4   | 3 / 0.75    | As below                    | Normalize R/G/B                 | 16         | 0.0371          | 42.43     |
| Adni_Net10   | ResNet-50 | 1e-3   | 3 / 0.75    | As below                    | Contrast / brightness 0.95-1.05 | 48         | 0.0549          | 41.98     |
| Adni_Net9    | ResNet-50 | 2e-3   | 3 / 0.75    | As below                    | Contrast / brightness 0.9-1.1   | 43         | 0.1239          | 50.45     |
| Adni_Net8    | ResNet-50 | 5e-3   | 3 / 0.75    | As below                    | As below                        | 43         | 0.2567          | 61.98     |
| Adni_Net7    | ResNet-34 | 1e-2   | 3 / 0.75    | AdamW, weight_decay=0.25e-3 | As below                        | 50         | 0.2296          | 61.29     |
| Adni_Net6    | Resnet-34 | 1e-4   | 3 / 0.75    | As below                    | As below                        | 32         | 0.2468          | 61.59     |
| Adni_Net6    | Resnet-34 | 1e-4   | 3 / 0.75    | As below                    | Contrast / brightness 0.75-1.25 | 43         | 0.1652          | 57.41     |
| Adni_Net5    | Resnet-34 | 1e-4   | 3 / 0.75    | As below                    | As below                        | 13         | 0.02624         | 39.57     |
| Adni_Net4    | Resnet-18 | 1e-4   | 3 / 0.75    | As below                    | Normalize R/G/B                 | 19         | 0.03235         | 41.06     |
| Adni_Net3    | Resnet-18 | 1e-4   | 3 / 0.75    | As below                    | As below                        | 30         | 0.3047          | 62.81     |
