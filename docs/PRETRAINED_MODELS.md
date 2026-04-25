# Pretrained Models

Put downloaded checkpoints under `checkpoints/`, then set `checkpoint.path` in `configs/test_spr_example.yaml`.

| Model | Generator config | File | Size | SHA256 | OneDrive |
| --- | --- | --- | --- | --- | --- |
| SPR-DR-SRRes | `srres` | `SPR-DR-SRRes.pth` | 6.2 MB | `55002b8cd53a0bbddecf82399b63873ffce2adf610117e7b26b6c28ed421cf36` | TODO |
| SPR-DR-RRDN | `rrdn` | `SPR-DR-RRDN.pth` | 63.9 MB | `bc479d1c1279e9b9a44a1ecf977f6384dc3d5202e81933a41a037acbd0a10e34` | TODO |
| SPR-DR-SwinIR-S-X4 | `swinir-s` | `SPR-DR-SwinIR-S-X4.pth` | 16.4 MB | `7846340242cc748c2605f31d12cf44ef69906d3653de01e7f92f93bc0c864655` | TODO |
| SPR-DR-SwinIR-M-X4 | `swinir-m` | `SPR-DR-SwinIR-M-X4.pth` | 64.7 MB | `5fcdc6dc9b03b84a807017346893082b2a8b53fdd480dd80e6fe68f1df820761` | TODO |

Example:

```yaml
model:
  generator: swinir-s
  srf: 4

checkpoint:
  path: ./checkpoints/SPR-DR-SwinIR-S-X4.pth
  key: null
  strict: true
```

Verify a downloaded file:

```bash
sha256sum checkpoints/SPR-DR-SwinIR-S-X4.pth
```

