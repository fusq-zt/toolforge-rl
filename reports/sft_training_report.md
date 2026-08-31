# Gate 5 SFT Report

Status: **PASS**

## Training

- Smoke: 256 usable rows, 0 truncated, loss 0.9302.
- Full: 2076 usable rows, 0 truncated, one epoch, loss 0.6521.
- Bounded protocol recovery: 704 train-only rows, 0 truncated, 0.5 epoch, loss 0.2422.
- Base → SFT Dev accuracy: 15.00% → 75.50% (+60.50 pp).

## Gate checks

- smoke_gradient_nonzero: PASS
- smoke_loss_finite: PASS
- smoke_adapter_saved: PASS
- smoke_adapter_reload: PASS
- smoke_truncation_le_10pct: PASS
- full_gradient_nonzero: PASS
- full_loss_finite: PASS
- full_adapter_reload: PASS
- dev_sft_gate: PASS
- recovery_gradient_nonzero: PASS
- recovery_loss_finite: PASS
- recovery_zero_truncation: PASS
- recovery_adapter_reload: PASS
