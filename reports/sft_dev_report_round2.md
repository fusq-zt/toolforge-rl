# ToolForge SFT Dev recovered round 2 rollout summary

Episodes: 200

| scope | accuracy | parse | schema | avg calls |
|---|---:|---:|---:|---:|
| overall | 76.50% | 96.00% | 96.00% | 1.070 |
| code_reasoning | 58.33% | 86.67% | 86.67% | 0.383 |
| direct_anchor | 97.50% | 100.00% | 100.00% | 1.000 |
| retrieval_reasoning | 70.00% | 100.00% | 100.00% | 1.800 |
| retrieve_then_compute | 92.50% | 100.00% | 100.00% | 1.075 |

SFT Gate: **FAIL**

- final_parse_ge_0_98: FAIL
- schema_valid_ge_0_98: FAIL
- tool_execution_ge_0_95: PASS
- infinite_loop_zero: PASS
- all_families_nonzero_accuracy: PASS
- retrieve_then_compute_ge_0_35: PASS
- accuracy_above_base: PASS
