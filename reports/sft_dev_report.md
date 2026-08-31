# ToolForge SFT Dev after bounded protocol recovery rollout summary

Episodes: 200

| scope | accuracy | parse | schema | avg calls |
|---|---:|---:|---:|---:|
| overall | 75.50% | 95.00% | 95.00% | 1.125 |
| code_reasoning | 63.33% | 83.33% | 83.33% | 0.467 |
| direct_anchor | 92.50% | 100.00% | 100.00% | 0.975 |
| retrieval_reasoning | 78.33% | 100.00% | 100.00% | 1.917 |
| retrieve_then_compute | 72.50% | 100.00% | 100.00% | 1.075 |

SFT Gate: **PASS**

- final_parse_ge_0_95: PASS
- schema_valid_ge_0_95: PASS
- tool_execution_ge_0_95: PASS
- infinite_loop_zero: PASS
- all_families_nonzero_accuracy: PASS
- retrieve_then_compute_ge_0_35: PASS
- accuracy_above_base: PASS
- noncode_parse_ge_0_98: PASS
- noncode_schema_ge_0_98: PASS
