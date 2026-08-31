# 20 项 Reward 测试计划

本计划在任何 GRPO 运行前转为可执行的参数化单元测试。测试只读取结构化
trajectory telemetry 与 verifier 结果，不相信模型对“已调用成功”或“答案正确”
的自述。

冻结公式如下：

- `R_vanilla = answer + format - invalid - final_parse`；
- `answer=1.00` 当且仅当 Final 通过 verifier；
- Final 可解析且全部已发出调用具有合法的已知工具 schema 时，`format=0.05`；
- 每个 invalid call 扣 `0.20`，总扣分封顶 `0.40`；
- Final 缺失或不可解析扣 `0.20`，且不能判为正确；
- `C_i=tool_call_count_i`，包含重复调用和执行失败的已发出调用；
- 同一 prompt 的 rollout 组内，若存在正确项，则
  `C*=min(C_j | correct_j=1)`；否则 `C*` 不存在；
- `R_efficient_i = R_vanilla_i - 0.15 * I(correct_i) * max(0,C_i-C*)`。

除特别说明外，Final 均可解析、调用 schema 均合法、没有 invalid call，因而
正确 rollout 的 Vanilla 为 `1.05`，错误 rollout 为 `0.05`。

## Vanilla Reward（R01–R10）

| ID | 输入场景 | 预期断言 |
|---|---|---|
| R01 | 零调用，Final 正确 | `answer=1.00`、`format=0.05`，Vanilla=`1.05`。零调用本身既不奖励也不惩罚。 |
| R02 | 零调用，Final 可解析但错误 | `answer=0`、`format=0.05`，Vanilla=`0.05`；错误的零调用不能获得正确性或效率奖励。 |
| R03 | 零调用，Final 缺失/不可解析 | `answer=0`、`format=0`、`final_parse=0.20`，Vanilla=`-0.20`；该轨迹绝不能被标为 correct。 |
| R04 | 一次合法且执行成功的调用，Final 正确 | Vanilla=`1.05`；执行成功不额外加分，正确性仍来自 Final verifier。 |
| R05 | 三次合法且执行成功的调用，Final 错误 | Vanilla=`0.05`；工具 observation 正确不能替代最终答案正确。 |
| R06 | 发出一个 unknown tool，Final 可解析但错误 | `invalid=0.20`、`format=0`，Vanilla=`-0.20`；`tool_call_count=1`。 |
| R07 | 一个参数 schema malformed 的调用，但另有可解析且 verifier 正确的 Final | `answer=1.00`、`format=0`、`invalid=0.20`，Vanilla=`0.80`。 |
| R08 | 三个 invalid calls，Final 可解析但错误 | invalid 原始和为 `0.60`，封顶为 `0.40`；`format=0`，Vanilla=`-0.40`。 |
| R09 | 一次 schema 合法但 timeout 的调用，随后没有 Final | timeout 计一个 invalid，另扣 missing-final；Vanilla=`-0.40`，且 `tool_call_count=1`。 |
| R10 | 连续两次完全相同但 schema/执行均合法的调用，Final 错误 | Vanilla=`0.05`；v1 不加隐藏重复惩罚，但 `tool_call_count=2` 且 repetition telemetry 必须为真。 |

## Efficient Reward（R11–R20）

以下数组顺序对应同一行给出的 rollout 顺序；除 R17 外，各正确 rollout 的
Vanilla 都是 `1.05`。

| ID | 同一 prompt 内的 rollout 组 | 预期断言 |
|---|---|---|
| R11 | 四条均正确，调用数 `c=[1,1,1,1]` | `C*=1`；四条 efficiency penalty 均为 `0`，Efficient 均等于 `1.05`。覆盖全组 ties。 |
| R12 | 四条均错误，`c=[0,1,2,3]` | 组内没有 `C*`；所有 efficiency penalty 都严格为 `0`，Efficient 逐条等于各自 Vanilla。 |
| R13 | 一条错误零调用、两条正确，`c=[0,1,2]` | `C*=1`；Efficient=`[0.05,1.05,0.90]`。错误零调用既无 bonus 也无 penalty。 |
| R14 | 三条正确，`c=[0,1,3]` | `C*=0`；excess=`[0,1,3]`，Efficient=`[1.05,0.90,0.60]`。只在正确轨迹间按实际调用数排序。 |
| R15 | 三条正确，`c=[1,1,3]` | `C*=1`；excess=`[0,0,2]`，Efficient=`[1.05,1.05,0.75]`。并列最低调用保持同分。 |
| R16 | 错误轨迹 `c=[0,1]`，唯一正确轨迹 `c=3` | `C*=3`；唯一正确项没有效率惩罚，Efficient=`1.05`；该组没有证据证明存在更便宜的正确路径。 |
| R17 | 正确但含一次 runtime-failed call 的轨迹 `c=1`（Vanilla=`0.85`），以及 clean 正确轨迹 `c=2`（Vanilla=`1.05`） | `C*=1`；efficiency penalty=`[0,0.15]`，Efficient=`[0.85,0.90]`。invalid 分量和相对效率分量分别计算、分别记录。 |
| R18 | 两条正确，`c=[1,3]`，第二条包含一次重复的合法调用 | `C*=1`；重复调用计入 `C`，penalty=`[0,0.30]`，Efficient=`[1.05,0.75]`。 |
| R19 | 两条均错误且基础 reward 分量相同，`c=[0,3]` | 两条 Efficient 都等于各自 Vanilla=`0.05`；错误项不因少调用获奖，也不因多调用被效率项扣分。 |
| R20 | Prompt A 有两条正确 `c=[1,3]`；Prompt B 有两条正确 `c=[2,2]`；输入顺序随机打乱 | A 独立得到 Efficient=`[1.05,0.75]`，B 独立得到 `[1.05,1.05]`；还原 `(prompt_id, rollout_id)` 后结果与打乱前完全一致，证明 group isolation 与 permutation invariance。 |

## 执行前验收

- 测试场景数必须恰好为 20：Vanilla 10 项、Efficient 10 项；
- 每项同时断言总 reward 和相关分量，不只比较最终浮点数；
- 浮点断言使用小容差，并检查所有分量为有限数；
- Efficient 与 Vanilla 共用同一个 `score_vanilla` 实现；Efficient 只追加单独记录的 `efficiency_penalty`；
- 覆盖 correct/wrong、全错组、错误零调用、最低调用 ties、唯一正确、invalid、missing/unparseable Final、重复调用、组隔离和顺序不变性。
