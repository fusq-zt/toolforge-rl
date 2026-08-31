# 40 项 Reward 测试计划

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

## 边界、Verifier 与日志一致性（R21–R30）

| ID | 输入场景 | 预期断言 |
|---|---|---|
| R21 | 分数 `1/2` 对参考小数 `0.5`，零调用 | numeric verifier 判正确，Vanilla=`1.05`；Reward 不自行实现第二套答案语义。 |
| R22 | 工具 observation 为正确数值，但 Final 给出另一个数 | verifier 判错，Vanilla=`0.05`；工具执行成功不替代最终正确。 |
| R23 | 恰好两个 invalid calls，Final 可解析但错误 | invalid penalty=`0.40`，与三个 invalid 的封顶结果一致。 |
| R24 | completion 标记 truncated，但截断前已有完整、可解析且正确的 Final | v1 没有隐藏 truncation 扣分，Vanilla=`1.05`；truncation 仍写 telemetry/质量报告。 |
| R25 | 达到三次最大调用后退出且没有 Final | `correct=false`，missing-final=`0.20`；合法调用不产生正奖励。 |
| R26 | malformed tool marker 后仍能独立解析出正确 Final | marker 计 invalid、format=`0`、answer=`1.00`，Vanilla=`0.80`。 |
| R27 | `python_exec` 因文件 I/O 策略被拒，随后给出错误 Final | policy rejection 计一个 invalid；Vanilla=`-0.15`（`0.05-0.20`）。 |
| R28 | `python_exec` 命中输出长度上限，但随后 Final 由 verifier 判正确 | output-limit 计一个 invalid；Vanilla=`0.85`，执行失败不抹掉真实 Final 正确性。 |
| R29 | 浮点预测分别位于 numeric verifier 容差内与刚超出容差 | 仅前者得到 answer=`1.00`；其余 reward 分量保持完全一致。 |
| R30 | MATH 等价表达式由 `math-verify` 判等与 parser 失败各一例 | 等价例 answer=`1.00`；parser 失败例 answer=`0`，不得由 Reward 猜测语义。 |

## Lambda、性质与故障输入（R31–R40）

| ID | 输入场景 | 预期断言 |
|---|---|---|
| R31 | 两条 clean 正确 `c=[1,3]`，`lambda=0.10` | Efficient=`[1.05,0.85]`。 |
| R32 | 两条 clean 正确 `c=[1,3]`，`lambda=0.25` | Efficient=`[1.05,0.55]`。 |
| R33 | 任意混合组，`lambda=0`（测试专用） | Efficient 逐条严格等于 Vanilla，验证实现只追加效率项。 |
| R34 | 原组 clean 正确 `c=[2,3]`，再加入 clean 正确 `c=0` | 新 `C*=0`；原两条 penalty 相应增大，错误项（若有）不变，体现 group-relative 语义。 |
| R35 | 两条错误 rollout 基础分量相同，分别 `c=0` 与 `c=100` | efficiency penalty 都为 0；错误项永不因成本得到 bonus 或 penalty。 |
| R36 | clean 正确 `c=0` 与正确但含一次 timeout 的 `c=1` | `C*=0`；后者同时扣 invalid=`0.20` 和 efficiency=`0.15`，总分 `0.70`。 |
| R37 | clean 正确 `c=1` 与含两次重复合法调用的 clean 正确 `c=3` | 第二条 efficiency penalty=`0.30`；重复调用全部计入实际成本。 |
| R38 | 对含 answer/format/invalid/final/efficiency 的混合组从组件日志重算 | 每条 `total` 与组件代数和在浮点容差内完全一致，且记录正确的 `C*`。 |
| R39 | telemetry 中调用数为负数、NaN 或非整数 | scorer 明确拒绝输入并返回结构化错误；不静默夹紧、不产生非有限 reward。 |
| R40 | 空 rollout 组与缺失 `prompt_id` 的记录 | group scorer 明确拒绝；不会跨 prompt 借用 `C*`，错误含可定位 reason code。 |

## 执行前验收

- 测试场景数必须恰好为 40：基础 Vanilla 10 项、核心 Efficient 10 项、
  边界/Verifier 10 项、Lambda/性质/故障输入 10 项；
- 每项同时断言总 reward 和相关分量，不只比较最终浮点数；
- 浮点断言使用小容差，并检查所有分量为有限数；
- Efficient 与 Vanilla 共用同一个 `score_vanilla` 实现；Efficient 只追加单独记录的 `efficiency_penalty`；
- 覆盖 correct/wrong、全错组、错误零调用、最低调用 ties、唯一正确、invalid、missing/unparseable Final、重复调用、组隔离和顺序不变性。
