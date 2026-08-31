# 20 条拟生成 Episode 设计示例

这些示例只用于 Gate 3 Pilot 的协议设计与实现自检，不是已经生成、验证或
入选的数据，也不代表模型效果。所有名称、数值和文档均为训练侧的合成
fixture，不来自 GSM8K test、MATH500、HotpotQA validation、2WikiMultiHopQA
validation/test 或其他冻结最终测试样本。

每个 Episode 均向模型提供相同的两个工具 `python_exec` 与 `local_search`。
下文的“典型路径”用于说明期望验证的 Agent Loop，不是静态 gold route：最终
是否正确只由指定 verifier 判定，直接得到正确答案或采用另一条合法路径仍可
通过；工具调用效率由实际轨迹统计。

## A. `direct_anchor`（5 条）

### E01 — 基础算术

- Prompt：`一盒有 8 支笔，4 盒共有多少支？`
- Episode 文档：无。
- 典型路径：直接输出 Final；也允许调用 `python_exec("8*4")` 后输出 Final。
- Reference：`32`。
- Verifier：numeric exact；允许附带“支”这一展示单位。

### E02 — 分数比较

- Prompt：`0.72 和 7/10 哪个更大？`
- Episode 文档：无。
- 典型路径：直接比较并输出 Final；Python 计算是合法但非必要路径。
- Reference：`0.72`。
- Verifier：numeric rational comparison，并检查所选答案等于 `0.72`。

### E03 — Prompt 内显式事实

- Prompt：`备忘录写着“临时标签是 CEDAR-24”。临时标签是什么？`
- Episode 文档：无；答案已经在用户 Prompt 中。
- 典型路径：零调用直接输出 Final。
- Reference：`CEDAR-24`。
- Verifier：Unicode、大小写、标点及空白归一化后的 exact match。

### E04 — 列表位置

- Prompt：`在 [榆木, 松木, 桦木, 杉木] 中，第三项是什么？`
- Episode 文档：无。
- 典型路径：直接读取列表并输出 Final；也允许 Python 列表索引。
- Reference：`桦木`。
- Verifier：normalized exact match。

### E05 — 简单时间相加

- Prompt：`计时器先记录 14 秒，又经过 9 秒，总共多少秒？`
- Episode 文档：无。
- 典型路径：直接输出 Final；若调用 Python，则执行 `14+9` 后输出 Final。
- Reference：`23`。
- Verifier：numeric exact；可忽略答案中的“秒”单位。

## B. `code_reasoning`（5 条）

### E06 — 多步整数运算

- Prompt：`仓库有 17 箱零件，每箱 26 个，质检淘汰 19 个，还剩多少个？`
- Episode 文档：无。
- 典型路径：`python_exec("17*26-19")` → observation → Final；正确的直接推理也接受。
- Reference：`423`。
- Verifier：numeric exact。

### E07 — 分数运算

- Prompt：`计算 5/14 + 3/7，并化为最简分数。`
- Episode 文档：无。
- 典型路径：使用 allowlisted `fractions.Fraction` 的 `python_exec` → Final。
- Reference：`11/14`。
- Verifier：exact rational equivalence，且规范化为最简分数。

### E08 — 中位数

- Prompt：`求 [13, 4, 9, 7, 16, 5] 的中位数。`
- Episode 文档：无。
- 典型路径：`python_exec` 调用 `statistics.median` → observation → Final。
- Reference：`8`。
- Verifier：numeric exact。

### E09 — 组合计数

- Prompt：`从 16 个不同的传感器中选出两个组成无序对，共有多少种选法？`
- Episode 文档：无。
- 典型路径：`python_exec("16*15//2")` → observation → Final。
- Reference：`120`。
- Verifier：numeric exact。

### E10 — 一元方程

- Prompt：`求解 4(x-3)=28。`
- Episode 文档：无。
- 典型路径：直接代数推理，或用 `python_exec` 验算候选解后输出 Final。
- Reference：`x=10`。
- Verifier：MATH expression equivalence；只解析 Final，不把 Python 成功视为答对。

## C. `retrieval_reasoning`（5 条）

### E11 — 单文档事实检索

- Prompt：`诺拉·芬恩演奏什么乐器？`
- Episode 文档：`D1 诺拉·芬恩在乐团中演奏大提琴。`；
  `D2 伊沃·莱恩是一名小提琴制作师。`
- 典型路径：`local_search("诺拉 芬恩 乐器")` → observation → Final。
- Reference：`大提琴`。
- Verifier：normalized exact match，alias 包含 `cello`。

### E12 — 人物到城市再到河流

- Prompt：`莱奥·坦出生的城市由哪条河流经？`
- Episode 文档：`D1 莱奥·坦出生于诺维克。`；
  `D2 埃兰河流经诺维克市中心。`；另含两个城市的干扰文档。
- 典型路径：第一次 `local_search` 找到出生地，第二次以城市名检索河流 → Final。
- Reference：`埃兰河`。
- Verifier：normalized alias match，aliases 为 `{埃兰河, 埃兰}`。

### E13 — 两实体关系比较

- Prompt：`赤隼实验室与琥珀工坊的创始人出生在同一个国家吗？`
- Episode 文档：分别给出两家公司创始人、两位创始人的出生国及无关人物干扰项。
- 典型路径：一次宽查询或两次实体查询的 `local_search` → 汇总证据 → Final。
- Reference：`是`。
- Verifier：boolean alias match，接受 `{是, 相同, yes}`。

### E14 — 同名作品消歧

- Prompt：`电影《静默子午线》的导演是谁？`
- Episode 文档：电影条目写明导演为 `安娜·博雷尔`；另有一本同名小说和两条无关电影记录。
- 典型路径：`local_search("静默子午线 电影 导演")` → observation → Final。
- Reference：`安娜·博雷尔`。
- Verifier：Unicode、标点与空白归一化后的 exact/alias match。

### E15 — 开馆时间比较

- Prompt：`云雀博物馆和北岸档案馆中，哪一个更早开放？`
- Episode 文档：`D1 云雀博物馆于 1987 年开放。`；
  `D2 北岸档案馆于 1992 年开放。`；另含装修年份干扰项。
- 典型路径：一次或两次 `local_search` 取得开放年份 → 比较 → Final。
- Reference：`云雀博物馆`。
- Verifier：normalized exact/alias match；年份只作为证据，不直接作为答案。

## D. `retrieve_then_compute`（5 条）

### E16 — 两期数量差

- Prompt：`奥林公园六月游客比五月多多少人？`
- Episode 文档：五月游客 `12,480`；六月游客 `15,205`；另有其他公园及其他月份干扰项。
- 典型路径：`local_search` 检索两个月数值 → `python_exec("15205-12480")` → Final。
- Reference：`2725`。
- Verifier：programmatic numeric exact；lineage 检查两个操作数均来自本 Episode 文档。

### E17 — 百分比

- Prompt：`报告显示发出 240 台曙光设备，其中退回 18 台，退回比例是多少？`
- Episode 文档：发货量与退回量分别位于两个文档片段，并含上一季度数字作为干扰项。
- 典型路径：`local_search` 取得同一期的 `240` 和 `18` →
  `python_exec("18/240*100")` → Final。
- Reference：`7.5%`。
- Verifier：numeric percent equivalence；lineage 和报告期必须一致。

### E18 — 数量乘单价求和

- Prompt：`7 枚维拉徽章和 4 个诺克斯夹子的总价是多少？`
- Episode 文档：维拉徽章单价 `$3.25`；诺克斯夹子单价 `$1.80`；另含相似商品价格。
- 典型路径：`local_search` 取得两个单价 → 使用 `decimal` 的 `python_exec` 计算 → Final。
- Reference：`$29.95`。
- Verifier：numeric decimal exact to cents；忽略货币展示符号并校验操作数 lineage。

### E19 — 年份间隔

- Prompt：`贝里尔附馆比奥尔德大厅晚开放多少年？`
- Episode 文档：奥尔德大厅开放于 `1968`；贝里尔附馆开放于 `1985`；另有修缮年份干扰项。
- 典型路径：`local_search` 检索两个“开放”年份 → `python_exec("1985-1968")` → Final。
- Reference：`17`。
- Verifier：programmatic numeric exact；字段语义必须是开放年份。

### E20 — 比值化简

- Prompt：`同一赛季中，索尔队得 84 分，卢门队得 63 分。给出索尔:卢门的最简比。`
- Episode 文档：两队当季得分位于不同片段，且各含一个往季分数干扰项。
- 典型路径：`local_search` 取得同赛季得分 → `python_exec` 用最大公约数化简 → Final。
- Reference：`4:3`。
- Verifier：rational ratio equivalence；校验赛季一致性及两个操作数 lineage。

## 设计自检

- 共 20 条，四类任务各 5 条；
- 只涉及 `python_exec`、`local_search` 与自然的 `local_search → python_exec`；
- 每条均给出典型路径、Reference 和确定性 verifier；
- 典型路径不作为唯一正确工具标签，工具成功也不替代最终答案正确；
- 所有内容均为训练侧合成设计 fixture，不含冻结最终测试题或其改写模板。
