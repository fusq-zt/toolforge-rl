# Failure Analysis

| type | Vanilla | Efficient |
|---|---:|---:|
| correct | 1859 | 1858 |
| parse_fail | 65 | 34 |
| retrieval_undercall | 12 | 32 |
| wrong_answer | 683 | 695 |

## Representative Efficient failures

- `math/train/130` (code_reasoning, wrong_answer, calls=0): Given the function $f(x)=3x^3+2$, find the value of $x$ so that $f^{-1}(x)=4$.
- `math/train/1596` (code_reasoning, wrong_answer, calls=0): The workers in a factory produce widgets and whoosits. For each product, production time is constant and identical for all workers, but not necessarily equal for the two products. 
- `math/train/1610` (code_reasoning, wrong_answer, calls=0): I randomly pick an integer $p$ between $1$ and $10$ inclusive. What is the probability that I choose a $p$ such that there exists an integer $q$ so that $p$ and $q$ satisfy the equ
- `math/train/1761` (code_reasoning, wrong_answer, calls=0): These two spinners are divided into thirds and quarters, respectively. If each of these spinners is spun once, what is the probability that the product of the results of the two sp
- `math/train/1831` (code_reasoning, wrong_answer, calls=0): Mary and James each sit in a row of 7 chairs. They choose their seats at random. What is the probability that they don't sit next to each other?
- `math/train/2004` (code_reasoning, parse_fail, calls=1): An ant moves on the following lattice, beginning at the dot labeled $A$. Each minute he moves to one of the dots neighboring the dot he was at, choosing from among its neighbors at
- `math/train/2052` (code_reasoning, wrong_answer, calls=1): A Senate committee has 5 Democrats and 5 Republicans. In how many ways can they sit around a circular table if each member sits next to two members of the other party? (Two seating
- `math/train/2080` (code_reasoning, wrong_answer, calls=0): In a tournament each player played exactly one game against each of the other players. In each game the winner was awarded $1$ point, the loser got $0$ points, and each of the two 
- `math/train/2474` (code_reasoning, wrong_answer, calls=1): Our water polo team has 15 members. I want to choose a starting team consisting of 7 players, one of whom will be the goalie (the other six positions are interchangeable, so the or
- `math/train/2647` (code_reasoning, wrong_answer, calls=0): Circles of radius 2 and 3 are externally tangent and are circumscribed by a third circle, as shown in the figure. Find the area of the shaded region. Express your answer in terms o
- `math/train/2744` (code_reasoning, wrong_answer, calls=0): Polygon $ABCDEF$ is a regular hexagon. What is the measure in degrees of angle $ABF$?
- `math/train/2870` (code_reasoning, wrong_answer, calls=0): Two boards, one four inches wide and the other six inches wide, are nailed together to form an X. The angle at which they cross is 60 degrees. If this structure is painted and the 
