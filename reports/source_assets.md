# Source Assets

Audit date: 2026-09-01. Canonical IDs and immutable revisions below are frozen
before any download. Transport mirrors do not change provenance.

| Kind | Canonical ID | Revision | License at audit | First use | Redistribution decision |
|---|---|---|---|---|---|
| code | `RUC-NLPIR/Tool-Star` | `df08f67a89b27feda425306cfe892d65f6569f9a` | MIT | Gate 1 protocol reference | keep upstream code under `third_party`, retain license/NOTICE |
| model | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Qwen Research (`other` metadata) | base/SFT/eval | never commit weights; publish download instructions |
| model | `dongguanting/Tool-Star-Qwen-3B` | `2350a1d6ec6230d48e41f752babab730bad8aa92` | MIT | Gate 1 sanity and teacher | reference only; never claim as project training result |
| dataset | `dongguanting/Tool-Star-SFT-54K` | `f85b4fe0809a30a3b360b6f61689e462f79e2ec1` | MIT | Gate 1 sanity; ≤1K filtered SFT | do not publish full copy; publish selected IDs/build script |
| dataset | `dongguanting/Multi-Tool-RL-10K` | `c9574c31986db590a322d8924925a8b9684fb6c5` | MIT | format reference only | no replacement for mined RL prompts |
| dataset | `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | MIT | train + frozen test | scripts/manifests; avoid committing full derived text |
| dataset | `EleutherAI/hendrycks_math` | `21a5633873b6a120296cce3e2df9d5550074f4a3` | MIT metadata | train only | scripts/manifests; MATH-500 is the final math set |
| dataset | `HuggingFaceH4/MATH-500` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | not declared on card | final eval only | no redistribution until terms are confirmed |
| dataset | `hotpotqa/hotpot_qa` | `1908d6afbbead072334abe2965f91bd2709910ab` | CC BY-SA 4.0 | distractor train; frozen validation 200 | attribution/share-alike; publish IDs/config, not full derivative corpus |
| dataset | `framolfese/2WikiMultihopQA` | `fe713bfbd1afbca1a65246741a75890405d56a3a` | upstream Apache-2.0; HF field empty | train; frozen validation/test 200 | retain upstream attribution/LICENSE; publish IDs/config |

## Acquisition procedure

1. Resolve the canonical ID to the exact revision in the manifest.
2. Download into the project HF cache, using `HF_ENDPOINT` only as transport when
   official access remains unavailable.
3. Record actual snapshot path, file list, sizes, and resolved commit.
4. Fail if the resolved revision differs, a repository becomes gated/private, or
   a license changes materially.
5. Generate the final evaluation ID manifest before any training example is built.

## Scope and lineage

- Public final splits are never passed to teacher generation, SFT construction,
  RL prompt mining, temperature selection, or lambda selection.
- `Multi-Tool-RL-10K` is inspected for format only.
- Tool-Star code/model/data are external assets and are not project results.
- The repository publishes reconstruction scripts, IDs, configs, manifests, and
  small examples; raw corpora and model weights remain external.

## Network note

Server probes found PyPI and `hf-mirror.com` reachable while direct GitHub and
Hugging Face HTTPS timed out. Tool-Star Git commit was independently queried from
the canonical Git remote on a connected host. Gate 1 must preserve this provenance
when transferring code or downloading mirrored snapshots.
