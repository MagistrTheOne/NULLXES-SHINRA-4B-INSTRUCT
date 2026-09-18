# Pilot builder: budgets, resume and recovery

The mixture in `data/sources.py`, tokenizer specials/DNA, filters and model
architecture are unchanged. `--max-tokens` is an upper bound, not a promise that
all quotas can be filled simultaneously. Sources are now scheduled by remaining
bucket/source/language deficits. Source weights provide soft scheduling priorities;
the existing bucket and language budgets remain the hard limits.

## Resume the existing Colab pilot

Update the repository, then run this in a **new process** (not a notebook import
that still holds the old builder):

```bash
git pull --ff-only
python -m data.build_pilot \
  --output-dir data/clean/pilot \
  --tokenizer-corpus-dir tokenizer/corpus \
  --max-tokens 100000000 \
  --resume
```

Run from `/content/NULLXES-SHINRA-4B-INSTRUCT` in Colab. Use the original
`--tokenizer-corpus-dir` if the earlier run used a different directory. No GPU or
model initialization is needed for ingestion. Do not run two builders at once.

Without `--resume`, any existing pilot shards/state or nonempty pilot corpus
cause an error **before streaming**. `--overwrite` is a separate, mutually
exclusive, explicitly destructive option for an intentional fresh build. It is
not needed for recovery. It removes only this pilot's managed shards/state,
report and `pilot_corpus.txt`, not arbitrary files in either directory.

For legacy `clean-00000.parquet` through `clean-00009.parquet`, the first resume:

1. Reads existing Parquet in batches and reconstructs bucket/language/source/
   script token counters and accepted document counts.
2. Computes `sum(max(int(row['n_chars']) // 4, 1) for row in rows)` exactly as
   ingestion does. **Do not replace this with `sum(n_chars) // 4`: integer
   rounding happens per document.** The 193,712,334-character corpus is around
   48.4M estimated tokens, but its exact count requires its actual rows.
3. Builds a persistent index of document IDs, normalized-text SHA256 hashes and
   MinHash signatures. Next shard is **highest existing index + 1**, so index 10
   follows `clean-00009.parquet`, including when earlier indices have gaps.
4. Preserves every existing shard byte-for-byte. Reads the existing corpus,
   indexes its lines, and appends only missing eligible lines up to the corpus
   character cap. Existing text, including an old unflushed tail, is preserved.
5. Replays HF streams from the beginning. This is **not** a remote cursor seek:
   old raw/filter-rejected rows must be scanned again. Existing IDs and exact
   text duplicates cannot produce new output; the production MinHash filter is
   also restored for approximate near-duplicate detection. Missing upstream IDs now receive
   deterministic normalized-content hashes instead of random UUIDs.

First reconstruction computes the historical MinHash index; it can take time
for large documents. Subsequent normal restarts restore counters from checkpoint
and load saved signatures without recomputing them from text. Replay time is
still real; ETA stays unknown until new tokens are accepted. Legacy raw scanned
and rejected counts are not inferable from Parquet: reports explicitly mark
`seen` as a lower bound, `scan_counts_complete=false`, cumulative acceptance as
`null`, and provide exact session counters/acceptance separately.

## Deterministic stopping and progress

Every source displays its name and bucket, accepted bucket tokens/cap/percentage,
source tokens, stored source documents, session scanned/kept/dropped counts,
acceptance, new estimated tokens/second, and ETA when there is a nonzero rate.
There is no tokenizer/model/config construction per document for accounting.

Whole documents are admitted only if they fit the **bucket, language and global**
budgets. A document that would cross the remaining bucket budget closes the
bucket: the builder never searches indefinitely for a smaller document. A small
remainder also closes it before reading another row:

```text
remaining <= min(--tail-tokens, bucket_cap // 1000)
```

`--tail-tokens` defaults to 2048; this threshold is at most 0.1% of the bucket.
The document-boundary rule may leave a larger remainder (less than the size of
the rejected candidate). Boundary stops survive a valid checkpoint resume under
the same budget. Changing the budget resets those stops. A forced-language
source, or a source with a known primary language, stops at that language's
cap/document boundary. Unknown mixed-language sources may continue looking for
languages with room. Language tails use the same 2048/0.1% rule as bucket tails.

`--max-source-docs` defaults to **5,000,000 raw rows per source per invocation**,
including resume replay. It provides a finite bound even for all-rejected or
all-duplicate streams. Its stop reason is `source_scan_limit`, not success. It
can be increased explicitly if necessary. As with any network stream, this row
bound does not impose a wall-clock timeout on a blocked HTTP request.

`pilot_report.json` contains absolute token counters, bucket/language caps and
progress, source counts, scan/keep/drop counters, acceptance, shard count, next
index, resume mode, per-source stops, overall stop reason, session rates and the
existing tokenizer friendship result. Reports expose underfilled budgets;
neither weights nor language quotas are silently changed to reach 100M.

## Scheduling an interrupted 75M pilot

The scheduler reevaluates eligibility after a shard, 4096 scanned rows, or roughly
1/32 of the source's recipe token target, whichever comes first. Each eligible
source gets a bounded visit per round, ordered by relative bucket deficit, then
source deficit, then language deficit. The token visit threshold yields between
whole documents; it is not a new source cap. Live iterators are retained during
switches: RU/DE/FR can all contribute without replaying their streams on every
switch or letting one source monopolize the multilingual bucket. Raw scan limits
remain per source per invocation, not per visit. Unflushed rows can share a shard
across sources; existing shard files and schema stay unchanged.

Full buckets and closed primary languages are excluded **before opening a stream**.
Known English-primary sources are FineWeb-Edu, python-edu, stack-smol and
OpenWebMath. Wikipedia's existing forced languages map to RU/EU. These profiles
only control scheduling: they do not relabel documents or override the filters.
Occasional non-English OpenWebMath rows do not make it a reliable source for
backfilling a Russian language quota after English has closed.

For the reported state:

| Budget | Accepted | Remaining |
| --- | ---: | ---: |
| general | 49,998,423 | 1,577 |
| code | 24,991,782 | 8,218 |
| math_stem | 13,049 | 14,986,951 |
| multilingual | 0 | 10,000,000 |
| English | 74,999,949 | **51** |

`--resume` schedules Wikipedia RU/DE/FR. It skips the general tail and English
sources, including code and OpenWebMath, rather than scanning millions of rows
for the final 51 English tokens. If English has sufficient headroom, OpenWebMath
is eligible and its math deficit gives it priority over mostly filled buckets.

**The requested final bucket and language totals are incompatible with the
existing source language profiles.** General + code already consume almost the
entire 75M English budget. The remaining math bucket asks for 14,986,951 tokens,
but its primary language has only 51 available: a shortfall of 14,986,900 tokens.
Multilingual has a 10M bucket cap, so it cannot absorb the entire remaining 25M
by itself. Finishing multilingual yields roughly 85M overall; the scheduler must
not report that as a completed 100M pilot.

The report adds `scheduling` (initial eligible/skipped sources, selection counts
and visit limits), `quota_conflicts` (required/available/shortfall tokens and
source language profiles), and `completion_status`. When language limits block
the remaining targets it reports `stop_reason=quota_conflict` and
`completion_status=blocked_by_language_quotas`. Resolving that conflict requires
an explicit change to quotas, source coverage or retained data. This patch does
none of those automatically.

The scheduler adds no SQLite tables, changes no checkpoint version, and does not
invalidate a valid existing checkpoint. Resume loads it normally and preserves
old index entries, corpus prefix and Parquet. Only active source streams replay
from the beginning across separate processes, as before. No index deletion or
rebuild is required merely to install this scheduling patch.

## Crash consistency

The POSIX output and corpus locks prevent concurrent writers. New Parquet is
written and fsynced to a temporary file, then published with an atomic hard link
that fails if the target exists. Existing `clean-*.parquet` is never opened for
writing during resume.

At each shard flush:

1. Publish the complete Parquet shard.
2. Persist an atomic append journal for new corpus text, append and fsync it.
   Recovery verifies any partially written suffix and appends its remainder,
   including interrupted UTF-8 characters. It never truncates the corpus.
3. Commit the ID/hash/MinHash index in `pilot_index.sqlite3` with a checkpoint
   mirror in the same SQLite transaction.
4. Atomically replace and fsync `pilot_state.json`.

The state records shard names/sizes/mtimes, corpus path/size/mtime, counters,
next index, stopping state and dedup library version. A missing, malformed or
stale JSON checkpoint, missing index, mismatched mirror, changed shard/corpus
manifest or changed datasketch version triggers reconstruction from Parquet.
No checkpoint needs unsafe pickle deserialization.

Ctrl+C during iteration saves the accepted buffer, checkpoint and report, then
exits interrupted. A kill/error during publication leaves either the preceding
checkpoint or a complete new shard; the next resume reconciles against Parquet.
A journal that disagrees with externally modified corpus bytes stops with an
error and preserves those bytes rather than guessing. Temporary unpublished
files are not treated as accepted shards. Keep state/index alongside the shards
when transferring the pilot if you want the fast checkpoint path.

## RoPE warning investigation

The original warning is reproducible with `ShinraConfig()` on Transformers
5.16.1: its default `rope_scaling` included `factor: 1.0`, while HF's default
RoPE validator does not accept `factor`. The config now omits it for default
RoPE, including legacy `rope_scaling` input, without changing actual rotary
calculations. Non-default scaling factors remain intact; input dicts are not
mutated.

The repository's document loop does **not** call `ShinraConfig`.
`tokenizer.corpus_friendship` and `tokenizer.special_tokens` do not initialize it
either. The unnecessary eager Transformers import was in `tokenizer/__init__.py`
and is now inside `load_shinra_tokenizer`. Friendship loads a trained tokenizer
only once during the final report, if artifacts exist. The opt-in toxicity model
(`SHINRA_TOXICITY_MODEL`) uses a cached classifier, not a per-row SHINRA config.
The repeated warnings in the old Colab process cannot be attributed to a specific
external notebook call without that process's stack; no blanket warning
suppression has been added.

## Local validation (no dataset downloads)

```bash
python -m pip install pyarrow datasets ftfy datasketch tqdm pytest transformers==5.16.1
python -m pytest -q tests/test_pilot_resume.py tests/test_pilot_schedule.py
python -m compileall -q data tokenizer model tests
python -m data.build_pilot --help
git diff --check
```

Tests cover fresh/legacy/checkpoint resume, all counter reconstructions, 10
synthetic shards with 40,960 rows and 193,712,334 characters, exact per-row
estimates, next index 10, byte-preserved shards/corpus, duplicate IDs/text/MinHash,
quota boundaries, infinite rejected input, fail-if-existing publication, locks,
Ctrl+C, crashes after Parquet/corpus and before JSON, partial UTF-8 append
recovery, missing corpus, and stale/missing/malformed state. A subprocess runs
real document filters with model/Transformers imports forbidden. A separate test
checks repeated config initialization against Transformers 5.16.1.

The user's actual Colab Parquet files are not available in the repository;
local synthetic fixtures validate the equivalent counts/layout. Resume prints
the actual recovered totals on the Colab machine before starting source replay.

Scheduler regressions cover the exact reported counters, early exclusion of
English sources, math selection with English headroom, dynamic deficit ordering,
bounded visits to rejected sources without iterator restarts, and an end-to-end
scaled 75M resume fixture. The latter forbids checkpoint reconstruction, checks
existing SQLite rows and an extra sentinel table, verifies byte-identical old
Parquet and corpus prefix, and confirms contributions from RU/DE/FR while the
math/English quota conflict remains explicitly reported.
