# ASD-STE100 audit: REAP and DSpark cost calculators

Scope: every user-visible string of the REAP Cost Calculator and the DSpark
Training Cost Calculator. That is every file in the two calculator directories,
plus their routes and their libraries:

```
src/routes/ReapCostCalculator.tsx
src/routes/DsparkTrainingCostCalculator.tsx
src/tools/reap-cost-calculator/          (6 files)
src/tools/dspark-training-cost-calculator/ (6 files)
src/lib/reap/                            (6 files)
src/lib/dspark/                          (6 files, faq.ts excluded by decision)
```

## Method

Four independent passes, because no single one of them sees every string:

1. **Rendered output.** A throwaway test rendered both results panels and both
   forms through `react-dom/server` across every runtime branch, then stripped
   the markup and measured the text a reader sees. Sentences were counted after
   the interpolation values were substituted, because a sentence that spans an
   expression reads differently on screen than it does in the source.
2. **Runtime strings.** A second throwaway test called `estimateReap`,
   `estimateDspark`, `detectMoeShape`, and `detectDsparkShape` directly and
   dumped every `steps[].label`, `steps[].detail`, `constants[].source`, and
   `assumptions[]` string with real values substituted. This is the copy that
   the collapsed breakdown accordion hides from a server render, and it is the
   only way to measure it.
3. **Source literals, in scope.** A third pass walked every string literal in all
   23 in-scope files, substituted each interpolation with the longest value its
   slot can hold, and checked the rules. This catches copy that only renders in a
   state the two passes above did not reach. It is the pass that found the hook
   strings below.
4. **Source literals, whole repository.** The same check over every non-test file
   under `src/`, so that the remaining violations in excluded files can be named
   rather than left implicit.

The built `dist` HTML of both routes was checked with the first pass, with the
FAQ section cut off.

Branches covered:

- REAP results: 10 GPUs x 3 weight precisions x 2 top-k settings, plus the paper
  recipe, a large micro batch, slow storage, a long run, 14 model configs, the
  idle, loading, not-moe, config error, and compute error states.
- DSpark results: 10 GPUs x 2 data modes x 3 GPU counts, plus the anchor clamp,
  slow storage, a disabled Markov head, a large token count, 8 configs that each
  trigger a shape note, the draft-config alert, and the idle, loading, error, and
  compute error states.
- Both forms: 3 input modes x a named recipe and a custom one x 2 data modes,
  plus the invalid-field branches and every `ModelIdField` status, which is what
  exercises the `idleHint`, `errorHint`, and `notMoeHint` copy.
- Runtime strings: 2,380 strings from 14 REAP configs and 8 DSpark configs x 7
  variants.

Totals: 7,520 rendered blocks, 2,380 runtime strings, 202 in-scope source
literals, and 644 repository-wide source literals.

## Rules checked

| Rule | Requirement | Result |
| --- | --- | --- |
| 1.1 | Use approved words only | Not applied, by decision. The approved-word dictionary would remove the domain terms the pages exist to explain. |
| 1.6 | Do not use contractions | Pass. 0 found in the in-scope files. |
| 2.1 | Maximum 3 words in a noun cluster | Pass, with the exceptions listed below. |
| 3.2 | Use the simple present or simple past tense | Pass in the in-scope files. 0 occurrences of `would`, `could`, `should`, `may`, or `might`, and no perfect tense. The 11 remaining repository-wide occurrences are all in excluded files and are named below. |
| 3.5 | Use the `-ing` form only as a noun or a modifier | Pass. No `-ing` verb form carries the action of a sentence. |
| 4.1 | Maximum 20 words in a sentence | Pass in the in-scope files. The longest rendered sentence is 19 words, and the longest runtime string sentence is 19 words. The 32 remaining repository-wide occurrences are all in excluded files and are named below. |
| 4.3 | Maximum 6 sentences in a paragraph | Pass. The longest rendered paragraph is 5 sentences, and the longest runtime string paragraph is 6 sentences. |
| 8.1 | Do not use the semicolon | Pass. 0 found. |
| 8.4 | Do not use the em dash or the en dash | Pass. 0 found. |
| Glossary | One term for one concept across both pages | Pass. See the term list below. |

## Violations fixed in the last pass

The first three passes missed the two `use*Shape` hooks, because the earlier file
lists for the rewrite stopped at the seven files per calculator and the audit
report's scope line was wider than that list. The fourth pass found them. Every
string below renders in the results alert or the form status line, so all of them
were reachable.

| Location | Before | After |
| --- | --- | --- |
| `useReapShape.ts` pasted config error | That config could not be read. | The calculator cannot read that config. |
| `useReapShape.ts` hub error | Could not read that model config. | The calculator cannot read that model config. |
| `useDsparkShape.ts` pasted config error | That config could not be read. | The calculator cannot read that config. |
| `useDsparkShape.ts` hub error | Could not read that model config. | The calculator cannot read that model config. |
| `useDsparkShape.ts` not a decoder | That config does not describe a decoder with a hidden size and a depth, so there is nothing to cost out. (21 words) | That config describes no decoder with a hidden size and a depth. The calculator therefore has nothing to cost. |

The shared `ModelIdField` also has a default `errorHint` of `Could not read that
config.` That component is excluded from the rewrite, and the KV cache calculator
still uses its default. Both in-scope forms now pass their own `errorHint`
instead, so neither calculator page renders the excluded default. This is
verified by rendering both forms in the `error` status and confirming that the
default string appears 0 times.

## Remaining violations, named, all in excluded files

Nothing in scope remains. These are the repository-wide occurrences, each in a
file that the agreed scope excludes. They are listed here so that the report
names every remaining violation rather than leaving them implicit.

**`src/lib/reap/faq.ts` and `src/lib/dspark/faq.ts`** (excluded: SEO copy, and
the user asked for the questions and answers to stay as they are). 18 long
sentences and 2 modals:

- `src/lib/reap/faq.ts`: lines 12, 14, 19, 24, 29, 32, 34 (long sentences), and 2 modals.
- `src/lib/dspark/faq.ts`: lines 14, 19 (x3), 24 (x2), 29 (x2), 34, 39, 44 (x2), 49.

**`src/lib/kvcache/**`** (excluded: shared library for the KV cache calculator).
17 long sentences and 7 modals:

- `src/lib/kvcache/compute.ts`: lines 284, 308, 353, 451, 563, 660, 786, 790, 869, 973, 1022, 1029, 1034, 1254, 1258, 1267, 1298.
- `src/lib/kvcache/fetch-config.ts`: lines 173, 190, 199, 222.
- `src/lib/kvcache/dtypes.ts`: line 178.

**`src/components/model-source/ModelSourceFields.tsx:76`** (excluded:
`src/components/**`). One modal, the `ModelIdField` default `errorHint`. Not
rendered on either calculator page, as described above.

**`src/lib/seo.ts`** (excluded: SEO metadata). Lines 40 and 44, two long
sentences, both meta descriptions.

**`src/tools/registry.ts`** (excluded: tool card descriptions). Lines 30 and 38,
two long sentences.

**`src/routes/KvCacheCalculator.tsx:45`** (a different page, not in scope). One
modal.

## Term glossary as applied

| Concept | Term used | Terms avoided |
| --- | --- | --- |
| The graphics processor | GPU | card, graphics card, accelerator |
| The memory on the GPU | VRAM | device memory, GPU memory, card memory |
| One layer that holds the expert bank | expert block | MoE block, decoder block |
| The whole set of expert blocks | expert bank | expert layers |
| The fraction of experts removed | pruning ratio | prune ratio, ratio, sparsity |
| The pass that measures expert saliency | calibration | calibration run, calibration pass |
| The published settings a run follows | recipe | preset, configuration |
| The published set of weights | checkpoint | model file, weights file |
| The bytes one weight takes | weight precision | weight format, dtype, format |
| The frozen model a drafter attaches to | target | target model, base model |
| The small model that proposes tokens | drafter | draft model |
| The hidden states written by the target | target cache | cache on its own |
| The medium the weights are read from | storage | disk, drive, medium |
| The headline number | estimate | figure, result |
| One job of the calculator | run | job, execution |
| The work the GPU does | arithmetic | compute, math, FLOPs work |
| The number the GPU reaches each second | throughput | rate, speed |
| A number read from the model config | value | constant |
| The time a run takes | duration | wall clock, time |

## Exceptions, each with a reason

1. `Router-weighted Expert Activation Pruning` is 4 words. It is the expansion of
   the acronym REAP and the name of the published method. It stays as written.
2. `DSpark Training Cost Calculator` is 4 words. It is the product name, and it
   is also the page heading and the browser title.
3. `3 x hidden_size x moe_intermediate_size` reads as a long cluster to a word
   counter, but it is a formula, and the two config keys inside it are names the
   model config uses. It stays as written.

No other exception was found. The remaining candidates from the checker were
false positives: a verb inside a clause that a word counter cannot separate from
the noun before it, a relative clause such as `verification schedule you deploy`,
or the GPU note `The largest single-GPU budget here.` which lives in the shared
`src/lib/hardware/gpus.ts` and is out of scope.

## Excluded on purpose and verified unchanged

`src/lib/reap/faq.ts`, `src/lib/dspark/faq.ts`, `src/lib/seo.ts`,
`src/tools/registry.ts`, `src/components/**`, `src/lib/hardware/**`,
`src/lib/model-shape/**`, `src/lib/kvcache/**`, and `src/lib/model-config/**` are
not rewritten. `git diff` confirms they are unchanged, apart from the two
`errorHint` props the in-scope forms now pass.

## Numbers

Every number, unit, formula, config key name, and threshold keeps its original
value. The only numeric differences in the diff are spelled-out numbers that
became digits, which the rules ask for: `one` to `1`, `two` to `2`, `six` to `6`,
`eight` to `8`, `ten` to `10`, and `twelve` to `12`. One expression was removed,
the plural branch of the GPU count sentence, and it was dead code: that sentence
only renders when the run needs 2 GPUs or more.

## Commands

```
pnpm test      # 17 files, 308 tests, all pass
pnpm build     # tsc, vite, the SSR bundle, and the prerender of 4 routes
pnpm lint      # no errors, 2 pre-existing warnings in src/components/ui
```
