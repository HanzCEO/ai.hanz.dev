# ASD-STE100 audit: REAP and DSpark cost calculators

Scope: every user-visible string of the REAP Cost Calculator and the DSpark
Training Cost Calculator.

Method: a throwaway script rendered both results panels and both forms through
`react-dom/server` across every runtime branch, then stripped the markup and
counted words in the text a reader actually sees. Sentences were counted after
the interpolation values were substituted, because a sentence that spans an
expression reads differently on screen than it does in the source. The built
`dist` HTML of both routes was checked the same way.

Branches covered:

- REAP results: 10 GPUs x 3 weight precisions x 2 top-k settings, plus the paper
  recipe, a large micro batch, slow storage, a long run, 14 model configs, the
  idle, loading, not-moe, config error, and compute error states.
- DSpark results: 10 GPUs x 2 data modes x 3 GPU counts, plus the anchor clamp,
  slow storage, a disabled Markov head, a large token count, 7 configs that each
  trigger a shape note, the draft-config alert, and the idle, loading, error, and
  compute error states.
- Both forms: 3 input modes x a named recipe and a custom one x 2 data modes,
  plus the invalid-field and single-epoch branches.
- The prerendered HTML of both routes, with the FAQ section excluded.

Total rendered text: 7,141 lines. The FAQ questions and answers, the SEO
metadata, the tool card descriptions, the shared components, and the shared
libraries are out of scope and unchanged.

## Rules checked

| Rule | Requirement | Result |
| --- | --- | --- |
| 1.1 | Use approved words only | Not applied, by decision. The approved-word dictionary would remove the domain terms the pages exist to explain. |
| 1.6 | Do not use contractions | Pass. 0 found. |
| 2.1 | Maximum 3 words in a noun cluster | Pass, with the exceptions listed below. |
| 3.2 | Use the simple present or simple past tense | Pass. No perfect tense, and no `would`, `could`, `should`, `may`, or `might`. |
| 3.5 | Use the `-ing` form only as a noun or a modifier | Pass. No `-ing` verb form carries the action of a sentence. |
| 4.1 | Maximum 20 words in a sentence | Pass. The longest rendered sentence is 19 words. |
| 4.3 | Maximum 6 sentences in a paragraph | Pass. The longest rendered paragraph is 5 sentences. |
| 8.1 | Do not use the semicolon | Pass. 0 found. |
| 8.4 | Do not use the em dash or the en dash | Pass. 0 found. |
| Glossary | One term for one concept across both pages | Pass. See the term list below. |

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
not in the diff.

## Numbers

Every number, unit, formula, config key name, and threshold keeps its original
value. The only numeric differences in the diff are spelled-out numbers that
became digits, which the rules ask for: `one` to `1`, `two` to `2`, `six` to `6`,
`eight` to `8`, `ten` to `10`, and `twelve` to `12`.

## Commands

```
pnpm test      # 17 files, 308 tests, all pass
pnpm build     # tsc, vite, the SSR bundle, and the prerender of 4 routes
pnpm lint      # no errors, 2 pre-existing warnings in src/components/ui
```
