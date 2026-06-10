# cancer-recog-apoptosis — working rules

This project INVENTS something not yet in the literature. Online research and spawned subagents can only
summarise already-published work — they cannot judge whether an un-invented thing will work. The data is
the oracle.

1. **Groundbreaking empirical testing > online research.** Default to building a runnable script that lets
   the DATA / atlas / simulation answer the question. A few targeted web lookups are fine for grounding;
   never substitute them for a real run.
2. **No subagent swarms.** Do NOT spawn Workflow/Agent fan-outs (20–30 subagents) to "verify feasibility"
   or do SOTA research — they burn tokens guessing at things that exist nowhere. Run the experiment instead.
3. **Honest negatives are first-class.** Never overclaim. Every predicted result is a HYPOTHESIS with a
   stated wet-lab residual; report threshold-sensitive numbers as ranges; let the atlas say no when it says no.
4. **Census / anndata (recurring fixes — apply in EVERY `get_anndata` call):**
   - Use `obs_column_names=[...]` / `var_column_names=[...]`, NOT the deprecated `column_names={"obs": [...]}` (FutureWarning). (filters stay `obs_value_filter`/`var_value_filter`.)
   - Immediately after `get_anndata`, **stringify the indices**: `ad.obs_names = ad.obs_names.astype(str); ad.var_names = ad.var_names.astype(str)` — else anndata coerces the integer `soma_joinid` index on first `ad.obs[...]`/`ad.var[...]` access and spams `ImplicitModificationWarning: Transforming to str index` (benign but noisy).
5. **Audit a result BEFORE building on it; re-audit old runs/data too.** Every result — especially a STRONG one, and ESPECIALLY a NEGATIVE that closes a direction — is a claim about the *method* until verified. A surprising result is more often a bug than a discovery. **Cost asymmetry:** a false positive gets caught downstream (you build on it, it fails); a false negative is SILENT — it kills a real opportunity you never revisit → verify negatives *at least as hard* as positives. When a run finishes (or when revisiting an old run/dataset), before propagating any conclusion or starting the next rung:
   - **Biological sanity first (cheapest tripwire):** does the number contradict known biology? *(RUNG-23 v1's 62% proliferation "leak" into POST-MITOTIC cardiomyocytes/neurons — which cannot divide — was impossible → a method bug, not a finding.)* Can't-be-true ⇒ a flag, not a result.
   - **Were the disciplines applied?** (checklist below). Most of our bugs were *"already fixed in another rung, not carried over"* — so apply the checklist **proactively when WRITING** a new atlas script, not only when auditing after.
   - **Could a better method flip it?** If plausibly yes ⇒ HARDEN + RE-RUN before concluding, and **downgrade any claim already written** until the hardened re-run confirms (never leave an overclaim standing).
   **Method-disciplines checklist (atlas / scRNA / expression runs):**
   (a) **DEPTH-GATE** (RUNG-18b) — score only well-sequenced cells (housekeeping panel ≥ k detected) so dropout doesn't deflate the signal / fake a "dead" reading.
   (b) **ROBUST DONOR STAT** (RUNG-8) — per-donor distribution (median / p90), NEVER the single worst donor (one outlier fakes a leak); report which donor/cell-type drives any extreme.
   (c) **DATASET-MEASURING FILTER** (RUNG-8) — Census returns 0 for "not assayed" too; only count datasets that detect the gene somewhere.
   (d) **SELFTEST the math first** with an artifact-resistance check; report the tumour-vs-control CONTRAST, not dropout-sensitive absolutes.
6. **We are INVENTING what isn't in the market/literature yet — "unprecedented" ≠ "impossible".** (The preamble's premise, as a guard so a future AI doesn't dismiss a novel design as confused/wrong.) Two surprises get OPPOSITE treatment — do not confuse them with rule 5:
   - **Violates a physical/biological LAW or contradicts MEASURED ground truth** → bug, reject (rule 5; e.g. proliferation in post-mitotic cells).
   - **Merely ABSENT from prior work / "SOTA doesn't do this"** → that is the NOVELTY we are building — do NOT use it to reject a design or downgrade a result. Examples that are *intended inventions, not errors*: neoantigen-gated bystander apoptosis wave; layered T+NK+wave that closes both escape routes; **autonomous intracellular mutation-sensing AND-gate self-destruct**. None exist in the literature — that absence IS the contribution.
   Training-data / online-research / spawned-subagent priors describe what EXISTS, never what is POSSIBLE; they cannot judge an un-invented thing — **the DATA / simulation is the only oracle.** Novelty is NEUTRAL evidence: "not in papers" neither proves it works (still must pass rule 1 empirical validation + rule 5 law-sanity) nor proves it can't. **Hold BOTH at once:** iron-man ambition (build the un-done) AND honesty (validate, state the wet-lab residual). Groundbreaking = law-respecting novelty CONFIRMED BY DATA — never a law-breaking artifact, never literature-mimicry.
7. **Notebook cells — magics on their OWN line + AST-check EVERY cell before shipping the Colab link (recurring bug, cost ≥3 round-trips).** In a Jupyter/Colab code cell an IPython `!shell` or `%magic` must be on its own line — it CANNOT follow a Python statement on the same line, and CANNOT be inline after `:` or `;`:
   - ✗ `os.chdir(x); !git log`  ✗ `if cond: !pip install ...`  → `SyntaxError: invalid syntax`.
   - ✓ split onto separate lines; for conditional shell use `subprocess.run([...])`, not an inline `!`/`%`.
   GPU/Colab notebooks can't be run locally, so a single bad line = a wasted user round-trip. **MANDATORY before handing over any .ipynb:** AST-parse every code cell with magics neutralised — for each line whose stripped form starts with `!`, `%`, or `#@`, replace it with `pass` (preserving indent), then `ast.parse` the cell; fix all SyntaxErrors first. Also: editing the GitHub notebook does NOT update an already-open Colab session (it loaded the pre-fix copy) — tell the user to re-open the link fresh or hand them the exact one-line in-place edit.
