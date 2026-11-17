# TeqoAI technical audit

**Prepared:** 17 November 2026  
**Repository:** `C:\Users\A\Desktop\Projects\Teqo phase 1\teqoai`  
**Commit reviewed:** `8bbcf31` on `main` ("new home page added"). The only working-tree change is an untracked `docs_assets/` folder.  
**Scope:** review of `src/`, `dashboard/`, `tests/`, packaging, README and stored artifacts against the six INSA graduation-project requirements. Metrics below were recomputed from the checked-in synthetic artifacts. No application code was changed as part of this audit.

---

## Implementation status

The findings below describe the code as reviewed at commit `8bbcf31`. The following P0 work has since been implemented in the working tree (not committed):

| Item | Status |
|---|---|
| P0-1 label exposure | Fixed. `auth_events_risk.csv` no longer carries `is_malicious` or `evaluation_split`; the investigation record uses an explicit display whitelist, and SHAP uses the report's training cutoff instead of the dropped split column. |
| P0-2 queue evaluation | Fixed. The report now records `decision_queue`, which measures the rule-adjusted score and the dashboard's actual tier-and-score ordering at the top-30 and top-100 budgets. |
| P0-3 tier calibration | Fixed. Percentiles and thresholds are computed against the training period only. High is the top 5% of training-period risk (95th percentile) and Medium extends to the top 20%. The unclipped `risk_rank_score` breaks display ties; `risk_score` remains a clipped 0–100 presentation value. |
| P0-5 safety items (partial) | Fixed. Dynamic HTML is escaped, the devcontainer no longer disables CORS/XSRF, and the demo pipeline holds a cross-process lock. |
| P0-4 real-world evidence | **Blocked.** Still requires an approved dataset and a decision about LANL versus a sanitized organization export. |

After regeneration on the same synthetic profile: queue average precision **0.794** (first-seen baseline 0.655), top-30 precision **0.733** (22 of 30), High tier 27 holdout events with 22 known positives (81.5% precision, 66.7% recall). These remain synthetic-demonstration figures, not evidence of real-world detection quality. Test suite: 40 tests plus 4 subtests passing.

---

## 0. Priority key

| Priority | Meaning |
|---|---|
| **P0** | Fix before any judged demo or submission. These affect correctness, credibility or safety. |
| **P1** | Needed to make the prototype methodologically defensible and complete. |
| **P2** | Production-readiness work. A deliberate scope expansion beyond the current local prototype. |
| **P3** | Cleanup, consistency and repository hygiene. |

---

## 1. Bottom line

The project is a coherent, well-presented local prototype. The pipeline is modular, the interface is unusually polished for a graduation project, and the documentation is mostly honest about limitations. The test suite covers the core logic and currently passes.

Three findings dominate the audit:

1. **The analyst interface exposes ground-truth labels.** `is_malicious` and `evaluation_split` are rendered in the raw event record. Any viewer can read the answer next to the model score.
2. **The deployed review queue is not the object being evaluated, and it underperforms the simple baseline.** On the stored holdout, the first-seen rule reaches average precision 0.655 and 24 positives in the top 30; the deployed risk score reaches 0.485 and 18 positives in the queue's top 30. The evaluation report only covers raw model scores.
3. **There is no real-world evidence.** All artifacts are synthetic: 1,246 events, 60 users, 10 days, 33 positives in the holdout. That is enough to demonstrate software behaviour, not to support a claim about real authentication environments.

The good news is that the first two are localized fixes, not rewrites. A focused P0 pass is realistic in one to two days. The real-data gap needs a separate decision about access and scope.

---

## 2. What was verified

| Check | Result |
|---|---|
| Test suite | 37 tests passed, plus 4 subtests, in about 27 seconds in the local virtual environment |
| Dataset present | Synthetic profile only; no LANL or organization run in `data/processed` |
| Events | 1,246 total; 872 training, 374 holdout |
| Labels | 60 known positives total; 33 in the holdout |
| Risk tiers | High 359 (28.8%), Medium 403, Low 484 |
| Generic reasons | 969 of 1,246 events (77.8%) have only "elevated anomaly score, no specific risk factor triggered" |
| Explanation alignment | Stored Isolation Forest scores match recomputed scores to `9.7e-17`; alignment currently works but is positional and fragile |
| Repository state | Only untracked `docs_assets/`; no uncommitted source edits |

The Streamlit interface was not re-run in a browser for this audit. UI findings are based on source inspection. Metric findings are reproducible from the stored CSV and report.

---

## 3. P0 — fix before demo or submission

### P0-1. Ground-truth labels are visible in the analyst workspace

**Evidence.** `dashboard/views/investigation.py:53-66` converts every column of the selected row into a field/value table. The serving file `data/processed/auth_events_risk.csv` contains `is_malicious`, `evaluation_split`, `anomaly_score_*`, `pipeline_run_id` and other pipeline fields. The "Raw event record" expander therefore shows whether the event is malicious and whether it belonged to the train or test split.

**Why it matters.** It invalidates any live demonstration of detection. It also makes the analyst workflow look like it has access to information a real analyst would never have. A reviewer who notices this will question every metric on the evaluation page.

**Fix.** Define an explicit display whitelist in the investigation view: time, source, user, device, target device, action, result, authentication type, logon type, risk fields and event ID. Remove `is_malicious` and `evaluation_split` from the serving artifact entirely. Keep labels in a separate evaluation-only file joined by event ID. Add a test asserting that the investigator payload contains no label or split columns.

**Done when.** A grep of the rendered investigator record and of the serving CSV returns no `is_malicious` or `evaluation_split`, and the evaluation code still has access to labels through its own input.

---

### P0-2. The deployed queue is not evaluated, and it loses to the baseline

**Evidence.** The evaluation page says so directly at `dashboard/views/evaluation.py:457-458`: the report evaluates raw ranking scores, not the rule-adjusted risk score. Recomputed on the holdout:

| Ranker | Average precision | ROC-AUC | Top-30 precision |
|---|---:|---:|---:|
| First-seen destination baseline | **0.655** | **0.911** | **0.800 (24/30)** |
| Isolation Forest (primary model) | 0.448 | 0.860 | 0.367 (11/30) |
| Autoencoder (optional) | 0.503 | 0.905 | 0.533 (16/30) |
| Deployed risk score (ranked by score) | 0.485 | 0.876 | 0.533 (16/30) |
| Actual dashboard queue (tier, then score, then time) | n/a | n/a | 0.600 (18/30) |

The actual queue is better than the pure score order because tier grouping changes which rows appear first, but it is still below the 0.800 baseline. At a budget of 100, all rankers reach 0.280 precision.

There is a second problem: `risk_score` clips at 100 (`src/risk_engine.py:76`). In this dataset, **218 events sit at exactly 100**, and the top-30 cutoff falls inside a 41-way tie. The queue breaks the tie by timestamp (`dashboard/filters.py:65-80`), so the first page depends on an arbitrary ordering inside a large tied block rather than on evidence strength.

**Why it matters.** The interface calls the Isolation Forest the primary model, but the stored results show a one-line rule beating it on every headline ranking measure. That is not fatal scientifically, but the project cannot claim model superiority while the data says otherwise. It also cannot claim reproducible prioritisation while 218 events share the maximum score.

**Fix.**

- Evaluate the deployed artifact: the risk score, the tiered queue and the actual first-30/first-100 selections. Report average precision, precision/recall at fixed budgets, and bootstrap confidence intervals.
- Either improve the learned ranker until it beats the baseline, or make the baseline the primary ranker and present the ML models as comparison experiments. The honest option is to show both and explain why the baseline wins on this synthetic generator.
- Remove the 100-point clipping from ranking, or define an explicit deterministic tie-break using the unclipped score and event ID.
- Show the baseline and deployed queue on the Evaluation page side by side, with the same budget definitions.

**Done when.** The Evaluation page reports the score analysts actually see, the queue's top-k precision is reproducible from a fresh run, and no claim of "primary model" contradicts the table.

---

### P0-3. "High risk" is a quantile band, not a calibrated risk level

**Evidence.** `anomaly_percentile` is a full-dataset percentile rank (`src/risk_engine.py:59`), and `risk_level` is cut at 70 and 40 (`src/risk_engine.py:79-83`). On this dataset, the High tier is exactly the top 30% of Isolation Forest scores:

| Tier | All events | Holdout | Holdout positives | Holdout precision |
|---|---:|---:|---:|---:|
| High | 359 (28.8%) | 76 | 28 | 36.8% |
| Medium | 403 | 153 | 0 | 0.0% |
| Low | 484 | 145 | 5 | 3.4% |
| High + Medium | 762 (61.2%) | 229 (61.2%) | 28 | 12.2% |

No event below the 70th percentile entered High, and no event at or above it was excluded. The four named risk factors did not change tier membership at all on this dataset; they only reorder events inside a tier and saturate at 100. The Medium tier contains zero known positives in the holdout. Meanwhile 77.8% of all events have no named factor, only the generic "elevated anomaly score" reason.

**Why it matters.** A tier labelled High that is defined as "top 30% by construction" will always flag 30% of events, no matter how benign the environment. Calling this a risk level overstates calibration. The empty Medium tier also suggests the factors and thresholds are not doing useful triage work.

**Fix.** Pick one of two honest designs:

- **Calibrated tiers:** fit thresholds on a validation period to an explicit analyst capacity, then report precision, recall and lift per tier on the holdout. Store the thresholds with the model artifact.
- **Relative bands:** rename the tiers to "Priority 1/2/3" or "Review band A/B/C" and state plainly that they are relative quantiles, not probabilities or calibrated risk levels.

Do not leave "High" implying an absolute judgement while the boundary is a fixed percentile. Also report how many High events carry at least one named factor, and treat "no specific factor" as its own diagnostic state rather than a High-tier explanation.

**Done when.** The tier definition is stated in one sentence, the holdout table supports it, and the UI does not imply calibration that the method does not provide.

---

### P0-4. The INSA "real world" requirement is supported by synthetic evidence only

**Evidence.** Every artifact in `data/processed` and `models/` was generated by the synthetic profile. The generator creates 60 users over 10 days and encodes attacks as a stereotyped three-hop chain using NTLM, network logon and off-hours activity. The first-seen baseline's 0.800 top-30 precision is partly a measure of how well that stereotype is encoded, not of real-world detection.

**Why it matters.** INSA asks for a real-world cybersecurity problem and a technically functional solution. The problem is real and well described. The evidence is a software demonstration. A reviewer is likely to ask "how does this behave on actual authentication logs?" and the current answer is "we have not run it."

**Fix.** In order of preference:

1. Run the authorized LANL sample the code already supports. Record the time range, row count, label coverage, split sizes, positive count and per-dataset metrics. Do not pool synthetic and LANL results.
2. If LANL cannot be used, obtain a small sanitized real authentication export with documented permission and redaction.
3. If neither is possible, add a documented real-data evaluation protocol, state clearly that it has not been executed, and stop short of claiming real-world validation.

The synthetic demo can remain, but it should be labelled as a software demonstration in the evaluation summary, not only in small captions.

**Done when.** The Evaluation page has a dataset selector or an evidence-status panel, and every metric states which dataset produced it.

---

### P0-5. Rendering, demo and container safety

**Evidence.**

- `dashboard/views/digital_twin.py:554-560` inserts `path_source` into HTML with `unsafe_allow_html=True` and no escaping. Node names come from ingested CSV data. A crafted device name can inject markup or script into the page.
- `.devcontainer/devcontainer.json` starts Streamlit with `--server.enableCORS false --server.enableXsrfProtection false`. This contradicts the README's local-only security posture and is unsafe if the forwarded port is reachable.
- The "Generate demo data and run evaluation" button starts a subprocess through `src/demo_pipeline.py` with no authentication, rate limit or concurrency lock. On a shared or public host, any visitor can trigger repeated pipeline runs and overwrite shared artifacts.
- Model artifacts are loaded with `pickle`. Acceptable for local files the analyst controls; not acceptable for uploaded or third-party artifacts.

**Fix.**

- Escape `path_source` (and any other dynamic node label) before insertion, or render it with `st.text` / a component that escapes by default. Add a test with a node name containing `<img onerror=...>`.
- Remove the CORS and XSRF flags. Bind to `127.0.0.1` by default. If the demo is hosted, put it behind TLS, authentication and a reverse proxy.
- Gate demo generation behind an environment flag, a one-shot session guard and a file lock; write to a per-run temporary directory instead of shared `data/` and `models/`.
- Document the pickle trust boundary, or move to a versioned, non-executable artifact format.

**Done when.** A payload-shaped device name renders as text, the app starts without disabling browser protections, and two simultaneous demo requests cannot corrupt each other's artifacts.

---

## 4. P1 — needed for a defensible, complete prototype

### P1-1. Percentile and tiers depend on the full dataset

`anomaly_percentile` is computed over all scored events, including the holdout and any future data. Historical risk tiers therefore change when new events are added, and the holdout's displayed risk depends on the test set it is being evaluated against. For a replay or audit, this is temporal leakage in the presentation layer.

**Fix.** Compute the percentile against a frozen reference distribution stored with the model, or use an expanding window that ends at each event's timestamp. Persist the reference distribution and thresholds in the artifact manifest.

### P1-2. Window-boundary and feature semantics

- The first occurrence of a user/destination pair inside the loaded prefix is treated as "new" even if the pair existed before the window. With the default 2,000,000-row prefix, a truncated history inflates novelty near the boundary.
- The model has no time-of-day feature, but the homepage demo highlights "Unusual hour" and "Consistent hour" (`dashboard/views/home.py:97,137`). The copy is labelled illustrative, but a viewer can reasonably read it as model behaviour. Either add time features deliberately (and retrain) or relabel the demo.
- The synthetic attacks are linearly separable by novelty, failure and asset name. The first-seen baseline's win is evidence that the comparison is too easy, not that the ML models are weak in general.

**Fix.** Add explicit boundary handling and a warning when the input is truncated. Align the homepage copy with the actual feature set. Add a harder synthetic scenario or a real dataset before drawing comparative conclusions.

### P1-3. Explanation alignment is positional and fragile

The Shapley view currently matches stored scores to rebuilt features by row order. It works today (maximum absolute difference `9.7e-17` between stored and recomputed Isolation Forest scores), but any upstream sort, filter or concat will silently attach explanations to the wrong event.

**Fix.** Join on `_event_id` plus `pipeline_run_id`, assert that the joined event identity and recomputed score match, and refuse to render an explanation when the check fails. Include the feature-row hash in the explanation payload.

### P1-4. Artifact provenance stops short of the models

The evaluation report includes the input SHA-256, run ID, Python and package versions, random state and risk-engine configuration. It does **not** include model artifact hashes, the git commit, the feature-list hash or the training-data row range. `probe_model_artifacts` reports only whether a file loaded.

**Fix.** Write a manifest with SHA-256 for each model and scaler, the git commit (or a clean-tree marker), the feature list hash and the exact training/test boundaries. Display the run ID and manifest status on the Evaluation page. Write artifacts atomically (temporary file plus `os.replace`) so an interrupted run cannot leave a mixed set.

### P1-5. The "digital twin" is a device-only observed graph

The graph adds edges only from successful authentications between `device` and `target_device`. It has no account nodes, includes self-loops, and uses a hardcoded criticality table and `SENSITIVE_ASSETS = {"DB-CORE", "APP-FIN", "APP-HR"}`. Shortest paths ignore time, account continuity and edge weight. `path_risk` is computed but never displayed (`src/digital_twin.py:130`), and the "user" branch in `classify_asset` is unreachable (`src/digital_twin.py:46`).

The documentation compounds this: the About page says "observed account and device authentication links", while the graph has no accounts; the Architecture page says paths are limited to four hops, while the UI slider allows one to six (`dashboard/views/architecture.py:273`, `dashboard/views/digital_twin.py:481-487`).

**Fix.** Either add identity nodes and account-to-device edges, or rename the feature to "Observed device relationships" everywhere. Remove self-loops, display the evidence behind each path (timestamps, observation counts), use time-ordered paths if the claim is investigative, and remove the unused criticality field or show it. Align the hop limit and copy.

### P1-6. Ingestion portability and field coverage

Only the synthetic and LANL layouts are supported. `destination_user` and `auth_orientation` are read but dropped from the common schema. `source` is a constant, and the label join depends on a specific key shape. The 2,000,000-row prefix can stop before later attacks, and there is no per-dataset schema validation report.

**Fix.** Introduce a small adapter interface with explicit field mappings, a required-field check and a validation summary (rows read, rows dropped, time range, truncation flag). Preserve destination identity and orientation where the model or graph can use them. Make truncation visible in the dashboard, not only in the console.

### P1-7. Engineering and test gaps

- The 37 tests cover data preparation, model causality, walk-forward validation, risk, graph, explanations, provenance, case storage and the demo pipeline. There are no tests for dashboard rendering, HTML escaping, security boundaries, CSV formula injection, end-to-end pipeline execution, artifact manifests or queue metrics.
- `requirements.txt` omits direct imports `altair` and `joblib`; it has no upper bounds and no lockfile. There is no `pyproject.toml`, CI workflow, lint configuration or license file.
- The README's repository structure omits `dashboard/views/`, `dashboard/styles/` and `dashboard/graph_view/`; its navigation names do not match the current Start/Workspace/Intelligence/System groups.
- `dashboard/app.py:52` defines an unused `BRAND`; `dashboard/app.py:247-248` has an unreachable data-quality branch.
- `kv_card` inserts its value without escaping, which is an injection footgun even if current callers pass trusted text.
- The homepage embeds a roughly 473 KB base64 logo and a 36 KB inline graph component. Acceptable for a demo, but wasteful on every rerun.

**Fix.** Add tests for the P0 cases, a CI job running `pytest` and a linter, a pinned dependency lock, and a current README. Escape by default in `kv_card`. Remove dead code. Serve the logo as a static asset or cache it.

### P1-8. Hosted demo isolation

`run_synthetic_demo` writes into the repository's `data/` and `models/` directories. On a shared host, one visitor's run can overwrite another's artifacts mid-session, and generated case records persist across visitors. This is fine for a single-user local demo and not fine for a public one.

**Fix.** Generate into a per-run directory keyed by run ID, or precompute read-only demo artifacts and disable generation in the hosted build. Reset the case database between sessions.

---

## 5. P2 — production readiness (scope expansion)

The current project is explicitly a local research prototype. If "production-ready" is the goal, these are not polish items; they are a new system boundary.

| Area | Required work |
|---|---|
| Identity and access | SSO/OIDC, role-based access, per-analyst case ownership, session expiry, admin separation |
| Data protection | Encryption at rest for case notes and exports, secrets management, retention policy, verified backups, PII minimisation |
| Backend | API/service layer separate from Streamlit, job queue and scheduler for pipeline runs, artifact registry, database migrations |
| Observability | Structured logs, metrics, traces, health checks, error reporting, alerting, tamper-evident audit history |
| Integrations | Identity-provider and SIEM log ingestion, schema registry, export/webhook/ticketing connectors |
| Deployment | Hardened container, TLS reverse proxy, network policy, least-privilege service account, SBOM and dependency scanning, signed artifacts |
| Model governance | Calibration, drift monitoring, retraining cadence, evaluation gates, privacy review, documented model ownership |
| Scale | Columnar storage, incremental scoring, pagination, caching keyed by run ID rather than whole DataFrames |

Automated containment should remain out of scope unless it is designed with explicit approvals, rollback and an audit trail. The current human-in-the-loop position is a strength, not a gap.

---

## 6. P3 — cleanup and removals

| Path | Issue | Recommended action |
|---|---|---|
| `TeqoAI Final Proposal.docx` (36 KB) | Superseded by the new proposal; tracked in the repository | Remove from the repository; keep an external archive if needed |
| `TeqoAi logo.jfif` (1.6 MB) | Unreferenced; the app uses `assets/teqoai-logo.png` | Remove |
| `README (1).md` | Duplicate pointer document | Remove |
| `docs_assets/` | Untracked working-tree folder | Track intentionally or add to `.gitignore` |
| `data/state/synthetic_cases.sqlite3` | Local demo cases, including an owner name and notes; ignored by git but present on disk | Reset or delete before a public demo; never ship it |
| `dashboard/app.py:52` | Unused `BRAND` constant | Remove |
| `dashboard/app.py:247-248` | Unreachable data-quality branch | Remove, or add a real data-quality page |
| `src/digital_twin.py:46` | Unreachable "user" criticality branch | Remove |
| `src/digital_twin.py:130` | `path_risk` computed but unused | Remove or display it |
| `dashboard/views/about.py` | Repeats problem, contributions, scope and limitations already covered on Home, Architecture and in the README | Consolidate to one source of truth |
| `dashboard/views/home.py:97,137` | "Unusual hour" demo signal does not correspond to a model feature | Relabel the demo or add time features deliberately |
| `dashboard/views/architecture.py:273` | Says four hops; UI allows one to six | Align copy and control |
| `README.md` | Stale structure and navigation names | Rewrite the repository map and run instructions |
| `requirements.txt` | Missing direct `altair` and `joblib`; no bounds or lock | Add direct dependencies and pin a lockfile |
| `.devcontainer/devcontainer.json` | CORS and XSRF disabled | Remove the flags and bind locally by default |
| `dashboard/components.py` (`kv_card`) | Inserts value without escaping | Escape by default |
| `dashboard/views/digital_twin.py` | Duplicate success filtering between graph and view | Consolidate into one helper |

Personal contact details in `about.py` (email, Telegram, GitHub, phone) look intentional for attribution. Keep them if the page is private or the author accepts publication; remove the phone number before broad public hosting.

---

## 7. Mapping to the six INSA requirements

| INSA requirement | Current evidence | Status | Main gap |
|---|---|---|---|
| 1. Problem identification | Clear authentication-triage problem, scoped to analyst review and explainability | Met | No evidence from a real environment or practitioner workflow study |
| 2. Research and analysis | Method comparison, chronological validation, documented limitations | Partial | Synthetic-only evidence; baseline beats the primary model; no real-data analysis |
| 3. Solution development | Modular pipeline, dashboard, exact Shapley view, observed graph, local cases | Largely met | Label leakage, queue not evaluated, twin semantics, ingestion breadth |
| 4. Technical demonstration | Polished interface and a scripted demo pipeline | Met for a software demo | Public-demo safety, no real-data demonstration, no one-command reset |
| 5. Testing and evaluation | 37 passing tests; honest metric definitions; fixed-budget reporting | Partial | Wrong artifact evaluated, tiny positive counts, no confidence intervals, no UI/security/e2e tests |
| 6. Documentation | Complete proposal; extensive in-app explanations and limitations | Largely met | Stale README and inconsistent architecture/feature copy |

On innovation: the combination of exact Shapley explanations, evidence-bounded observed paths and analyst-owned case records is a solid engineering integration for a graduation project. It is not a novel research contribution, and the report should present it as a practical integration rather than claim methodological novelty.

---

## 8. Recommended sequence

| Step | Work | Effort | Exit criterion |
|---|---|---|---|
| 1 | Remove label and split columns from the serving artifact and investigator view; add the regression test | 2–3 hours | No ground-truth field can reach the UI |
| 2 | Evaluate the deployed queue; remove score clipping; add deterministic tie-break; recalibrate or rename tiers | 1 day | Evaluation page reports the score analysts see, with baseline alongside |
| 3 | Escape the path source; remove insecure container flags; gate demo generation and isolate writes | Half day | XSS test passes; app starts with browser protections; concurrent runs cannot collide |
| 4 | Run the authorized LANL sample or document why it is unavailable; add dataset-aware evidence status | 1–2 days, depends on access | At least one non-synthetic result or an explicit "not executed" protocol |
| 5 | P1 methodology: causal percentile, keyed explanations, artifact manifest, twin semantics, ingestion validation | 3–5 days | Reproducible run manifest; explanation join is keyed; docs and UI agree |
| 6 | Engineering hygiene: CI, pinned dependencies, README, dead-code removal, cleanup table | 1–2 days | Clean tree, green CI, current docs |
| 7 | P2 production work, only if that is the agreed target | Weeks | Separate design and threat model |

The first three steps are the ones that change how the project is judged. They are also the least risky to implement because they do not retrain anything.

---

## 9. Decisions that change the plan

1. **Demo target: local graduation demo or hosted public demo?** Local allows the current architecture with documented limits. Hosted requires P0-5 plus P1-8 and a reset strategy before anyone sees it.
2. **Primary ranker: keep the Isolation Forest narrative or follow the evidence?** The stored results favour the first-seen rule. Either improve the learned model or change the story. Keeping the current claim and the current table is not defensible.
3. **Real-data access: LANL run, sanitized organization sample, or protocol only?** This determines whether requirement 1 is evidenced or merely argued.
4. **Production scope: prototype or product?** P2 is a different project. The current README is right to call this a local prototype; the audit should not push it into production without a deliberate decision.

---

## Appendix: metric method

- Source: `data/processed/auth_events_risk.csv`, 1,246 rows, holdout selected by `evaluation_split == "test"` (374 rows, 33 positives).
- Average precision and ROC-AUC computed with scikit-learn on the holdout.
- "Top-30" for models is the first 30 rows after a stable descending sort by the named score.
- "Actual dashboard queue" reproduces `dashboard/filters.py:65-80`: severity order High, Medium, Low, then `risk_score` descending, then `time` descending, with all case statuses treated as New for the calculation.
- Shapley alignment check compared stored `anomaly_score_iforest` values with scores recomputed from `build_features` and the persisted model; maximum absolute difference was `9.7e-17`.
- Tests were run with the repository virtual environment: `venv/Scripts/python.exe -m pytest -q`.
