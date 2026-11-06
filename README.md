# TeqoAI — Graduation MVP

An academic prototype for explainable authentication-anomaly triage. It turns
authentication events into risk-ranked alerts, shows observed account/device
relationships and investigation context, and keeps case decisions with the analyst.
It is a local research/demo system, not a production security product.

## Project structure

```
teqoai/
├── assets/
│   └── teqoai-logo.png      ← supplied transparent TeqoAI emblem
├── data/
│   ├── raw/                  ← synthetic inputs; LANL inputs go in raw/lanl/
│   └── processed/            ← dataset-profile outputs and evaluation report
├── src/
│   ├── __init__.py            ← importable pipeline package
│   ├── generate_synthetic_data.py
│   ├── data_prep.py           ← parse authentication logs into a common schema
│   ├── model.py               ← Isolation Forest and optional autoencoder
│   ├── evaluation.py          ← held-out ranking metrics and label coverage
│   ├── provenance.py          ← dataset/run consistency checks
│   ├── risk_engine.py         ← transparent rule-assisted risk score
│   ├── digital_twin.py        ← successful-auth relationship graph and paths
│   ├── dataset_paths.py       ← synthetic/LANL output isolation
│   └── case_store.py          ← local analyst case records
├── dashboard/
│   └── app.py                 ← welcome page and Streamlit analyst workspace
├── tests/                     ← data, model, graph, case, and provenance checks
├── run_demo.ps1               ← reproducible local graduation demo
├── requirements.txt
└── README.md
```

## Setup

```bash
cd teqoai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: Get the dataset

You have two options — start with the synthetic one so you're unblocked today, swap in the real one whenever it's ready.

### Option A — Synthetic data (instant, no download needed)

From the project root, run the complete reproducible demo:

```powershell
.\run_demo.ps1
```

To run it and launch the dashboard in the same terminal:

```powershell
.\run_demo.ps1 -LaunchDashboard
```

The script uses the project's `venv` when available (otherwise `python` on
`PATH`), selects the synthetic profile for its child processes, and restores
your previous `TEQO_DATASET` setting when it exits. It regenerates the synthetic
inputs and overwrites the synthetic processed outputs and model artifacts.

Run module commands from the repository root. The `src` package provides a
single import path for the pipeline, dashboard, and tests:

```bash
python -m src.generate_synthetic_data
python -m src.data_prep
python -m src.model
python -m src.risk_engine
```

This generates `data/raw/auth.txt.gz` and `data/raw/redteam.txt.gz` using the LANL column format, but the records are synthetic—not LANL data. It includes normal logins for 60 users, benign-but-unusual activity, and labeled lateral-movement chains. Chains are deliberately placed in both the early training window and later holdout window, so the evaluation workflow can be demonstrated locally. Results validate the pipeline only, not detection quality on LANL or real networks. The generator is seeded (`random.seed(42)`) for repeatable output.

### Option B — Real LANL data

Official page: **https://csr.lanl.gov/data/cyber1/**

The page has a short form near the bottom — enter your email and a one-line description of your use (e.g. *"Academic prototype for a cybersecurity project"*) to get the download links for `auth.txt.gz` and `redteam.txt.gz`. The official description says the complete five-file dataset is about 12 GB compressed. The files are gated by the access form; processing remotely still requires a server/runtime to fetch them.

Put the two files in `data/raw/lanl/`. This keeps them separate from the synthetic sample in `data/raw/` and prevents the LANL run from overwriting the synthetic outputs.

### Important: don't load the whole file at once

`auth.txt.gz` is enormous. `src/data_prep.py` streams the gzip file and stops after a bounded prefix (default: first 2,000,000 rows). To process a larger prefix, set `TEQO_MAX_AUTH_ROWS` before data preparation; choose a size your machine can handle and verify the resulting time range and held-out label coverage before interpreting metrics.

In PowerShell, select the LANL pipeline before running each step:

```powershell
$env:TEQO_DATASET = "lanl"
$env:TEQO_MAX_AUTH_ROWS = "20000000" # optional; default is 2000000
python -m src.data_prep
python -m src.model
python -m src.risk_engine
python -m streamlit run dashboard/app.py
```

The clean, scored, and risk files and model artifacts will be stored under `data/processed/lanl/` and `models/lanl/`. Keep `$env:TEQO_DATASET` set to `lanl` when launching the dashboard to view those outputs. Unset it (or set it to `synthetic`) to use the synthetic sample instead.

### Testing with organization data

The current data-preparation code accepts the LANL `auth.txt.gz` layout; it
does **not** directly ingest arbitrary organization exports. The supported
profiles are `synthetic` and `lanl`. Do not point an organization's logs at the
LANL profile and treat the resulting report as an organization evaluation.

For an authorized, isolated research test:

1. Get written approval from the data owner for the exact fields, time period,
   storage location, processing, and whether results may be shown to INSA.
   Never use live collection or upload raw logs to a public service or repository.
2. Export a small, time-bounded sample to a protected local environment. Remove
   or consistently pseudonymize account and device identifiers, and restrict
   access to the raw export and `data/state/` case database.
3. Build and validate an organization-specific adapter that maps records to
   `source`, `time`, `user`, `device`, `target_device`, `action`, `result`, and
   `is_malicious`; add a separate dataset profile so its outputs and case notes
   cannot overwrite the synthetic or LANL results. Configure sensitive assets
   and confirm the source log's authentication semantics before interpreting
   the rules or graph.
4. For a meaningful evaluation, populate `is_malicious` only from independently
   confirmed incidents or approved red-team ground truth. Keep labels out of
   model features and preserve a later chronological period as a holdout.
   Uninvestigated events are **unknown**, not known-benign examples. The current
   pipeline expects binary labels, so it must be extended to handle unknown
   labels before evaluation if the export does not have reliable labels.
5. Run the pipeline only after that adapter/profile work, then report the
   time range, event counts, labeled-positive counts in train and holdout,
   precision/recall at the review budget, and comparison with the simple
   first-seen-destination baseline. If the holdout has too few confirmed cases,
   present the run as an analyst-reviewed exploratory pilot—not an accuracy
   result.
6. For the INSA presentation, use organization-derived screenshots or metrics
   only if the owner explicitly approved disclosure. Otherwise present the
   synthetic demo or authorized LANL results, and describe the organization
   pilot as proposed future validation.

### Graduation MVP scope

- **Research question:** can unsupervised event anomaly rankings help prioritize
  known red-team authentication events in a chronological holdout?
- **Prototype contribution:** event-level features, Isolation Forest and an
  optional autoencoder comparison, transparent rule-assisted risk, and analyst
  investigation context including observed device paths.
- **Demonstration evidence:** the synthetic sample exercises the pipeline and
  evaluation code only. Its generated attack chains are intentionally split
  across train and holdout; its metrics are not evidence of LANL or real-world
  effectiveness.
- **Out of scope:** production monitoring, live integrations, automated
  containment, and claims that every unlabeled event is benign.

For a final submission, state the dataset used, the exact split and label
coverage, the generated evaluation report, and these limitations. Real LANL
validation is a future extension if the authorized dataset becomes available;
it is not required to describe the current output honestly as a synthetic-data MVP.

## Step 2: Run the data prep script

```bash
python -m src.data_prep
```

This produces `data/processed/auth_events_clean.csv` in the common schema:

`source | time | user | device | target_device | action | result | is_malicious`

Where:
- **source** — the log source (`windows_auth`, for now the only source)
- **time** — elapsed event seconds (LANL's calendar epoch is undisclosed)
- **user** — the user account that initiated the authentication (`source_user@domain`)
- **device** — the device the user authenticated *from*
- **target_device** — the device the user authenticated *to* (needed for the attack-path graph later)
- **action** — authentication type + logon type combined (e.g. `Kerberos/Network`)
- **result** — `Success` or `Failure`
- **is_malicious** — `1` if this event matches a labeled red-team event, else `0` (your ground truth for evaluation — the model itself must NOT be trained on this column, since it's meant to be unsupervised/semi-supervised)

Feature construction orders events deterministically. For an account, rolling
one-hour activity uses only strictly earlier timestamps plus the current event;
events tied at the same timestamp do not increase one another's counts. All
events tied for the first observed account/destination timestamp receive the
first-seen feature. Authentication labels are never model inputs.

## Step 3: Train and evaluate

```bash
python -m src.model
```

The model learns action categories from training events only. It trains on the
earliest 70% of distinct event times and reserves the latest 30% as a final
chronological holdout. Before that final holdout, the first-seen-destination
baseline, Isolation Forest, and optional autoencoder are compared on the same
expanding chronological validation folds. Preprocessing scalers and action
categories are learned from the applicable training period only.

The autoencoder is a small dense reconstruction baseline, not a claim that deep
learning is superior. It uses a training-only standard scaler, fixed-epoch
mini-batch mean-squared-error training, seeded initialization/shuffling, and
bounded-batch inference. Its checkpoint and scaler are saved separately for
reproducible inference. Its reconstruction scores are rankings; there is no
calibrated compromise probability or selected decision threshold.

The pipeline writes `data/processed/evaluation_report.json` and a shared
`pipeline_run_id` into the scored and risk files. The dashboard verifies the
input SHA-256, dataset profile, and run identifier before presenting metrics.
The report includes split/label coverage, package versions, average precision,
ROC-AUC when both classes are present, and precision, recall, F1, confusion
counts, and false-positive rate at fixed top-30 and top-100 alert budgets.
Windows without known positives are marked not evaluable for detection quality;
their false-alert counts/rates are still reported when possible.
After a feature or evaluation schema change, rerun the model pipeline before
using the previously generated scores, evaluation report, or model artifacts.

The dashboard's **Results & Method** page presents final holdout and
walk-forward results, labels small validation samples, and identifies stale
or mismatched pipeline artifacts. Risk scores are prioritization signals, not
calibrated probabilities; holdout metrics evaluate raw model rankings rather
than the rule-adjusted risk score.

Evaluation is against exact red-team label matches. Events without a match are
treated as negatives for calculation, but may include malicious behavior not
covered by the labels; metrics are therefore agreement with known labels, not
proof that all alerts or non-alerts are correct.

Run the full local demo and dashboard with:

```powershell
.\run_demo.ps1 -LaunchDashboard
```

The dashboard binds to `127.0.0.1` by default. To run only the focused checks:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The pipeline modules are run with `python -m src.<module>` from the repository
root. This keeps imports consistent across scripts, dashboard code, and tests.

## Dashboard navigation

The app opens on a project welcome page with creator information, project
overview, and links into the analyst workspace. The top navigation separates
the **Home**, **Overview**, **Investigate**, **Observed Paths**, and
**Results & Method** pages. Home can be viewed without generated event or model
artifacts; the analyst workspace requires the local pipeline outputs described
above.

## Current implementation status

- [x] `src/data_prep.py` — map authentication events into the common schema
- [x] `src/model.py` — Isolation Forest and optional autoencoder scoring with chronological evaluation reporting
- [x] `src/risk_engine.py` — transparent Low/Medium/High risk scoring
- [x] `src/digital_twin.py` — observed-relationship graph, path analysis, and SHAP helpers
- [x] `dashboard/app.py` — branded welcome page and top navigation, plus an
  analyst dashboard with search, filters, investigation, evaluation evidence,
  and human-controlled triage
- [ ] Validate model quality on downloaded LANL data with a held-out time period (future extension; not claimed by this synthetic MVP)

Run the full pipeline using the commands above, then launch the welcome page
and analyst workspace:

```bash
python -m streamlit run dashboard/app.py
```

The observed-path graph is built only from successful authentication events.
Failed attempts are not treated as confirmed device relationships, and even a
successful authentication is not proof of general network reachability or
authorization.

Case status, analyst assignment, notes, checklist progress, and a change audit trail are
stored unencrypted locally in `data/state/` using SQLite. Do not enter sensitive
production data; protect and back up this directory according to your organization's
handling requirements. TeqoAI presents recommendations for analyst review and does not
automate response actions.
