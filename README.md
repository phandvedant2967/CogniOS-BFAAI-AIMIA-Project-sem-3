# Final fixes (latest): inference, signal categories, event-level trends, small-cell control

**Inference.** XGBoost no longer needs PyTorch: the saved sequence-builder pickle used to import `torch` at module level, so a machine
without torch failed to load even the XGBoost model. The import is now guarded (`ml/sequence_interface.py`); the CNN-GRU model still
needs torch and reports a precise 503 message if it is missing. Verified in this build: real XGBoost predictions with TreeSHAP
attribution on imported patients in the running app (`tests/e2e_model_ui.py`). **CNN-GRU could not be executed in the build
environment (PyTorch cannot be installed there): it is unverified.** Run `python tests/e2e_model_ui.py` on your machine.

**Manufacturer page.**
- Signal categories per drug are back (five record-based statuses, one per patient, with scope tabs: all events / starts / dose changes / no overlap).
- The cognitive trend is anchored at EVERY recorded start and dose change, not only the first start. A patient contributes one value per window.
  Overlapping medication changes and clinical events are never attributed to one drug; the "No overlap" scope shows events with nothing recorded nearby.
- Small-cell and differencing control (`app/mfr_disclosure.py`): minimum 5 distinct patients, plus rules that withhold a count when it could be
  subtracted from another view (removing a filter, widening the date range, other core measures) to reveal a group of 1-4 patients. Date ranges
  are widened to whole calendar quarters. Coverage counts are rounded down to a multiple of 5 and shown with "+". Some cells show "withheld"
  (protected) instead of "<5". `tests/mfr_attack.py` is an integer-program attack audit run by `tests/test_mfr_disclosure.py`.
- Limits: the audit covers the core measures with static filters; it does not prove safety against every possible combination of released numbers.
  No statistical significance is computed: observed differences are descriptive, not causal. There is no export feature.

# Explanations and the test CSV (latest)

## What is explained, and how
**A. Observed medication-safety signal (record-based, no model).** For every medication start and dose change the page shows the
drug, dose before/after, baseline and follow-up MMSE with their dates, the difference, how many assessments fell in each window,
the reason the status was assigned (citing the actual records), nearby clinical events and other medication changes as possible
alternative explanations, and a "Missing information" list (e.g. no baseline, how many more follow-up scores are needed, dose
not recorded). This is a rule on the patient's own scores (see the workflow section); it is never described as the model's reasoning.

**B. Model prediction (separate section).**
- *XGBoost*: exact **TreeSHAP** from XGBoost itself (`Booster.predict(pred_contribs=True)`; no extra package). It explains the
  predicted class's raw score (log-odds, not probability or risk). Contributions of a feature's value column and its missing-value
  indicator are summed and mapped back to the 125 named model inputs, shown with this patient's computed value, direction
  (raises/lowers) and input type. Every explanation is checked: contributions + bias must equal the booster's own raw margin
  (tolerance 1e-3); otherwise **no attribution is shown** and the reason is displayed. Inputs the record could not supply (filled
  by the saved preprocessor with training medians) are labelled "not computable from the record".
- *CNN-GRU*: **no attribution is implemented.** The page shows the prediction, the patient-record evidence, and states that
  model-specific attribution is unavailable.
- Class wording: disease_progression / other_cause / uncertain are described as labels the model cannot separate reliably in
  this synthetic data (existing project finding). If the top two raw scores are within 0.10 this is stated. When several top
  inputs are drug-type properties rather than the patient's scores, that is stated too.
- Feature contribution is not causation; scores are uncalibrated and are never shown as clinical-risk percentages.

## Test CSV
`sample_data/cognios_sample_ehr.csv` (also on the Import page and at `/api/import/sample.csv`): 8 synthetic patients
(`SAMPLE_P001`..`P008`), 75 data rows (8 patient, 11 comorbidity, 11 medication, 2 dose_change, 38 assessment, 5 event rows).
Only values the importer accepts are used (frequency "Once daily", unit "mg", real medication names/IDs, brand name Aricept
mapped to Donepezil with the original kept). The signal engine's outcome on it is 5 potential signals, 1 follow-up needed,
3 insufficient evidence, 2 improvement, 4 no measurable decline. These were hand-checked and are asserted in the tests. Nothing
is tuned to make the AI model output a particular class. To use it: sign in as a clinician, open Import EHR data, choose the
file, Validate, Confirm, then look up `SAMPLE_P001`. Re-uploading the same IDs is rejected (imports never overwrite).

# CogniOS: uploaded-EHR workflow (current)

This section supersedes the older stage notes below wherever they disagree (training-patient lookup, Model Evaluation, the
CSV importer for one patient, the old manufacturer cohort view).

## Workflow
1. **Import EHR data** (`/import`): upload a CSV with one or many patients. The whole file is validated first; any error rejects
   it and nothing is written. Download the working sample from the page (`/api/import/sample.csv`, also
   `sample_data/cognios_sample_ehr.csv`; 8 clearly synthetic patients, 75 rows, tested through the real importer).
2. **Patient review** (`/`): search an *uploaded* patient ID. Training-dataset patients are not reachable. An unknown ID gives a
   not-found message; with nothing uploaded the page says to import a CSV first. No demo patients are ever inserted.
3. **Medication Safety Signals** (top of the patient page) are computed only from that patient's records, using the project's
   rules: baseline = latest MMSE in the 270 days up to the event; follow-up = latest MMSE in the 180 days after; at least 1
   before and 2 after; a change of 1.5 points or more is a decline or improvement. Statuses: Potential Medication-Related
   Signal, Cognitive Improvement Observed, No Measurable Decline Observed, Insufficient Evidence, Follow-up Needed. A signal is a
   temporal association for clinical review, never a confirmed adverse drug reaction, and the app never advises stopping or
   changing a medicine.
4. **Model prediction** is a separate section. The saved v4.2 model (unchanged) is run on the uploaded record with the model's own
   preprocessing and feature order. Missing required inputs or schema columns produce an explicit "no prediction" result; signals
   do not depend on it.
5. **Manufacturer** (`/manufacturer`): aggregates from uploaded records only, with the minimum-5-unique-patient suppression. Manufacturers can also upload an EHR CSV themselves (see *Manufacturer upload and drug analytics* below).

Uploads go to `data/uploaded_ehr.db` (git-ignored). The training databases and `models/` are never opened by the web app or
modified. Model Evaluation was removed.

## CSV format
Long format, one row per record, columns in `record_type` order: `patient`, `comorbidity`, `medication`, `dose_change`,
`assessment`, `event`. Every row has `patient_id`; each patient needs exactly one `patient` row. Dates are `YYYY-MM-DD`.
Medicines: give `medication_id`, `medication_name`, or both (brand names such as Aricept map to the generic; the name you wrote
is kept; unknown names are rejected). Dose changes: either inline on the medication row (`dose_change_date` + `new_dose`) or as
separate `dose_change` rows (any number). Allowed values (sex, diagnosis, stage, frequency, units, event types ...) are listed
on the import page and are the strict values the model's data uses.

## Limitations
- The model's anchor is the medication start or the *first* recorded dose change; the signal engine considers every dose change.
- Enums are strict (for example frequency is only "Once daily", dose unit only "mg") because the model was trained on those.
- Unmapped drug names are rejected rather than guessed.
- The model's non-medication classes are weak; its output is a pattern flag, not a diagnosis.
- Real XGBoost / CNN-GRU inference could not be run in the sandbox this was built in (xgboost and torch could not be
  installed). The model-plumbing tests use a stub classifier; tests marked REAL_model run the saved artifacts and are skipped
  where they cannot load. Run `pytest` and `node --test tests/js/*.test.js` locally with the full requirements to exercise them.

# EHR Data Foundation (Flask + SQLite)

Step 1 of the medication-safety / cognitive-risk project: store the 5 synthetic datasets as related SQLite tables and expose read-only retrieval endpoints. **No ML, no causal logic, no review logic** — data storage and retrieval only.

## Setup & run

```bash
python3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu

python run.py                     # http://127.0.0.1:5000
python -m pytest -q               # full test suite
```

Requires **Python 3.12+** (the pinned `xgboost==3.4.1` needs it; `scikit-learn==1.9.1` must match
the saved model artifacts). The first `python run.py` creates the demo `clinician` and `manufacturer`
accounts; both sign in with the fixed demo password (see **First login**).
Pre-built databases are included (`data/ehr.db`, and `data/v4_2/ehr.db` used by the clinician app).
Do not run `scripts/import_data.py` unless you mean to rebuild `data/ehr.db` from scratch.

Try it:

```bash
curl http://127.0.0.1:5000/api/patients/P0001
curl http://127.0.0.1:5000/api/patients/P0001/timeline
curl http://127.0.0.1:5000/api/drugs/Sertraline
```

## Running on Windows

Tested status: the dependency set was resolved for Windows (all packages have Windows wheels for
Python 3.12), but the app has **not been run on a Windows PC** by the developer. Report any problem
using the troubleshooting list below.

**Requirements:** Windows 10/11, 64-bit, **Python 3.12** from https://www.python.org/downloads/
(keep "py launcher" ticked in the installer). About 1.5 GB free disk space for the virtual environment.
An internet connection is needed once, for setup. Other Python versions are not supported: the saved
models need `xgboost==3.4.1` (Python 3.12+) and exactly `scikit-learn==1.9.1`.

1. Unzip `CogniOS-final.zip`, for example to `C:\CogniOS-final`. Use a short path without `#` or `%`.
2. Double-click **`setup_windows.bat`** (one time only). It creates `.venv` and installs
   `requirements.txt`. This takes several minutes because torch is large.
3. Double-click **`start_windows.bat`**. Wait for `Running on http://127.0.0.1:5000`.
4. Open **http://127.0.0.1:5000** in your browser. **Keep the black server window open** while you use
   the app; closing it (or pressing Ctrl+C in it) stops CogniOS.

**First login (demo only).** Choose a role on the sign-in page, then enter the username `clinician` or
`manufacturer` and the password `Vedant@123` (the same for both). The password is one server-side value
(`DEMO_LOGIN_PASSWORD` in `config.py`, overridable with the `COGNIOS_DEMO_PASSWORD` environment variable)
and is compared directly, with no hashing. It is never sent to the browser or printed. **This is an
intentional demo-only setting and is not secure for public deployment.** Each role still reaches only
its own pages and APIs.

**Importing an EHR file.** Import EHR works in two visibly separate stages: **Validated** (nothing is
saved; counts, per-patient drug and health summary, data gaps) and **Imported** (saved in one atomic
transaction in `data/uploaded_ehr.db`, then read back so the stored counts shown are the real ones). Past
imports are listed from the database on the Import page, and imported patients stay available in Patient
Review after navigation, refresh and restart. The summary shown is rule-based; it is not a model
prediction, and missing data (for example no dose changes) is stated, never inferred.

**Run the tests (optional):** `.venv\Scripts\python.exe -m pip install pytest`, then
`.venv\Scripts\python.exe -m pytest -q`. A real-browser end-to-end check (needs `playwright` and Chromium) is `python tests/e2e_browser.py`. The two JavaScript test files are skipped unless Node.js is installed.

**Troubleshooting**
- `"py" was not found` / `Python 3.12 was not found`: install Python 3.12 (64-bit) from python.org
  with the py launcher, then run `setup_windows.bat` again.
- `pip install` fails part-way: check the internet connection and run `setup_windows.bat` again (it
  reuses `.venv`). If that does not help, delete the `.venv` folder and run it again.
- Error mentioning `InconsistentVersionWarning` or a model failing to load: the wrong scikit-learn or
  xgboost version is installed. Delete `.venv` and rerun `setup_windows.bat`.
- `Address already in use` / port 5000 busy: another program (or a second CogniOS window) is using the port.
  Close it and start again.
- Torch fails to import with a DLL error: install the Microsoft Visual C++ Redistributable (x64)
  from Microsoft, then restart the server.
- Sign-in fails: use `Vedant@123` (or the value of `COGNIOS_DEMO_PASSWORD` if you set it).
- To start completely fresh (all accounts, AI predictions and doctor reviews removed), stop the server and
  delete `data\cognios_app.db` and `data\.cognios_secret_key`.

**What is not in the ZIP:** the older dataset versions used during development (`data/v2`, `data/v3`,
`data/v4`, `data/v4_corrected`, `data/stress_test`, `data/processed`). The app and tests do not use
them. Only the old training and audit scripts in `scripts/` for those versions need them.

## Project layout

```
config.py                  paths (DB, raw data); env overrides EHR_DB_PATH / EHR_RAW_DIR
run.py                     starts Flask
scripts/import_data.py     CLI: builds the database, prints a row-count check
app/__init__.py            app factory
app/schema.sql             table definitions, keys, indexes
app/importer.py            loading + cleaning logic
app/db.py                  per-request SQLite connection
app/queries.py             plain data-access functions (reusable by the ML step)
app/timeline.py            build_patient_timeline(conn, patient_id)
app/routes.py              API endpoints
data/raw/                  the 5 original files, untouched
data/ehr.db                SQLite database
```

## Tables

| Table | Primary key | Foreign keys | Source |
|---|---|---|---|
| `patients` | patient_id | – | dataset 1 |
| `patient_comorbidities` | (patient_id, comorbidity) | patient_id → patients | dataset 1 (comorbidities split) |
| `drugs` | medication_id | – | dataset 5 (+ dataset 2 `medication_reference`, identical) |
| `medication_history` | medication_record_id | patient_id → patients, medication_id → drugs | dataset 2 |
| `cognitive_behavioral` | cognitive_record_id | patient_id → patients | dataset 3 |
| `clinical_events` | clinical_event_id | patient_id → patients | dataset 4 |

Full column definitions: `app/schema.sql`.

## Endpoints (all GET)

| Endpoint | Returns |
|---|---|
| `/api/patients/<id>` | patient record |
| `/api/patients/<id>/medical-history` | diagnosis, stage, comorbidity list, dates |
| `/api/patients/<id>/medications` | medication episodes (joined with drug name/class) |
| `/api/patients/<id>/cognitive` | cognitive/behavioral assessments |
| `/api/patients/<id>/events` | clinical events |
| `/api/patients/<id>/timeline` | all dated records merged chronologically |
| `/api/drugs` | all drugs |
| `/api/drugs/<id or name>` | one drug, e.g. `M005` or `Sertraline` |

Patient IDs are case-insensitive. Unknown IDs return 404 with a JSON error.

## Data handling rules

- **Dates**: stored as `YYYY-MM-DD` text. Clinical-event dates arrived as `YYYY-MM-DD 00:00:00`; the time part (always midnight) is dropped.
- **Missing values**: empty cells become `NULL`. No rows are removed; the importer verifies source vs loaded row counts and foreign keys.
- **Literal `'None'`**: appears in the source as a real category (`drugs.anticholinergic_burden`, `cognitive_behavioral.trajectory_context`, `patients.comorbidities`). It is kept as the text `'None'`, not converted to NULL. For comorbidities, `'None'` is kept in `patients.comorbidities` but produces no row in `patient_comorbidities`.
- **Source labels kept as given**: `trajectory_context`, `adherence_status` (`Unknown` stays `Unknown`), `cognitive_impact_possible`, etc. are stored as-is and not interpreted.
- **Normalization**: `drug_name`, `drug_class`, `therapeutic_area` were repeated in dataset 2 and verified identical to dataset 5, so they are stored once in `drugs` and joined in queries.
- No values are derived, imputed or inferred.

## Timeline

`build_patient_timeline(conn, patient_id)` in `app/timeline.py` merges: diagnosis, enrollment, medication start / dose change / stop, cognitive assessments, clinical events and their resolution dates. Entries are sorted by date; same-day entries use a fixed display order, which is only a tie-break and implies nothing about sequence or cause. Each entry has `date`, `entry_type`, `source_table`, `source_id`, `summary`, `details` (the full source record).

## Stage 2: ML data preparation

`python scripts/preprocess_ml.py` reads `data/ehr.db` (read-only) via `app/queries.py` and `build_patient_timeline()`, and regenerates:

- `data/processed/patient_observations.csv` – one row per cognitive assessment / clinical event, date-ordered per patient, with day offsets (since diagnosis, enrollment, last medication start / dose change / stop of any drug).
- `data/processed/medication_episodes.csv` – one row per medication record, with day offsets between start, dose change and stop.

Dates are ISO; missing values stay blank; nothing imputed or dropped. No labels are used or created, and `trajectory_context` is deliberately not carried over. The script prints row counts and descriptive data checks on each run. No model training.

## Stage 2B (ground truth audit)
`python scripts/stage2b_ground_truth.py` writes `data/processed/ground_truth_labels.csv` and prints the leakage audit. See `STAGE2B_REPORT.md`. No model is trained and raw data is untouched.

## Stage 7: CogniOS doctor dashboard

```bash
pip install -r requirements.txt
python run.py            # open http://127.0.0.1:5000/
```

Look up a patient by ID (IDs in the assessment database look like `V4_2_P0001`), review history and timeline, pick a medication, run an XGBoost (default) or CNN-GRU assessment, and record a review.

- `app/dashboard.py`: serves the page and `GET /api/dashboard/patients/<id>`, a read-only bundle (profile, medications, assessments, events, timeline) built with the existing `app.queries` and `app.timeline` functions.
- It reads `ML_DB_PATH` (the V4.2 database), the same database `POST /api/predictions` uses. The original `/api/patients/...` routes read `data/ehr.db`, which holds different patients (`P0001`...), so they are not used by the dashboard.
- `app/static/`: `index.html`, `css/cognios.css` (design system, see `docs/DESIGN_SYSTEM.md`), `js/logic.js` (pure logic), `js/render.js` (pure renderers), `js/app.js` (controller). No build step.
- Doctor reviews are held in the page only. Nothing is written to the server; a JSON download is offered.
- Tests: `python -m pytest tests` (API and integration), `node --test tests/js/dashboard.test.js` (logic and renderers; also run by pytest when Node is installed).


---

## Stage 8–9 additions

**Install (versions matter):** the saved models were written with scikit-learn **1.9.1**, which is pinned in `requirements.txt`. `pip install -r requirements.txt`.

**Pages**

| URL | Audience |
|---|---|
| `/login` | Sign in (both roles) |
| `/` | Clinician dashboard (patient review, AI assessment, doctor review) |
| `/manufacturer` | Manufacturer dashboard (aggregated, de-identified safety signals) |

**Application database** `data/cognios_app.db` (override with `COGNIOS_APP_DB_PATH`) is created on first run. It holds logged AI predictions and doctor reviews. The EHR databases and datasets are only ever opened read-only. The shipped project contains no stored predictions or reviews, so the manufacturer dashboard starts with an empty state until clinicians run assessments.

**Endpoints added**

| Endpoint | Purpose |
|---|---|
| `POST /api/reviews` | Save a doctor assessment (`patient_id`, `review_category`, optional `clinical_note`, `medication_id`, `prediction_id`). The server sets the timestamp. |
| `GET /api/reviews?patient_id=&medication_id=&prediction_id=&limit=` | List saved reviews, newest first |
| `GET /api/reviews/<id>` | One review |
| `POST /api/predictions` | Unchanged request; response now also has `record_type: "ai_prediction"`, `prediction_id`, and `anchor_details.previous_dose / new_dose / dose_unit` |
| `GET /api/manufacturer/filters` | Filter options (only drugs with at least 5 patients) |
| `GET /api/manufacturer/summary?drug=&from=&to=&diagnosis=&stage=&sex=&outcome=` | Aggregated, de-identified signals; groups under 5 patients are suppressed |
| `GET /api/manufacturer/sources` | Which datasets have data (`manufacturer` = own upload, `clinician` = clinician imports) |
| `POST /api/manufacturer/upload/validate` , `/commit` , `/reset` , `GET /upload/status` | Upload the same EHR CSV as the clinician import; same validator and atomic commit; aggregate-only replies |
| `GET /api/manufacturer/drug-analytics?source=&drug=&from=&to=&diagnosis=&stage=&sex=&outcome=` | Summary plus per-drug analytics (patients, outcomes, trend, subgroups, findings) |

Doctor reviews are returned as `record_type: "doctor_assessment"`; the AI prediction a review refers to is a separate `prediction_reference` object.

**Tests:** `python -m pytest` (Python and, via wrappers, the Node tests; Node.js is optional).

---

## Stage 9: sign-in and roles

Every page and API now needs a signed-in user. Two roles share one app, one layout and one sidebar (`app/static/js/shell.js` adds the signed-in user and **Sign out** to both dashboards):

| Role | Lands on | May use |
|---|---|---|
| `clinician` | `/` | clinician dashboard, `/api/dashboard/*`, `/api/predictions`, `/api/reviews*`, the original `/api/patients/*` and `/api/drugs*` routes |
| `manufacturer` | `/manufacturer` | the manufacturer page and `/api/manufacturer/*` only (aggregated, de-identified, groups under 5 suppressed) |

Anything else is refused: not signed in → pages redirect to `/login`, APIs return 401; wrong role → 403. The rules are one table (`POLICY` in `app/auth.py`) and default to deny, so a new route is closed until it is listed. `tests/test_auth.py` fails if a route is left unclassified.

**First run:** `python run.py` creates a `clinician` and a `manufacturer` demo account that share the fixed demo password `Vedant@123` (demo only; see `config.py`). Add a user with `python scripts/create_user.py --help`. Users, sessions and login failures live in the existing application database (`data/cognios_app.db`).

**Security details:** fixed demo password compared directly (no hashing, demo only); server-side sessions revoked on logout, 30 min idle / 8 h absolute timeout; `HttpOnly` + `SameSite=Lax` cookie; CSRF token required on every POST (sent by the page scripts); 5 failed logins per username or 20 per address in 15 min locks sign-in for the window; `next=` redirects are restricted to pages the role may open; `no-store` caching and a strict Content-Security-Policy. The session signing key comes from `COGNIOS_SECRET_KEY`, or is generated once into `data/.cognios_secret_key`. Set `COGNIOS_COOKIE_SECURE=1` when serving over HTTPS (automatic on Vercel). The Flask debugger is now off unless `FLASK_DEBUG=1`.

**Not covered:** no self-service sign-up, password change page, MFA or password hashing (the demo password is not secure for public deployment); failed-login limits use the connecting address (behind a proxy, configure forwarded headers first); the differencing risk from many filtered manufacturer queries is reduced to authorised users, not removed.


## Stages 10–14: final update (clinician workflow)

Everything here runs on **synthetic data**. Nothing in CogniOS is clinically validated.

### What a clinician can do
1. **Patient review** (`/`): look up an ID (e.g. `V4_2_P4791`, or open `/?patient=ID`). The page has
   four sections: *Patient overview*, *What changed* (at most 3 findings backed by the record),
   *Medication history* (doses, start/stop/dose-change dates, MMSE before and after each event) and
   *AI assessment*.
2. **Run assessment**: the saved XGBoost model (default, chosen on validation macro-F1) returns one
   category plus a three-part explanation built from that patient's own rows: what the record shows,
   what the model predicts, and what is still uncertain. Raw class scores sit in a collapsed
   *Advanced details* section, labelled as uncalibrated model scores, not probabilities of harm.
   Patients from the training or validation split are flagged there, because their results are optimistic.
3. **Record a clinician review**: this works as before, and reviews are stored separately from AI predictions.
4. **Clinical Analytics** (`/analytics`): unique patients per medication, and observed MMSE trajectories.
   The trajectory rule: baseline = latest MMSE in the 270 days up to the start; follow-up = latest MMSE in
   the 180 days after. A change of 1.5 points or more counts as declined or improved; anything
   smaller is stable/mixed; a missing baseline or follow-up is *insufficient evidence*. The page has
   medication and date filters.
5. **Model Evaluation** (`/evaluation`): XGBoost vs CNN-GRU on the saved patient-level `final_test` split
   (n = 752). Metrics are recomputed with scikit-learn from the confusion matrices saved in
   `results/v4_2_metrics.json`. Per-patient test predictions were never saved.

   | | Accuracy | Macro-F1 | Macro-F2 | Medication-associated vs rest P / R / F2 |
   |---|---|---|---|---|
   | XGBoost | 0.342 | 0.342 | 0.342 | 0.583 / 0.580 / 0.580 |
   | CNN-GRU | 0.364 | 0.364 | 0.364 | 0.551 / 0.521 / 0.527 |
6. **Import patient** (`/import`): upload one UTF-8 CSV per new patient, 1 MB at most. Download the
   template on that page, which lists every field. The upload is validated first and shown as a
   preview with any errors. Then the whole file is written in one transaction, or nothing is written.
   Imported IDs must start with `IMP_` and are recorded in an `imported_patients` table. They go
   through the same preprocessing and model as the existing cohort; with too few assessments around a
   medication start, the app says so and shows no prediction.

   **Where imported (`IMP_`) patients appear:**

   | Place | Imported patients |
   |---|---|
   | Patient review and AI prediction | included |
   | Clinician reviews | included |
   | Clinical Analytics charts | excluded (synthetic cohort only) |
   | Manufacturer cohort table | excluded |
   | Manufacturer "AI flags" and "Clinician reviews" aggregates | included, because these count assessments clinicians actually ran |
   | Model Evaluation | unaffected; it uses only the saved `final_test` results |

   **Undoing imports:** imports are written into `data/v4_2/ehr.db`, which is tracked by git.
   `git restore data/v4_2/ehr.db` puts back the committed file and removes **all** imported patients at
   once; there is no per-patient undo. A verified copy taken before any import is at
   `../CogniOS-stage9-backups/v4_2_ehr_2026-10-21_pre-import.db`. Stop the app before restoring.

### Manufacturer upload and drug analytics
A manufacturer can upload the **same EHR CSV** the clinician import accepts (same template, same validator, same atomic commit and read-back).
The dataset is stored in its own file with the same schema (`data/manufacturer_ehr.db`, override with `MFR_UPLOAD_DB_PATH`), so it never shows up in
clinician Patient Review and "Remove my dataset" never touches clinician records. The dashboard's **Data source** selector switches between
the manufacturer's upload and clinician-imported records.

Everything shown is computed from the uploaded records with the existing record-based signal rules (`app/signals.py`) and is aggregate-only:
patients per drug, patient-level decline / no change / improvement, signal categories, mean MMSE and domain changes, MMSE in time windows around
the first start, clinical events recorded afterwards, and decline share by diagnosis, disease stage and age group. Every count needs 5+ unique
patients, small groups are pooled, a lone hidden subgroup count triggers a second hidden one, and means are withheld next to a hidden count.
Upload replies contain counts and validation messages only; patient IDs in messages are replaced by "a patient".
Findings are *observed associations*, not confirmed adverse effects. The 20-point subgroup difference used to order "further investigation"
rows is a screening heuristic, not a statistical test. Residual risk: many differently filtered queries by an authorised user could still be differenced.

### Manufacturer side (kept minimal)
`/manufacturer` shows an aggregate table built from the synthetic cohort. For each medication it gives
patients exposed, episodes, episodes that have both a baseline and a follow-up, and the observed
trajectory counts. It also shows the earlier aggregates of AI flags and clinician reviews. Any count
from fewer than 5 patients is suppressed. No patient IDs, dates or notes are returned. CSV-imported
patients are excluded from the cohort table.

### Model limitations (important)
- Both models are weak: about 34–36% accuracy on 4 balanced classes, where chance is 25%. Only
  *medication-associated* carries real signal.
- In the V4.2 generator (`scripts/generate_v4_2_data.py`), *disease progression*, *other cause* and
  *uncertain* come from the same process and differ only in the label. The observable record cannot
  tell them apart. The app states this beside those predictions. Classes were not merged and the
  models were not retrained.
- Predictions describe patterns in labelled synthetic examples. They are not diagnoses and do not
  show that a medicine caused a change.
- Model inputs use only records up to the anchor + 180-day cutoff. Leakage tests
  (`tests/test_stage10_leakage.py`) inject later assessments, events and dose changes and check that
  the features and prediction do not change.

### Not implemented (future scope)
- **External FDA adverse-event data (openFDA)**: not connected.
- **Hospital EHR / HL7 FHIR integration**: not implemented. A real connection needs an authorised
  provider FHIR endpoint, credentials and access approval (e.g. SMART on FHIR), patient identity
  matching, mapping of FHIR resources (Patient, MedicationStatement / MedicationRequest, Observation,
  Condition) onto the tables above, and the same validation the CSV import applies. A public test
  server is not a hospital integration.
- Excel upload, column mapping, editing or deleting imported patients, further analytics charts
  (time trends, demographics, dose vs score), and manufacturer trend or demographic views.

### Tests
`python -m pytest -q`: 225 tests (Python and the Node-based JS unit tests). They cover the
prediction API, leakage, explanations, analytics counts against direct SQL, evaluation metrics
against the saved JSON, CSV validation/rollback/duplicates, auth and role restrictions, and
manufacturer suppression.

### macOS note
`ml/__init__.py` imports xgboost before torch and runs torch single-threaded. Without this, loading
both libraries crashes, or the CNN-GRU model hangs, because each bundles its own OpenMP runtime.

## Troubleshooting: "Assessment service unavailable (ModuleNotFoundError)"
The prediction models need `xgboost`, `torch`, `scikit-learn` and `joblib` in the **same Python that runs the app**.
`run.py` prints that interpreter and any missing packages at start-up. Fix: run
`<that python> -m pip install -r requirements.txt` (Windows: `setup_windows.bat`, then `start_windows.bat`; Python 3.12).
The error shown on the patient page names the missing module and the interpreter; no prediction is ever invented.
