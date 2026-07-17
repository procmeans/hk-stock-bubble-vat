# Intraday Paper Web Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the A-share minute-factor strategy show up in the existing paper-trading website and refresh its buy/sell and NAV data through the same file-based workflow used by the seven existing strategies.

**Architecture:** Reuse the current `paper/<account>/state.json`, `nav.csv`, and `orders.csv` contract instead of inventing a second publishing path. Add a small intraday paper-runner that writes one new paper account, then let `paper.html` read it the same way it reads the other accounts. Keep the six-month validation report as a separate static artifact, but link it from the same page so users can move between live simulation state and research evidence.

**Tech Stack:** Python 3.11, pandas, GitHub Actions, static GitHub Pages, existing `intraday` and `strategies` packages, existing `paper.html` frontend.

## Global Constraints

- Keep the site static; do not add a server or database.
- Preserve the `paper/<account>/{state.json,nav.csv,orders.csv}` contract.
- Keep the intraday validation report under `output/intraday_6m/`.
- Use GitHub Actions to publish refreshed files by committing to `main`.
- Do not require Level2 data for the web integration path.

---

### Task 1: Add an intraday paper-runner that writes a paper account

**Files:**
- Create: `intraday/paper.py`
- Modify: `intraday/__init__.py`
- Create: `intraday/tests/test_paper.py`

**Interfaces:**
- Consumes: the existing intraday cache/plan artifacts from `intraday/run.py`, plus the existing `intraday.evaluate` and `intraday.portfolio` helpers.
- Produces: a command-line entry point that can materialize and update one paper account under `paper/a_intraday_6m/` with `state.json`, `nav.csv`, and `orders.csv`.

- [ ] **Step 1: Write the failing tests**

```python
def test_intraday_paper_run_is_idempotent_same_day(tmp_path, monkeypatch):
    ...

def test_intraday_paper_run_writes_state_nav_and_orders(tmp_path, monkeypatch):
    ...
```

The tests should assert that:
- the first run creates `paper/a_intraday_6m/state.json`, `nav.csv`, and `orders.csv`
- the runner records `strategy`, `market`, `capital`, `cash`, `pending_targets`, and `last_run`
- a second run on the same signal day makes no duplicate writes
- the generated CSV schemas match the existing paper-account schema

- [ ] **Step 2: Run the new tests and confirm they fail**

Run: `pytest intraday/tests/test_paper.py -v`

Expected: the module or command is missing, so the tests fail before implementation.

- [ ] **Step 3: Implement the runner with the smallest working surface**

```python
def run(account: str = "a_intraday_6m") -> None:
    ...
```

The implementation should:
- load the cached intraday plan and the latest available simulation inputs
- compute the current target weights using the existing minute-factor pipeline
- update the paper account using the same “pending target then execute next step” pattern as `strategies.paper`
- write files atomically so interrupted runs do not corrupt `state.json`

- [ ] **Step 4: Run the tests again**

Run: `pytest intraday/tests/test_paper.py -v`

Expected: pass.

- [ ] **Step 5: Commit the runner**

```bash
git add intraday/paper.py intraday/__init__.py intraday/tests/test_paper.py
git commit -m "feat: add intraday paper runner"
```

### Task 2: Surface the intraday account and report link in the existing paper page

**Files:**
- Modify: `paper/accounts.json`
- Modify: `paper.html`
- Modify: `index.html`
- Modify: `strategies/tests/test_paper.py`

**Interfaces:**
- Consumes: the new `paper/a_intraday_6m/*` account files from Task 1 and the existing static validation artifact at `output/intraday_6m/report.md`.
- Produces: a new selectable account button in `paper.html`, an updated title/link copy on the landing page, and a visible path to the six-month report.

- [ ] **Step 1: Write the failing tests**

```python
def test_repository_manifest_includes_intraday_account():
    ...

def test_paper_dashboard_includes_intraday_report_link():
    ...
```

The tests should assert that:
- `paper/accounts.json` now contains `a_intraday_6m` with a human-readable title and `¥` currency
- `paper.html` renders the new account from the manifest
- the page includes a link to the intraday research report
- the homepage copy reflects the new total strategy count after adding the intraday account

- [ ] **Step 2: Run the tests and confirm the current page does not satisfy them**

Run: `pytest strategies/tests/test_paper.py -v`

Expected: the new account and link assertions fail until the page copy and manifest are updated.

- [ ] **Step 3: Update the manifest and frontend copy**

```json
[
  {
    "account": "a_intraday_6m",
    "title": "A股 分钟线量价因子",
    "currency": "¥"
  }
]
```

The frontend changes should:
- let the new account appear as another tab in `paper.html`
- add a short description string for the intraday account
- add a visible report link to `output/intraday_6m/report.md`
- update the landing-page link text so it matches the current number of strategies shown in the paper page

- [ ] **Step 4: Run the tests again**

Run: `pytest strategies/tests/test_paper.py -v`

Expected: pass.

- [ ] **Step 5: Commit the UI changes**

```bash
git add paper/accounts.json paper.html index.html strategies/tests/test_paper.py
git commit -m "feat: surface intraday account in paper ui"
```

### Task 3: Automate the refresh path in GitHub Actions

**Files:**
- Create: `.github/workflows/intraday-paper.yml`
- Modify: `.github/workflows/update.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: the runner from Task 1 and the published account files from Task 2.
- Produces: a scheduled and manually triggerable workflow that updates the intraday paper account after fresh data is available and publishes the new state files.

- [ ] **Step 1: Write the failing workflow-level test**

```python
def test_infra_docs_or_workflow_mentions_intraday_paper_runner():
    ...
```

The test should assert that the repository mentions the intraday paper-runner command and the commit/push step that publishes `paper/a_intraday_6m/`.

- [ ] **Step 2: Run the test and confirm the workflow path is still absent**

Run: `pytest strategies/tests/test_paper.py intraday/tests/test_paper.py -v`

Expected: the refresh-path assertions fail until the workflow exists.

- [ ] **Step 3: Add the workflow**

```yaml
name: intraday-paper

on:
  schedule:
    - cron: "15 8 * * 1-5"
  workflow_dispatch: {}
```

The job should:
- check out the repo
- install the existing Python dependencies needed by the intraday pipeline
- run the intraday paper-runner
- commit and push `paper/a_intraday_6m/` so GitHub Pages can pick up the new state

- [ ] **Step 4: Update the docs**

Document the new refresh path in `README.md` next to the existing automation section so a reader can see where the new paper account comes from and when it updates.

- [ ] **Step 5: Run the relevant tests and the full suite**

Run:
```bash
pytest intraday/tests/test_paper.py strategies/tests/test_paper.py -v
pytest
```

Expected: all tests pass.

- [ ] **Step 6: Commit the automation**

```bash
git add .github/workflows/intraday-paper.yml .github/workflows/update.yml README.md
git commit -m "feat: automate intraday paper refresh"
```

