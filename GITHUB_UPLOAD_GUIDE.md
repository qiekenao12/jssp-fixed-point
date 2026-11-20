# GITHUB UPLOAD GUIDE — exact commands and pre-push checklist

> **Status (2026-11-20): DONE.** Core repo pushed to
> `git@github.com:qiekenao12/jssp-fixed-point.git` (commit `220bf87`, 2,058 files, verified via
> the GitHub API). Release **v1.0.0** published with all 7 assets; every GitHub-reported SHA256
> digest matches `release_assets/SHA256SUMS.txt`, and a test download re-verified byte integrity.
> Remaining: Zenodo DOI (§4) and the author inputs in §6. The commands below are kept as the
> reproducible record of what was executed.

Prepared 2026-11-20. Two deliverables were staged from the private source workspace
(never modified):

- `github_release/` — the core Git repository (this directory).
- `release_assets/` — five bulky historical tarballs for GitHub Releases / Zenodo
  (sibling directory; **not** committed to Git).

## 1. Pre-push checklist (measured on the staged tree)

| Gate | Requirement | Measured | Status |
|---|---|---|---|
| Max single file | < 50 MB (GitHub hard block: 100 MB) | 10.37 MB (`projects/analysis/fp_decision_study/original_final_analysis.json`) | PASS |
| Core repo size | < 300 MB | ~127 MB | PASS |
| Core file count | no tens of thousands of tiny files | ~2,058 files | PASS |
| Virtualenvs / caches | none shipped (`venv_ext/`, `venv/`, `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `.DS_Store`) | none | PASS |
| Manuscript drafts | no Word/PDF drafts, no unrelated-project files | none (only the hash-referenced study Markdown source, see `RELEASE_NOTES.md` §4–5) | PASS |
| Secrets / personal paths | no `.env`, API keys, `/Users/`, `/home/`, `C:\Users`, usernames | none outside the documented hash-lock allowlist (`RELEASE_NOTES.md` §4) | PASS |
| Lock integrity | all SHA256 locks re-verified in the staged tree | 32 primary + 10 new / 12 reused ext code + 84 benchmark hashes; `design_hash`/`instances_hash`/`lock_hash` recomputed | PASS |
| Clean-room verification | `python scripts/verify_all.py --figures` from inside `github_release/` | 25 PASS / 0 FAIL (Python 3.13.13, numpy 2.4.6) | PASS |
| Historical replay | `python scripts/verify_all.py --historical` after extracting `release_assets/` | 29 PASS / 0 FAIL with `--figures --historical` (all four historical cohort audits included; 168.5 s total) | PASS |
| Raw-data split | historical raw > 200 MB / > 2,000 files ⇒ Release assets | 563 MB / 2,007 files → 5 tarballs (~127 MB compressed) + `SHA256SUMS.txt` | DONE |

Re-run any time:

```bash
cd github_release
python scripts/verify_all.py --quick      # locks + hygiene (seconds)
python scripts/verify_all.py --figures    # full clean-room verification + figure regeneration
```

## 2. Initialize and push the core repository

```bash
cd github_release

git init -b main
git add -A
# sanity: confirm nothing ignored slipped in and no large file is staged
git status --short | head
git ls-files | wc -l                      # expect ~2,058
git ls-files -z | xargs -0 -I{} sh -c 'test $(stat -c%s "{}") -lt 52428800 || echo "TOO BIG: {}"'

git commit -m "Reproducibility package: FP decision sufficiency, two-solver JSSP evidence + prospective 50-instance extension

- 221,040 provenance branches across three strictly separated cohorts
- All source, frozen models, lock manifests, paper tables/figures, S8 diagnostics
- Independent schedule audits + scalar replays (primary + extension), one-command verify_all.py
- Historical raw branches distributed via GitHub Releases / Zenodo (see release_assets/)"

# create an EMPTY GitHub repository (no README/license auto-init), then:
git remote add origin git@github.com:<YOUR_ACCOUNT>/<REPO_NAME>.git
git push -u origin main
```

HTTPS alternative: `git remote add origin https://github.com/<YOUR_ACCOUNT>/<REPO_NAME>.git`.

## 3. Upload the large archives to a GitHub Release

All five tarballs are < 51 MB each (GitHub Release asset limit: 2 GB per file).

```bash
# with GitHub CLI (gh auth login first):
gh release create v1.0.0 \
  ../release_assets/historical_raw_fp_v2_development.tar.gz \
  ../release_assets/historical_raw_fp_v2_validation.tar.gz \
  ../release_assets/historical_raw_fp_empirical_repeatability_development.tar.gz \
  ../release_assets/historical_raw_fp_empirical_confirmation.tar.gz \
  ../release_assets/historical_analysis_large_jsons.tar.gz \
  ../release_assets/SHA256SUMS.txt \
  ../release_assets/README.md \
  --title "v1.0.0 — SWEVO FP-decision study reproducibility data" \
  --notes "Historical genetic/memetic raw branches (109,440) + frozen historical analysis JSONs.
Core repository contains all code, locks, models, paper tables/figures and the primary +
extension raw cohorts. Verify downloads with SHA256SUMS.txt; extract per release_assets/README.md.
License: code MIT, data CC-BY-4.0 (see LICENSE)."
```

Without `gh`: GitHub → repository → *Releases* → *Draft a new release* → tag `v1.0.0` →
drag the seven files from `release_assets/` into the asset area → publish.

## 4. Zenodo DOI (citable archive)

Option A (automatic): Zenodo → *Settings* → *GitHub integration* → enable the repository;
each published GitHub Release is archived and a DOI is minted (the release assets are included).

Option B (manual): zenodo.org → *New upload* → upload the five tarballs + `SHA256SUMS.txt` +
`release_assets/README.md` **and** a ZIP of the core repo (`git archive --format=zip -o
../fp_decision_core_v1.0.0.zip HEAD` from inside `github_release/`) → fill metadata
(title = manuscript title; creators = author input; license = MIT for code / CC-BY-4.0 for data;
related identifiers = the GitHub URL and the SWEVO submission) → publish → copy the DOI.

Then replace the placeholders in `README.md` §9 (citation block) and §6 with the real
GitHub URL / Zenodo DOI, and commit the update.

## 5. Post-upload verification (fresh clone)

```bash
git clone git@github.com:<YOUR_ACCOUNT>/<REPO_NAME>.git fp_check
cd fp_check
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_all.py --figures          # expect 25 PASS / 0 FAIL

# historical replay (optional):
# download the five tarballs + SHA256SUMS.txt from the Release into ./release_assets/
sha256sum -c release_assets/SHA256SUMS.txt
mkdir -p raw_extracted
for f in release_assets/historical_*.tar.gz; do tar -xzf "$f" -C raw_extracted/; done
python scripts/verify_all.py --historical       # expect 28 PASS / 0 FAIL
```

Historical-audit notes (also in `RELEASE_NOTES.md` §2): the fp_v2 cohorts lock
pre-2026-10-26 versions of three solver modules; `verify_all.py --historical` automatically
extracts the shipped frozen snapshot `projects/experiments/fp_v2_full_replay_sources.zip`
over its temporary sandbox before auditing them. The repeatability replay consumes
`fp_v2_development_analysis.json` from `historical_analysis_large_jsons.tar.gz`.

## 6. Final author inputs before flipping the repo public

- Author names/affiliations, funding, conflicts of interest, AI-use disclosure wording.
- Related-manuscript disclosure (the study documents state that any shared material with the
  separate resource-allocation manuscript must be disclosed and rights-checked by the authors).
- Rights confirmation for redistributing the third-party benchmark texts in your jurisdiction
  (attribution + fingerprints are provided; `prepare_benchmarks.py` can re-fetch them instead).
- Replace citation/DOI placeholders; choose the final repository name.
- The private workspace delivery README explicitly warned: do not publish the private archive
  before resolving third-party material rights. This staged release contains no private drafts,
  but the benchmark-rights and disclosure inputs above remain author responsibilities.
