#!/usr/bin/env python3
"""
Build dataset for Question A: author-level stats, papers, topics, journals.
Reads data/master_dataset.csv and the OpenAlex cache; writes five CSVs to data/.
Run from the repository root: python build_dataset.py
"""

import sys
import os
import json
import re
import unicodedata
import time
from urllib.parse import urlencode
from collections import defaultdict, Counter

import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF

MAILTO = "huynhatminh307@gmail.com"
CACHE_FILE = "data/openalex_cache.json"
SNAPSHOT_YEAR = 2026
JOURNALS = {
    "Financial Services Review": ("FSR", "1057-0810"),
    "Financial Planning Review": ("FPR", "2573-8615"),
    "Journal of Financial Counseling and Planning": ("JFCP", "1052-3073"),
}
SUSPECT_WORKS = 300
WORK_SELECT = "id,doi,title,publication_year,authorships"
AUTHOR_SELECT = "id,display_name,works_count,cited_by_count,summary_stats,last_known_institutions,counts_by_year,ids"
SOURCE_SELECT = "id,display_name,issn_l,issn,summary_stats,works_count,cited_by_count,counts_by_year"
# Question B's final topic model (question_b_topic_modelling.ipynb): stop words,
# labels (report Appendix A) and first top term of each component, used as a drift check.
QB_STOPS = {
    "study", "results", "paper", "research", "article", "using", "based", "data",
    "analysis", "findings", "significant", "significantly", "academy", "service",
    "services", "right", "rights", "reserved",
}
TOPICS = [
    ("retirement", "Retirement Planning"),
    ("fund", "Mutual Funds"),
    ("literacy", "Financial Literacy"),
    ("risk", "Risk Tolerance"),
    ("credit", "Credit and Debt"),
    ("portfolio", "Portfolio Investment"),
    ("household", "Household Finance"),
    ("financial planning", "Financial Advice"),
    ("student", "Student Finance"),
    ("behavior", "Financial Behaviour"),
    ("tax", "Tax and Retirement Accounts"),
]

_cache = {}  # OpenAlex responses keyed by request


class BudgetExhausted(RuntimeError):
    pass


# Parsing and matching helpers


def norm_surname(name: str) -> str:
    """Normalize surname: NFKD + strip combining, drop suffix tokens, last token, lowercase, [a-z0-9]."""
    if not name:
        return ""
    # NFKD normalize + strip combining marks (accent-fold)
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    # Take text before first comma if present
    if "," in normalized:
        normalized = normalized.split(",")[0]
    # Drop suffix tokens (jr, sr, ii, iii, iv, with/without period)
    tokens = normalized.split()
    while tokens and tokens[-1].rstrip(".").lower() in ("jr", "sr", "ii", "iii", "iv"):
        tokens.pop()
    # Take last remaining token
    surname = tokens[-1] if tokens else ""
    # Lowercase, keep only [a-z0-9]
    return re.sub(r"[^a-z0-9]", "", surname.lower())


def parse_scopus_authors(authors_id: str, authors_full_name: str) -> list:
    """Parse Scopus author list. Returns [(scopus_id, surname_norm), ...] in order."""
    if not authors_id:
        return []
    ids = [x.strip() for x in authors_id.split(";") if x.strip()]
    # Build {id: surname} lookup from authors_full_name via regex
    matches = re.findall(r"([^;(]+?)\s*\((\d+)\)", authors_full_name or "")
    id_to_surname = {}
    for name_part, sid in matches:
        name_part = name_part.strip()
        # Extract surname (text before first comma)
        surname_text = name_part.split(",")[0] if "," in name_part else name_part
        id_to_surname[sid] = norm_surname(surname_text)
    # Build result in order of authors_id
    return [(sid, id_to_surname.get(sid, "")) for sid in ids]


def match_authorships(scopus: list, authorships: list) -> dict:
    """Greedy left-to-right: match scopus authors to OpenAlex authorships by surname."""
    result = {}
    claimed = set()
    for scopus_idx, (scopus_id, scopus_surname) in enumerate(scopus):
        if not scopus_surname:
            continue
        # Find unclaimed candidates with matching surname
        candidates = []
        for oa_idx, authorship in enumerate(authorships):
            if oa_idx in claimed:
                continue
            author = authorship.get("author") or {}
            raw_name = authorship.get("raw_author_name") or ""
            display_name = author.get("display_name") or ""
            oa_surname = norm_surname(raw_name or display_name)
            oa_id = author.get("id") or ""
            if oa_surname == scopus_surname and oa_id:
                candidates.append((oa_idx, oa_id))
        if candidates:
            # Pick candidate with nearest index
            oa_idx, oa_id = min(candidates, key=lambda x: abs(x[0] - scopus_idx))
            claimed.add(oa_idx)
            result[scopus_id] = oa_id
    return result


def title_matches(title_a: str, year_a, title_b: str, year_b) -> bool:
    """Match titles: normalize both, check equality or prefix, and year within 1."""

    def norm_title(t):
        t = t.lower() if t else ""
        t = re.sub(r"[^a-z0-9 ]", "", t)
        return re.sub(r"\s+", " ", t).strip()

    norm_a, norm_b = norm_title(title_a), norm_title(title_b)
    if not norm_a or not norm_b:
        return False
    match = norm_a == norm_b or norm_a.startswith(norm_b) or norm_b.startswith(norm_a)
    return match and abs((year_a or 0) - (year_b or 0)) <= 1


def prior_counts(counts_by_year: list, year: int) -> tuple:
    """Return (prior_works, prior_cites) = sums of works and citations before year."""
    if not counts_by_year:
        return (0, 0)
    prior_works = sum(
        e.get("works_count", 0) for e in counts_by_year if e.get("year", 0) < year
    )
    prior_cites = sum(
        e.get("cited_by_count", 0) for e in counts_by_year if e.get("year", 0) < year
    )
    return (prior_works, prior_cites)


def clean_text(text: str, lemmatize) -> str:
    """Question B cleaning: lowercase, letters only, drop stop words, lemmatise."""
    tokens = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    stops = ENGLISH_STOP_WORDS | QB_STOPS
    return " ".join(lemmatize(t) for t in tokens if t not in stops)


def author_theme(W_rows: np.ndarray, topic_ids_rows: np.ndarray) -> tuple:
    """Compute author theme: argmax of mean W weights, share = fraction of papers in that topic."""
    m = W_rows.mean(0)
    if m.sum() == 0:
        return (-1, 0.0)
    t = int(m.argmax())
    share = float((topic_ids_rows == t).mean())
    return (t, share)


# Cache and API


def _load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    else:
        _cache = {}


def _save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(_cache, f)


def _cache_key(path: str, params: dict) -> str:
    """Cache key: path + "?" + urlencode of params (sorted, excluding api_key/mailto)."""
    filtered = {k: v for k, v in params.items() if k not in ("api_key", "mailto")}
    query = urlencode(sorted(filtered.items()))
    return f"{path}?{query}"


def api_get(path: str, params: dict) -> dict:
    """GET to OpenAlex with backoff, caching, rate limiting."""
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key and os.path.exists(".openalex_key"):
        api_key = open(".openalex_key").read().strip()

    # Always add auth params
    params = dict(params)
    if api_key:
        params["api_key"] = api_key
    params["mailto"] = MAILTO

    # Check cache
    key = _cache_key(path, params)
    if key in _cache:
        return _cache[key]

    url = f"https://api.openalex.org{path}"
    print(f"  network GET {path}", flush=True)

    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                _cache[key] = data
                time.sleep(0.1)
                # Flush every 25 new responses
                if len(_cache) % 25 == 0:
                    _save_cache()
                return data
            elif r.status_code == 429:
                try:
                    body = r.json()
                except Exception:
                    body = {}
                if body.get("creditsRemaining", 1) < body.get("creditsRequired", 0):
                    reset_hrs = int(r.headers.get("Retry-After", 0)) / 3600
                    raise BudgetExhausted(
                        f"Daily OpenAlex budget exhausted - resets in ~{reset_hrs:.1f}h "
                        f"(midnight UTC). Set OPENALEX_API_KEY (or put the key in .openalex_key) "
                        f"or re-run later; cached progress is preserved."
                    )
                retry_after = min(int(r.headers.get("Retry-After", 2**attempt)), 30)
                print(f"  [429] backing off {retry_after}s: {url}", flush=True)
                time.sleep(retry_after)
            elif r.status_code == 409:
                print(
                    "Out of OpenAlex credits today - set OPENALEX_API_KEY or wait for daily reset"
                )
                sys.exit(1)
            elif r.status_code >= 500:
                time.sleep(2**attempt)
            else:
                r.raise_for_status()
        except BudgetExhausted:
            raise
        except Exception:
            if attempt < 4:
                time.sleep(2**attempt)
            else:
                raise
    raise RuntimeError(f"Failed to fetch {url} after 5 attempts")


def cached_results(prefix: str) -> list:
    """Iterate cache, extend a list with results from keys starting with prefix."""
    results = []
    for key, value in _cache.items():
        if key.startswith(prefix):
            results.extend(value.get("results", []))
    return results


# Pipeline functions


def load_scopus() -> pd.DataFrame:
    """Load Scopus CSV, assign paper_id, add journal/year/cited_by/paper_age columns."""
    print("Loading Scopus CSV...")
    raw = pd.read_csv("data/master_dataset.csv", dtype={"authors_id": str})

    # Drop rows with null authors_id
    before = len(raw)
    df = raw.dropna(subset=["authors_id"]).copy()
    dropped = before - len(df)

    # Reset index and assign paper_id
    df = df.reset_index(drop=True)
    df["paper_id"] = range(len(df))

    # Add computed columns
    df["year"] = df["year"].astype(int)
    df["cited_by"] = df["cited_by"].astype(int)
    df["journal"] = df["source_title"].map(lambda x: JOURNALS.get(x, ("", ""))[0])
    df["paper_age"] = SNAPSHOT_YEAR - df["year"] + 1

    # Count authors
    def count_authors(authors_id):
        if pd.isna(authors_id):
            return 0
        return len([x for x in str(authors_id).split(";") if x.strip()])

    df["n_authors"] = df["authors_id"].apply(count_authors)

    assert len(df) == 1360, f"Expected 1360 rows, got {len(df)}"

    print(f"  Rows in: {before}, dropped: {dropped}, kept: {len(df)}")
    journal_counts = df.groupby("journal").size()
    for j in sorted(journal_counts.index):
        print(f"    {j}: {journal_counts[j]}")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")

    return df


def resolve_works(df: pd.DataFrame) -> tuple:
    """Resolve papers to OpenAlex works via DOI or title search."""
    print("Resolving papers to OpenAlex works...")

    paper_works = {}
    resolution_status = {}

    # Split by DOI availability. Batches are formed in paper order so the cache keys
    # written by the first run are reproduced exactly on every later run.
    has_doi = []
    needs_title = []
    for row in df.itertuples():
        doi = row.doi if pd.notna(row.doi) else None
        if doi:
            doi_clean = doi.strip().lower()
            if "," not in doi_clean and "|" not in doi_clean:
                has_doi.append((row.paper_id, doi_clean, row))
            else:
                needs_title.append((row.paper_id, row))
        else:
            needs_title.append((row.paper_id, row))

    # Batch DOI requests (50 per batch)
    for i in range(0, len(has_doi), 50):
        batch_rows = has_doi[i : i + 50]
        dois_batch = [doi_clean for _, doi_clean, _ in batch_rows]

        if dois_batch:
            try:
                filter_str = "|".join(dois_batch)
                result = api_get(
                    "/works",
                    {
                        "filter": f"doi:{filter_str}",
                        "select": WORK_SELECT,
                        "per-page": 50,
                    },
                )
                works_by_doi = {
                    w["doi"].lower().removeprefix("https://doi.org/"): w
                    for w in result.get("results", [])
                    if w.get("doi")
                }
                for pid, doi_clean, row in batch_rows:
                    if doi_clean in works_by_doi:
                        paper_works[pid] = works_by_doi[doi_clean]
                        resolution_status[pid] = "doi"
                    else:
                        needs_title.append((pid, row))
            except Exception:
                for pid, _, row in batch_rows:
                    needs_title.append((pid, row))

    # Title fallback. Once the daily API budget is gone, the remaining papers are
    # marked unresolved rather than each waiting through the retry loop.
    budget_gone = False
    for pid, row in needs_title:
        title_clean = re.sub(
            r"[,:|]", "", row.title.lower() if pd.notna(row.title) else ""
        )
        params = {
            "filter": f"title.search:{title_clean}",
            "select": WORK_SELECT,
            "per-page": 5,
        }
        if not title_clean or (budget_gone and _cache_key("/works", params) not in _cache):
            resolution_status[pid] = "unresolved"
            continue

        try:
            result = api_get("/works", params)
            found = False
            for work in result.get("results", []):
                if title_matches(
                    row.title, row.year, work["title"], work.get("publication_year")
                ):
                    paper_works[pid] = work
                    resolution_status[pid] = "title"
                    found = True
                    break
            if not found:
                resolution_status[pid] = "unresolved"
        except BudgetExhausted as e:
            print(f"  {e}")
            budget_gone = True
            resolution_status[pid] = "unresolved"
        except Exception:
            resolution_status[pid] = "unresolved"

    # Tally by journal
    resolution_by_journal = defaultdict(
        lambda: {"total": 0, "doi": 0, "title": 0, "unresolved": 0}
    )
    for row in df.itertuples():
        journal = row.journal
        resolution_by_journal[journal]["total"] += 1
        status = resolution_status.get(row.paper_id, "unresolved")
        resolution_by_journal[journal][status] += 1

    print(f"  Resolved {len(paper_works)} of {len(df)} papers")
    for journal in ["FSR", "FPR", "JFCP"]:
        counts = resolution_by_journal[journal]
        total = counts["total"]
        doi_c = counts["doi"]
        title_c = counts["title"]
        unres = counts["unresolved"]
        pct = 100 * (doi_c + title_c) / total if total else 0
        print(
            f"    {journal}: {total} total, {doi_c} doi, {title_c} title, {unres} unresolved ({pct:.0f}%)"
        )

    return paper_works, resolution_status


def match_authors(df: pd.DataFrame, works: dict, methods: dict) -> pd.DataFrame:
    """Vote on OpenAlex author matches across papers."""
    print("Matching Scopus authors to OpenAlex...")

    votes = defaultdict(lambda: defaultdict(int))
    vote_methods = defaultdict(lambda: defaultdict(set))

    for row in df.itertuples():
        paper_id = row.paper_id
        if paper_id not in works:
            continue
        work = works[paper_id]
        authorships = work.get("authorships", [])
        scopus_authors = parse_scopus_authors(row.authors_id, row.authors_full_name)
        matches = match_authorships(scopus_authors, authorships)
        method = methods.get(paper_id, "title")
        for scopus_id, openalex_id in matches.items():
            votes[scopus_id][openalex_id] += 1
            vote_methods[scopus_id][openalex_id].add(method)

    # Build list of all unique Scopus author IDs
    all_scopus_ids = set()
    for row in df.itertuples():
        parsed = parse_scopus_authors(row.authors_id, row.authors_full_name)
        for sid, _ in parsed:
            if sid:
                all_scopus_ids.add(sid)

    # Resolve to modal
    crosswalk = []
    for scopus_id in sorted(all_scopus_ids):
        if scopus_id not in votes:
            crosswalk.append(
                {
                    "scopus_author_id": scopus_id,
                    "openalex_author_id": "",
                    "match_method": "unmatched",
                    "n_votes": 0,
                }
            )
            continue
        candidates = votes[scopus_id]
        modal_oa_id = max(candidates.items(), key=lambda x: x[1])[0]
        methods_set = vote_methods[scopus_id][modal_oa_id]
        match_method = "doi" if "doi" in methods_set else "title"
        crosswalk.append(
            {
                "scopus_author_id": scopus_id,
                "openalex_author_id": modal_oa_id,
                "match_method": match_method,
                "n_votes": candidates[modal_oa_id],
            }
        )

    crosswalk_df = pd.DataFrame(crosswalk)
    matched_count = sum(1 for row in crosswalk if row["openalex_author_id"])
    unique_oa_ids = set(
        row["openalex_author_id"] for row in crosswalk if row["openalex_author_id"]
    )

    # Count OA ids shared by two Scopus ids
    oa_to_scopus = defaultdict(set)
    for row in crosswalk:
        if row["openalex_author_id"]:
            oa_to_scopus[row["openalex_author_id"]].add(row["scopus_author_id"])
    shared_count = sum(1 for ids in oa_to_scopus.values() if len(ids) > 1)

    print(
        f"  Matched {matched_count} of {len(crosswalk)} Scopus authors ({100*matched_count/len(crosswalk):.1f}%)"
    )
    print(f"  Unique OpenAlex authors: {len(unique_oa_ids)}")
    print(f"  OpenAlex ids shared by multiple Scopus ids: {shared_count}")

    return crosswalk_df


def fetch_author_records(oa_ids: list) -> dict:
    """Fetch OpenAlex author records for the given ids."""
    print("Fetching OpenAlex author records...")

    # Load from cache first
    cached_authors = {}
    for author in cached_results("/authors?"):
        cached_authors[author["id"]] = author

    from_cache = sum(1 for oa_id in oa_ids if oa_id in cached_authors)
    still_missing = [oa_id for oa_id in oa_ids if oa_id not in cached_authors]

    # Batch fetch missing (50 per batch)
    fetched = 0
    for i in range(0, len(still_missing), 50):
        batch = still_missing[i : i + 50]
        filter_str = "|".join(batch)
        try:
            result = api_get(
                "/authors",
                {
                    "filter": f"openalex:{filter_str}",
                    "select": AUTHOR_SELECT,
                    "per-page": 50,
                },
            )
            for author in result.get("results", []):
                cached_authors[author["id"]] = author
                fetched += 1
        except Exception as e:
            print(f"  Author batch failed: {e}")

    final_missing = sum(1 for oa_id in oa_ids if oa_id not in cached_authors)

    print(
        f"  From cache: {from_cache}, fetched: {fetched}, still missing: {final_missing}"
    )

    # Check career coverage
    early_years = []
    for author in cached_authors.values():
        counts = author.get("counts_by_year", [])
        if counts:
            years = [e.get("year") for e in counts if e.get("works_count", 0) > 0]
            if years:
                early_years.append(min(years))

    if early_years:
        pre_2010 = sum(1 for y in early_years if y <= 2010)
        pct = 100 * pre_2010 / len(early_years) if early_years else 0
        print(
            f"  Career coverage: {pre_2010}/{len(early_years)} ({pct:.1f}%) have career start at or before 2010"
        )

    return cached_authors


def fetch_journals() -> pd.DataFrame:
    """Fetch and validate journal records from OpenAlex."""
    print("Fetching journal records...")

    issns = [issn for _, issn in JOURNALS.values()]
    result = api_get(
        "/sources",
        {"filter": "issn:" + "|".join(issns), "select": SOURCE_SELECT, "per-page": 10},
    )
    by_issn = {src.get("issn_l"): src for src in result.get("results", [])}

    journals_data = []
    for source_title, (code, issn) in JOURNALS.items():
        found = by_issn.get(issn)
        if not found:
            print(f"  ERROR: could not find source with issn_l={issn}")
            sys.exit(1)
        summary = found.get("summary_stats", {})
        journals_data.append(
            {
                "journal": code,
                "source_title": source_title,
                "openalex_source_id": found["id"].removeprefix("https://openalex.org/"),
                "issn_l": found.get("issn_l"),
                "two_yr_mean_citedness": summary.get("2yr_mean_citedness"),
                "h_index": summary.get("h_index"),
                "i10_index": summary.get("i10_index"),
                "works_count": found.get("works_count"),
                "cited_by_count": found.get("cited_by_count"),
            }
        )

    journals_df = pd.DataFrame(journals_data)

    # Verify source ids
    expected_ids = {"S102700472", "S4210200979", "S2912066455"}
    actual_ids = set(journals_df["openalex_source_id"])
    assert (
        actual_ids == expected_ids
    ), f"Expected source ids {expected_ids}, got {actual_ids}"

    print(f"  Fetched {len(journals_df)} journals")
    return journals_df


def topic_model(df: pd.DataFrame):
    """Question B's final NMF (k = 11), fitted on all 1,362 master_dataset rows so the
    topics match the report exactly, then restricted to the papers kept in df."""
    from nltk.stem import WordNetLemmatizer

    print("Building topic model...")
    raw = pd.read_csv("data/master_dataset.csv", dtype={"authors_id": str})
    text = (raw["title"].fillna("") + " " + raw["abstract"].fillna("") + " "
            + raw["author_keywords"].fillna("").str.replace(";", " "))
    lemmatize = WordNetLemmatizer().lemmatize
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=5, ngram_range=(1, 3))
    X = vectorizer.fit_transform(text.map(lambda t: clean_text(t, lemmatize)))
    nmf = NMF(n_components=len(TOPICS), init="nndsvda", random_state=42, max_iter=500)
    W = nmf.fit_transform(X)[raw["authors_id"].notna().to_numpy()]
    H = nmf.components_
    assert len(W) == len(df)

    # Normalise W rows to topic proportions; a paper's topic is its largest proportion
    row_sums = W.sum(axis=1, keepdims=True)
    W_norm = W / np.where(row_sums > 0, row_sums, 1)
    topic_ids = np.argmax(W_norm, axis=1)
    topic_ids[W_norm.sum(axis=1) == 0] = -1

    terms = vectorizer.get_feature_names_out()
    topics_data = []
    for k, (first_term, label) in enumerate(TOPICS):
        top = [terms[i] for i in np.argsort(H[k])[-10:][::-1]]
        assert top[0] == first_term, f"topic {k} drifted from Question B: {top[:3]}"
        topics_data.append({
            "topic_id": k,
            "topic_label": label,
            "top_terms": ", ".join(top),
            "n_papers": int(sum(topic_ids == k)),
            "n_papers_fsr": int(sum((topic_ids == k) & (df["journal"] == "FSR"))),
            "n_papers_fpr": int(sum((topic_ids == k) & (df["journal"] == "FPR"))),
            "n_papers_jfcp": int(sum((topic_ids == k) & (df["journal"] == "JFCP"))),
        })
    topics_df = pd.DataFrame(topics_data)
    for _, row in topics_df.iterrows():
        print(f"    Topic {row['topic_id']}: {row['topic_label']} ({row['n_papers']} papers)")
    return W_norm, topic_ids, topics_df


def build_paper_authors(
    df: pd.DataFrame, crosswalk: pd.DataFrame, records: dict
) -> pd.DataFrame:
    """Build paper_authors table (one row per paper×author)."""
    print("Building paper_authors table...")

    pa_list = []
    for row in df.itertuples():
        parsed = parse_scopus_authors(row.authors_id, row.authors_full_name)
        for position, (scopus_id, _) in enumerate(parsed, start=1):
            pa_row = {
                "paper_id": row.paper_id,
                "scopus_author_id": scopus_id,
                "author_position": position,
                "openalex_author_id": "",
                "prior_works": np.nan,
                "prior_cites": np.nan,
                "h_index": np.nan,
            }

            # Look up OpenAlex id in crosswalk
            cw = crosswalk[crosswalk["scopus_author_id"] == scopus_id]
            if len(cw) > 0 and cw.iloc[0]["openalex_author_id"]:
                oa_id = cw.iloc[0]["openalex_author_id"]
                pa_row["openalex_author_id"] = oa_id

                # Look up prior counts and h_index
                if oa_id in records:
                    rec = records[oa_id]
                    prior_w, prior_c = prior_counts(
                        rec.get("counts_by_year", []), row.year
                    )
                    pa_row["prior_works"] = float(prior_w)
                    pa_row["prior_cites"] = float(prior_c)
                    summary = rec.get("summary_stats", {})
                    h_idx = summary.get("h_index")
                    if h_idx is not None:
                        pa_row["h_index"] = float(h_idx)

            pa_list.append(pa_row)

    pa_df = pd.DataFrame(pa_list)
    print(f"  Created {len(pa_df)} paper_author rows")
    return pa_df


def build_papers(
    df: pd.DataFrame,
    pa: pd.DataFrame,
    W: np.ndarray,
    topic_ids: np.ndarray,
    topics_df: pd.DataFrame,
    journals_df: pd.DataFrame,
    works: dict,
    resolution_status: dict,
) -> pd.DataFrame:
    """Build paper-level dataset (1360 rows)."""
    print("Building paper dataset...")

    # Aggregate team features
    team_agg = pa.groupby("paper_id")[["prior_works", "prior_cites", "h_index"]].max()
    n_authors_matched = pa[pa["openalex_author_id"] != ""].groupby("paper_id").size()
    n_authors_matched = n_authors_matched.reindex(df["paper_id"], fill_value=0)

    papers_list = []
    topics_by_id = {row["topic_id"]: row for _, row in topics_df.iterrows()}

    for row in df.itertuples():
        paper_dict = {
            "paper_id": row.paper_id,
            "journal": row.journal,
            "year": row.year,
            "title": row.title,
            "doi": row.doi if pd.notna(row.doi) else "",
            "cited_by": row.cited_by,
            "n_authors": row.n_authors,
            "paper_age": row.paper_age,
            "openalex_work_id": (
                works.get(row.paper_id, {}).get("id", "")
                if row.paper_id in works
                else ""
            ),
            "resolution": resolution_status.get(row.paper_id, "unresolved"),
            "n_authors_matched": int(n_authors_matched.get(row.paper_id, 0)),
        }

        # Team track record
        if row.paper_id in team_agg.index:
            paper_dict["team_prior_works_max"] = team_agg.loc[
                row.paper_id, "prior_works"
            ]
            paper_dict["team_prior_cites_max"] = team_agg.loc[
                row.paper_id, "prior_cites"
            ]
            paper_dict["team_h_index_max"] = team_agg.loc[row.paper_id, "h_index"]
        else:
            paper_dict["team_prior_works_max"] = np.nan
            paper_dict["team_prior_cites_max"] = np.nan
            paper_dict["team_h_index_max"] = np.nan

        # Topic assignment
        topic_id = int(topic_ids[row.paper_id])
        paper_dict["topic_id"] = topic_id
        if topic_id >= 0 and topic_id in topics_by_id:
            paper_dict["topic_label"] = topics_by_id[topic_id]["topic_label"]
        else:
            paper_dict["topic_label"] = "unassigned"

        # Topic weights
        for k in range(len(topics_df)):
            paper_dict[f"topic_w{k}"] = float(W[row.paper_id, k])

        # Journal citedness
        journal_row = journals_df[journals_df["journal"] == row.journal]
        if len(journal_row) > 0:
            paper_dict["journal_2yr_citedness"] = journal_row.iloc[0][
                "two_yr_mean_citedness"
            ]
        else:
            paper_dict["journal_2yr_citedness"] = np.nan

        papers_list.append(paper_dict)

    papers_df = pd.DataFrame(papers_list)
    assert len(papers_df) == 1360, f"Expected 1360 papers, got {len(papers_df)}"

    print(f"  Built {len(papers_df)} papers")
    return papers_df


def build_authors(
    df: pd.DataFrame,
    pa: pd.DataFrame,
    crosswalk: pd.DataFrame,
    records: dict,
    W: np.ndarray,
    topic_ids: np.ndarray,
    topics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build author-level dataset (1953 rows)."""
    print("Building author dataset...")

    # Merge paper_authors with papers to get journal, year, cited_by
    pa_with_papers = pa.merge(
        df[["paper_id", "journal", "year", "cited_by"]], on="paper_id", how="left"
    )

    # Aggregate per author
    authors_list = []
    for scopus_id in sorted(pa["scopus_author_id"].unique()):
        author_papers = pa_with_papers[pa_with_papers["scopus_author_id"] == scopus_id]

        # Total and per-journal counts
        n_papers = len(author_papers)
        cites_total = author_papers["cited_by"].sum()

        n_papers_fsr = len(author_papers[author_papers["journal"] == "FSR"])
        cites_fsr = author_papers[author_papers["journal"] == "FSR"]["cited_by"].sum()

        n_papers_fpr = len(author_papers[author_papers["journal"] == "FPR"])
        cites_fpr = author_papers[author_papers["journal"] == "FPR"]["cited_by"].sum()

        n_papers_jfcp = len(author_papers[author_papers["journal"] == "JFCP"])
        cites_jfcp = author_papers[author_papers["journal"] == "JFCP"]["cited_by"].sum()

        # Year range
        years = author_papers["year"].dropna().unique()
        first_year = int(min(years)) if len(years) > 0 else np.nan
        last_year = int(max(years)) if len(years) > 0 else np.nan

        # Scopus name
        cw = crosswalk[crosswalk["scopus_author_id"] == scopus_id]
        match_method = cw.iloc[0]["match_method"] if len(cw) > 0 else "unmatched"
        n_votes = int(cw.iloc[0]["n_votes"]) if len(cw) > 0 else 0

        # Parse surname from first paper's authors_full_name
        scopus_name = ""
        first_paper = author_papers.iloc[0]
        for row in df[df["paper_id"] == first_paper["paper_id"]].itertuples():
            matches = re.findall(r"([^;(]+?)\s*\((\d+)\)", row.authors_full_name or "")
            for name_part, sid in matches:
                if sid == scopus_id:
                    name_text = (
                        name_part.strip().split(",")[0]
                        if "," in name_part
                        else name_part.strip()
                    )
                    scopus_name = name_text
                    break

        # OpenAlex data
        oa_id = cw.iloc[0]["openalex_author_id"] if len(cw) > 0 else ""
        openalex_name = ""
        oa_works_count = np.nan
        oa_cited_by_count = np.nan
        h_index = np.nan
        i10_index = np.nan
        oa_first_year = np.nan
        institution = ""
        orcid = ""
        oa_suspect = False

        if oa_id and oa_id in records:
            rec = records[oa_id]
            openalex_name = rec.get("display_name", "")
            oa_works_count = float(rec.get("works_count", np.nan))
            oa_cited_by_count = float(rec.get("cited_by_count", np.nan))
            summary = rec.get("summary_stats", {})
            h_idx = summary.get("h_index")
            if h_idx is not None:
                h_index = float(h_idx)
            i10_idx = summary.get("i10_index")
            if i10_idx is not None:
                i10_index = float(i10_idx)

            # First year with works
            counts = rec.get("counts_by_year", [])
            for entry in sorted(counts, key=lambda x: x.get("year", 0)):
                if entry.get("works_count", 0) > 0:
                    oa_first_year = float(entry.get("year", np.nan))
                    break

            # Institution
            institutions = rec.get("last_known_institutions")
            if institutions and len(institutions) > 0:
                institution = institutions[0].get("display_name", "")

            # ORCID
            orcid = rec.get("ids", {}).get("orcid", "") or ""

            # Suspect
            oa_suspect = oa_works_count > SUSPECT_WORKS

        # Theme: argmax of mean W over this author's papers
        author_paper_indices = []
        for pid in author_papers["paper_id"].unique():
            df_idx = list(df[df["paper_id"] == pid].index)
            if df_idx:
                author_paper_indices.append(df_idx[0])

        theme_id = -1
        theme_label = ""
        theme_share = 0.0
        if author_paper_indices:
            W_rows = W[author_paper_indices, :]
            topic_ids_rows = topic_ids[author_paper_indices]
            theme_id, theme_share = author_theme(W_rows, topic_ids_rows)
            if theme_id >= 0 and theme_id < len(topics_df):
                theme_label = topics_df.iloc[theme_id]["topic_label"]

        author_dict = {
            "scopus_author_id": scopus_id,
            "scopus_name": scopus_name,
            "openalex_author_id": oa_id,
            "openalex_name": openalex_name,
            "match_method": match_method,
            "n_votes": n_votes,
            "n_papers": n_papers,
            "n_papers_fsr": n_papers_fsr,
            "n_papers_fpr": n_papers_fpr,
            "n_papers_jfcp": n_papers_jfcp,
            "cites_total": int(cites_total),
            "cites_fsr": int(cites_fsr),
            "cites_fpr": int(cites_fpr),
            "cites_jfcp": int(cites_jfcp),
            "first_year": first_year,
            "last_year": last_year,
            "oa_works_count": oa_works_count,
            "oa_cited_by_count": oa_cited_by_count,
            "h_index": h_index,
            "i10_index": i10_index,
            "oa_first_year": oa_first_year,
            "oa_suspect": oa_suspect,
            "institution": institution,
            "orcid": orcid,
            "theme_id": theme_id,
            "theme_label": theme_label,
            "theme_share": theme_share,
        }
        authors_list.append(author_dict)

    authors_df = pd.DataFrame(authors_list)
    assert len(authors_df) == 1953, f"Expected 1953 authors, got {len(authors_df)}"

    matched_count = sum(authors_df["openalex_author_id"] != "")
    suspect_count = sum(authors_df["oa_suspect"])

    print(
        f"  Matched: {matched_count}/{len(authors_df)} ({100*matched_count/len(authors_df):.1f}%)"
    )
    print(f"  Suspect matches: {suspect_count}")

    return authors_df


def selftest():
    """Run pure-function tests."""
    # norm_surname
    assert norm_surname("Café") == "cafe", "norm_surname accent"
    assert norm_surname("Smith, Jr.") == "smith", "norm_surname suffix"

    # parse_scopus_authors
    authors_id = "23968423300; 6508100573"
    authors_full = "Trainor, William J. (23968423300); Chhachhi, Indudeep (6508100573)"
    result = parse_scopus_authors(authors_id, authors_full)
    assert len(result) == 2 and result[0][0] == "23968423300", "parse_scopus_authors"

    # match_authorships
    scopus = [("123", "smith"), ("456", "jones")]
    authorships = [
        {"author": {"id": "A1"}, "raw_author_name": "Smith"},
        {"author": {"id": "A2"}, "raw_author_name": "Jones"},
    ]
    result = match_authorships(scopus, authorships)
    assert result.get("123") == "A1" and result.get("456") == "A2", "match_authorships"

    # title_matches
    assert title_matches(
        "The Quick Brown Fox", 2020, "the quick brown fox", 2020
    ), "title_matches exact"
    assert title_matches(
        "The Quick Brown Fox", 2020, "The Quick Brown", 2020
    ), "title_matches prefix"
    assert not title_matches(
        "Test Title", 2020, "Test Title", 2022
    ), "title_matches year"

    # prior_counts
    assert prior_counts(
        [
            {"year": 2010, "works_count": 1, "cited_by_count": 5},
            {"year": 2015, "works_count": 2, "cited_by_count": 10},
            {"year": 2020, "works_count": 1, "cited_by_count": 50},
        ],
        2016,
    ) == (3, 15), "prior_counts"
    assert prior_counts([], 2000) == (0, 0), "prior_counts empty"

    # clean_text
    assert clean_text("The Study of Funds; 2020", lambda t: t.rstrip("s")) == "fund", "clean_text"

    # author_theme
    result = author_theme(
        np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]), np.array([1, 0, 1])
    )
    assert result == (1, 2 / 3), f"author_theme: got {result}"
    result2 = author_theme(np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([0, 1]))
    assert result2 == (-1, 0.0), f"author_theme empty: got {result2}"

    # _cache_key (no mailto in key)
    key = _cache_key("/x", {"mailto": "a", "b": 1})
    assert "mailto" not in key, "_cache_key should exclude mailto"

    print("self-tests passed")
    return 0


def main():
    """Full pipeline."""
    _load_cache()

    try:
        df = load_scopus()

        # Fetch journals first: the only step that is never fully cached
        journals_df = fetch_journals()

        works_dict, resolution_status_dict = resolve_works(df)

        crosswalk = match_authors(df, works_dict, resolution_status_dict)

        oa_ids = set(
            crosswalk[crosswalk["openalex_author_id"] != ""]["openalex_author_id"]
        )
        records = fetch_author_records(sorted(oa_ids))

        W, topic_ids_arr, topics_df = topic_model(df)

        pa = build_paper_authors(df, crosswalk, records)

        authors_df = build_authors(
            df, pa, crosswalk, records, W, topic_ids_arr, topics_df
        )

        papers_df = build_papers(
            df,
            pa,
            W,
            topic_ids_arr,
            topics_df,
            journals_df,
            works_dict,
            resolution_status_dict,
        )

        os.makedirs("data", exist_ok=True)

        papers_df.to_csv("data/papers.csv", index=False)
        pa.to_csv("data/paper_authors.csv", index=False)
        authors_df.to_csv("data/authors.csv", index=False)
        journals_df["n_papers_corpus"] = [
            len(papers_df[papers_df["journal"] == j]) for j in ["FSR", "FPR", "JFCP"]
        ]
        journals_df.to_csv("data/journals.csv", index=False)
        topics_df.to_csv("data/topics.csv", index=False)
        print("Wrote papers, paper_authors, authors, journals and topics to data/")

        assert len(papers_df) == 1360, f"papers: expected 1360, got {len(papers_df)}"
        assert len(authors_df) == 1953, f"authors: expected 1953, got {len(authors_df)}"
        assert len(journals_df) == 3, f"journals: expected 3, got {len(journals_df)}"
        assert len(topics_df) == 11, f"topics: expected 11, got {len(topics_df)}"
        assert (
            len(pa) == papers_df["n_authors"].sum()
        ), f"paper_authors rows {len(pa)} != sum of n_authors {papers_df['n_authors'].sum()}"

        resolved = papers_df.groupby("journal")["resolution"].apply(lambda r: (r != "unresolved").mean())
        print(f"Resolved to OpenAlex: {resolved.round(3).to_dict()}")
        print(f"Authors matched: {(authors_df['openalex_author_id'] != '').mean():.3f}")
        print(f"Papers with a matched co-author: {(papers_df['n_authors_matched'] > 0).sum()} of {len(papers_df)}")
        print(f"Suspect profiles (> {SUSPECT_WORKS} works): {authors_df['oa_suspect'].sum()}")

    except BudgetExhausted as e:
        print(e)
        sys.exit(1)
    finally:
        _save_cache()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    else:
        main()
