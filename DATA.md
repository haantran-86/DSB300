# Data for Question A: how the dataset was collected and built

This document describes every step between the raw Scopus export (`data/master_dataset.csv`) and the five analysis tables in `data/`. The whole pipeline is `build_dataset.py`; nothing in `notebook/question_A_star_authors.ipynb` recomputes any of it.

## 1. Purpose

Question A asks whether citations in Financial Services Review (FSR) are concentrated among a few prolific or well-known authors, and whether an author's prior track record predicts how their FSR papers perform. Answering it needs (a) a paper table with citation counts and team-level track-record measures, (b) an author table with career statistics from outside the three-journal corpus, (c) a journal table with an impact-factor style metric, and (d) a topic assignment for every paper so that topic can be held constant when track record is tested.

## 2. Sources

### 2.1 Scopus export (`master_dataset.csv`)

A Scopus CSV export of every record in three journals: Financial Services Review (FSR), Financial Planning Review (FPR) and Journal of Financial Counseling and Planning (JFCP). 1362 rows, years 1990 to 2026, Scopus citation counts as at September 2026.

Fields used: `authors_full_name` (surnames, initials and Scopus author ids, e.g. `Trainor, William J. (23968423300); ...`), `authors_id` (the same ids, `;`-separated), `title`, `year`, `source_title`, `cited_by`, `doi` (306 rows have none), `abstract`, `author_keywords` (135 rows have none). Two rows have no `authors_id` and cannot be attributed to any author, so they are dropped: 1360 papers remain (FSR 625, FPR 111, JFCP 624) with 1953 distinct Scopus author ids.

### 2.2 OpenAlex API

OpenAlex (https://api.openalex.org) is free and covers author careers across all journals, which the Scopus export cannot. Three endpoints are used:

| Endpoint | Purpose | `select` fields |
|---|---|---|
| `/works` | Resolve each Scopus paper to an OpenAlex work so that its author list can be read | `id,doi,title,publication_year,authorships` |
| `/authors` | Career statistics for every matched author | `id,display_name,works_count,cited_by_count,summary_stats,last_known_institutions,counts_by_year,ids` |
| `/sources` | Journal-level statistics for the three journals | `id,display_name,issn_l,issn,summary_stats,works_count,cited_by_count,counts_by_year` |

Every response is cached in `data/openalex_cache.json`, keyed by endpoint path plus the sorted query parameters (the `api_key` and `mailto` parameters are excluded from the key so a run with or without a key hits the same cache). The cache holds 643 responses and is committed, so the build is reproducible offline. Works and authors were fetched on 2026-09-13; sources on 2026-09-16.

The anonymous API budget is about 100 requests per day and resets at midnight UTC. `build_dataset.py` reads an API key from the `OPENALEX_API_KEY` environment variable or a `.openalex_key` file (git-ignored). When the budget runs out mid-run the script stops calling the network: papers whose title search was not yet cached are marked `unresolved` and the run completes from cache. 23 JFCP papers whose titles contain a question mark were never fetched for this reason and are part of the 120 unresolved JFCP papers.

## 3. Pipeline steps

In the order `main()` runs them.

**`load_scopus`.** Reads the export, drops the 2 rows without author ids, resets the index so `paper_id` is 0 to 1359, maps `source_title` to the codes FSR / FPR / JFCP, casts `year` and `cited_by` to integers, counts `n_authors` from `authors_id`, and sets `paper_age = 2026 - year + 1` so a 2026 paper has age 1.

**`fetch_journals`.** One `/sources` request filtered on the three ISSNs (FSR 1057-0810, FPR 2573-8615, JFCP 1052-3073). Each result is matched back on `issn_l`, and the OpenAlex source ids are asserted to be S102700472, S4210200979 and S2912066455. `two_yr_mean_citedness` is OpenAlex's version of the two-year journal impact factor (citations received in a year to the previous two years' papers, divided by the number of those papers). This step runs first because it is the only one that may need the network on every machine.

**`resolve_works`.** Papers with a single clean DOI are sent to `/works` in batches of 50 (`filter=doi:a|b|...`), in paper order so that the batch composition, and therefore the cache key, is identical on every run. Papers with no DOI, or whose DOI OpenAlex does not know, fall back to `filter=title.search:<title>` (5 results). A result is accepted when `title_matches` holds: the two titles, lower-cased and stripped of punctuation, are equal or one is a prefix of the other, and the publication years differ by at most one. Each paper ends with `resolution` = `doi`, `title` or `unresolved`.

**`match_authors`.** Scopus and OpenAlex use different author ids, so the two author lists of each resolved paper are aligned by surname. `parse_scopus_authors` pairs each Scopus id with its normalised surname (accents folded, suffixes such as Jr. removed, lower-case letters and digits only). `match_authorships` walks the Scopus list left to right and, for each surname, takes the unclaimed OpenAlex authorship with the same normalised surname closest to the same position. Each (Scopus id, OpenAlex id) pair found on a paper is one vote; after all papers are processed the OpenAlex id with the most votes wins for each Scopus id. Output: `author_match_crosswalk` with `openalex_author_id` (empty when no paper gave a match), `match_method` (`doi` if any vote came from a DOI-resolved paper, else `title`, or `unmatched`), `n_votes`.

**`fetch_author_records`.** The matched OpenAlex ids are fetched from `/authors` in batches of 50. Each record carries `works_count`, `cited_by_count`, `summary_stats` (`h_index`, `i10_index`, `2yr_mean_citedness`), `last_known_institutions`, `ids.orcid` and `counts_by_year`, a list of `{year, works_count, cited_by_count}` covering the author's full history (`cited_by_count` here is citations received in that year to any of the author's works).

**`topic_model`.** Reproduces Question B's final topic model (`notebook/question_B_topic_modelling.ipynb`) so Questions A and B share one set of topics. For each of the 1,362 rows of `master_dataset.csv` the text is `title + abstract + author_keywords` (`;` replaced by spaces), lower-cased, reduced to letters, with English stop words plus Question B's research and copyright words (`study`, `results`, `rights`, `reserved`, ...) removed and every token lemmatised with NLTK's WordNet lemmatiser. `TfidfVectorizer` with 1- to 3-grams, `min_df=5`, `max_df=0.85`. `NMF(n_components=11, init="nndsvda", random_state=42, max_iter=500)` is fitted on all 1,362 rows and the weights are then restricted to the 1,360 papers kept here. Each paper's weight vector is normalised to sum to one and its topic is the largest weight (every paper has text, so none is `-1`). Labels are the ones in the report's Table B1; the build asserts that each component's first top term matches Question B's, so a drift in the fit fails loudly.

**`build_paper_authors`.** One row per (paper, author) in byline order, 3336 rows. Each row carries the author's OpenAlex id from the crosswalk and, when the OpenAlex record exists, `prior_works` and `prior_cites` from `counts_by_year` restricted to years strictly before the paper's `year`, and the author's current `h_index`. Unmatched authors have these three fields empty.

**`build_authors`.** Groups `paper_authors` by Scopus id: paper and citation counts overall and per journal (full counting, each co-author credited the paper's whole `cited_by`), first and last corpus year, then the crosswalk columns, the OpenAlex career fields, `oa_first_year` (earliest year in `counts_by_year` with at least one work), the first `last_known_institutions` entry, ORCID, the `oa_suspect` flag and the theme (section 4).

**`build_papers`.** Adds to each paper the team track-record maxima over its matched co-authors, `n_authors_matched`, the topic id, label and full weight vector, the journal's `two_yr_mean_citedness` and the OpenAlex work id and resolution method. Writes the five CSVs to `data/` and asserts 1360 papers, 3336 paper-author rows, 1953 authors, 3 journals and 11 topics.

## 4. Author-level construction rules

**Point-in-time prior counts.** For an author on a paper published in year *Y*, `prior_works` = sum of `counts_by_year[].works_count` over years < *Y* and `prior_cites` = sum of `counts_by_year[].cited_by_count` over years < *Y*. Both are career-wide (all journals OpenAlex indexes) and contain nothing that happened in or after the paper's year, so they cannot leak the paper's own success into its predictor.

**Team maximum.** `team_prior_works_max`, `team_prior_cites_max` and `team_h_index_max` are the maximum over the paper's matched co-authors, regardless of byline position. The question they encode is "did this paper have at least one established author". Papers with no matched co-author (43 of 1360) have these fields empty.

**h-index is a snapshot.** OpenAlex only publishes the current h-index, so `h_index` (author table, paper-author table and `team_h_index_max`) is as at September 2026 and includes reputation the author built after the paper appeared. The notebook treats it as an upper bound and reports a leak-free variant that uses only `prior_cites`.

**Theme.** An author's theme is the largest component of the mean normalised topic-weight vector over their papers in the corpus, and `theme_share` is the fraction of their papers whose own topic equals that theme. Authors with one paper have `theme_share = 1`.

**Suspect profiles.** OpenAlex occasionally merges several people under one author record. `oa_suspect` flags matched authors with more than 300 works (53 authors). They are kept in every table; the notebook refits the main model without their papers as a sensitivity check.

## 5. Data dictionary

### `data/papers.csv` (1360 rows, one per paper)

| Column | Type | Meaning |
|---|---|---|
| `paper_id` | int | Row index in the cleaned Scopus export, 0 to 1359 |
| `journal` | str | `FSR`, `FPR` or `JFCP` |
| `year` | int | Publication year |
| `title` | str | Title as exported |
| `doi` | str | DOI, empty if Scopus has none |
| `cited_by` | int | Scopus citation count, September 2026 (the outcome) |
| `n_authors` | int | Number of Scopus author ids on the paper |
| `paper_age` | int | `2026 - year + 1` |
| `openalex_work_id` | str | OpenAlex work URL, empty if unresolved |
| `resolution` | str | `doi`, `title` or `unresolved` |
| `n_authors_matched` | int | Co-authors with an OpenAlex id |
| `team_prior_works_max` | float | Max over matched co-authors of works published before `year`; empty if none matched |
| `team_prior_cites_max` | float | Max over matched co-authors of citations received before `year`; empty if none matched |
| `team_h_index_max` | float | Max over matched co-authors of current h-index; empty if none matched |
| `topic_id` | int | NMF topic 0 to 10, `-1` if no text |
| `topic_label` | str | Topic label (report Table B1) |
| `topic_w0` ... `topic_w10` | float | Normalised NMF weights, sum to 1 |
| `journal_2yr_citedness` | float | The journal's `two_yr_mean_citedness` |

### `data/paper_authors.csv` (3336 rows, one per paper and author)

| Column | Type | Meaning |
|---|---|---|
| `paper_id` | int | Foreign key to `papers` |
| `scopus_author_id` | str | Scopus author id |
| `author_position` | int | 1-based byline position |
| `openalex_author_id` | str | OpenAlex author URL, empty if unmatched |
| `prior_works` | float | Works before the paper's year; empty if unmatched |
| `prior_cites` | float | Citations received before the paper's year; empty if unmatched |
| `h_index` | float | Current h-index; empty if unmatched |

### `data/authors.csv` (1953 rows, one per Scopus author id)

| Column | Type | Meaning |
|---|---|---|
| `scopus_author_id` | str | Scopus author id |
| `scopus_name` | str | Surname as exported by Scopus |
| `openalex_author_id` | str | OpenAlex author URL, empty if unmatched |
| `openalex_name` | str | OpenAlex display name |
| `match_method` | str | `doi`, `title` or `unmatched` |
| `n_votes` | int | Papers on which the winning OpenAlex id was matched |
| `n_papers` | int | Papers in the corpus |
| `n_papers_fsr`, `n_papers_fpr`, `n_papers_jfcp` | int | Papers per journal |
| `cites_total` | int | Sum of `cited_by` over the author's papers (full counting) |
| `cites_fsr`, `cites_fpr`, `cites_jfcp` | int | The same per journal |
| `first_year`, `last_year` | int | First and last corpus paper |
| `oa_works_count` | float | OpenAlex career works; empty if unmatched |
| `oa_cited_by_count` | float | OpenAlex career citations; empty if unmatched |
| `h_index`, `i10_index` | float | OpenAlex `summary_stats`, September 2026 |
| `oa_first_year` | float | Earliest year with a work in `counts_by_year` |
| `oa_suspect` | bool | `oa_works_count > 300` |
| `institution` | str | First `last_known_institutions` entry |
| `orcid` | str | ORCID URL if OpenAlex has one |
| `theme_id` | int | Dominant topic 0 to 10, `-1` if none |
| `theme_label` | str | Label of that topic |
| `theme_share` | float | Share of the author's papers whose topic is the theme |

### `data/journals.csv` (3 rows)

| Column | Type | Meaning |
|---|---|---|
| `journal` | str | `FSR`, `FPR`, `JFCP` |
| `source_title` | str | Full journal name |
| `openalex_source_id` | str | OpenAlex source id |
| `issn_l` | str | Linking ISSN |
| `two_yr_mean_citedness` | float | OpenAlex two-year mean citedness (impact factor proxy) |
| `h_index`, `i10_index` | int | Journal-level OpenAlex `summary_stats` |
| `works_count`, `cited_by_count` | int | OpenAlex totals for the journal |
| `n_papers_corpus` | int | Papers in the Scopus export |

### `data/topics.csv` (11 rows)

| Column | Type | Meaning |
|---|---|---|
| `topic_id` | int | 0 to 10 |
| `topic_label` | str | Topic label (report Table B1) |
| `top_terms` | str | Top twelve terms, comma-separated |
| `n_papers` | int | Papers whose largest weight is this topic |
| `n_papers_fsr`, `n_papers_fpr`, `n_papers_jfcp` | int | The same per journal |

## 6. Coverage

| Step | Result |
|---|---|
| Scopus rows | 1362 in, 2 dropped (no author id), 1360 kept: FSR 625, FPR 111, JFCP 624; years 1990 to 2026 |
| Papers resolved to OpenAlex | 1239 of 1360. FSR 623 by DOI, 1 by title, 1 unresolved (99.8%). FPR 111 by DOI (100%). JFCP 318 by DOI, 186 by title, 120 unresolved (80.8%) |
| Authors matched to OpenAlex | 1787 of 1953 (91.5%), mapping to 1753 distinct OpenAlex ids; 32 OpenAlex ids are claimed by more than one Scopus id (Scopus splits some people across ids); 3 OpenAlex records could not be fetched before the daily budget ran out |
| Career history | 1309 of 1750 fetched authors (74.8%) have a first work in 2010 or earlier, so prior counts are meaningful for most of the corpus |
| Papers with at least one matched author | 1317 of 1360 (96.8%): FSR 617 of 625, FPR 111 of 111, JFCP 589 of 624 |
| Suspect (merged) OpenAlex profiles | 53 authors |
| Topics | 8, sizes 116 / 193 / 170 / 234 / 94 / 221 / 82 / 249; 1 paper with no text |
| Paper-author rows | 3336 |
| Cached API responses | 643 |

## 7. Limitations

- **Snapshot h-index.** `h_index` cannot be reconstructed for the publication year, so any association with it partly reflects reputation gained afterwards. `prior_works` and `prior_cites` do not have this problem.
- **Unmatched authors.** 8.5% of Scopus authors have no OpenAlex id, mostly on the 120 unresolved JFCP papers and on papers where surnames did not align. Their team fields are empty and the 43 papers with no matched co-author drop out of the model.
- **Surname matching.** Two authors with the same normalised surname on one paper are assigned by byline position; a wrong assignment on one paper is outvoted when the author appears on others, but single-paper authors have no such protection.
- **Full counting.** Every co-author gets the paper's whole citation count, which inflates totals for authors who work in large teams.
- **Two citation indexes.** The outcome `cited_by` is Scopus; the predictors come from OpenAlex, which counts citations differently.
- **Merged OpenAlex profiles.** 53 flagged; there will be undetected cases below the 300-work threshold.
- **Topic model.** NMF with 11 components (Question B's final model) is one reasonable partition of the corpus, not the only one; topics are used as controls, not as findings in themselves (that is Question B).

## 8. Reproduce

```
python build_dataset.py --selftest      # unit checks on the pure functions, no network
python build_dataset.py                 # rebuilds data/ from data/master_dataset.csv and data/openalex_cache.json
jupyter nbconvert --to notebook --execute --inplace notebook/question_A_star_authors.ipynb
```

The build is deterministic given the cache. Every response it needs is cached except the outstanding JFCP title searches and the 3 missing author records, which it retries on each run and skips if the budget is exhausted or there is no network. To fetch them, put an OpenAlex API key in `.openalex_key` and rerun; new responses are appended to the cache.
