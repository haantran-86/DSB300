# DSB300 Capstone: Journal Citation Analytics

Group 15, DSB300 Data Science Capstone, QUT, Semester 2 2026. Industry partner: the Academy of Financial Services (AFS).

The project asks how *Financial Services Review* (FSR) can attract more citations. It compares FSR with its two closest comparators, *Financial Planning Review* (FPR) and the *Journal of Financial Counseling and Planning* (JFCP), using Scopus records for 1,362 articles and author and citation histories from OpenAlex. The work is split into four questions:

| Question | Topic | Notebook |
|---|---|---|
| A | Are citations concentrated among a few prolific or well-known authors, and does an author's track record predict how their FSR articles perform? | `notebook/question_A_star_authors.ipynb` |
| B | Which research topics are associated with more citations, and where is FSR under-represented? | `notebook/question_B_topic_modelling.ipynb` |
| C | Is abstract readability associated with citations? | `notebook/question_C_readability.ipynb` |
| D | When do citations peak, and which articles become high hitters? | `notebook/analysis_question_D.ipynb` |

`analysis.ipynb`, `analysis_v2.ipynb`, `analysis_v3.ipynb` and `analysis_final.ipynb` in `notebook/` are other versions of the Question D analysis.

## Repository layout

```
data/
  master_dataset.csv                 Scopus export for FSR, FPR and JFCP (1,362 articles, citations as at September 2026)
  fsr_citations_by_year_openalex.csv citations per year for FSR articles from OpenAlex (Question D)
  papers.csv, paper_authors.csv,     Question A tables built by build_dataset.py
  authors.csv, journals.csv,
  topics.csv
  openalex_cache.json                cached OpenAlex API responses used by build_dataset.py
notebook/                            one notebook per question (see above)
build_dataset.py                     builds the Question A tables from the Scopus export and OpenAlex
DATA.md                              how the Question A tables were collected and constructed, column by column
DSB300 Projects 2026.pdf             project brief
```

## Running the code

The notebooks read their data from `../data`, so run them from the `notebook/` folder (Jupyter does this by default when you open a notebook there).

```
pip install pandas numpy scipy statsmodels scikit-learn nltk matplotlib seaborn textstat jupyter
python -m nltk.downloader wordnet
```

Question B also needs `gensim` (topic coherence) and `bertopic`, `sentence-transformers` and `umap-learn` (the BERTopic comparison). The first run downloads the `all-MiniLM-L6-v2` sentence embedding model.

Questions A and C were last run with Python 3.13, pandas 3.0, statsmodels 0.15, scikit-learn 1.9 and textstat 0.7.13, and reproduce the numbers in the final report. Question B was run with Python 3.12. BERTopic results can vary slightly across platforms because UMAP is not fully deterministic, but the final model is NMF, which is fixed by its random seed.

The Question A tables are committed, so the notebook runs without network access. To rebuild them, run from the repository root:

```
python build_dataset.py --selftest
python build_dataset.py
```

The build reads `data/openalex_cache.json` and only calls the OpenAlex API for requests that are not cached. It reads an API key from the `OPENALEX_API_KEY` environment variable or a git-ignored `.openalex_key` file. See `DATA.md` for details.

## Shared choices across questions

All four questions use the same topic partition: Question B's NMF model with eleven topics, fitted on the title, abstract and author keywords of every article. Questions A and C rebuild it from Question B's code, and Question A checks that the rebuilt model gives every article the same topic. Citation counts are heavily overdispersed, so the citation count models are negative binomial regressions reporting incidence rate ratios, and they control for log article age because citations build up quickly at first and then slow down.

## Outputs

Question B saves its figures and result tables (PNG and CSV) next to the notebook. Question C saves its figures to `notebook/output/`.
