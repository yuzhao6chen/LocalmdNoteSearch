# md-knowsearch

`md-knowsearch` is a Rust command-line tool for searching a local Markdown/TXT
knowledge base. It scans a user-selected directory, parses document metadata,
builds a local inverted index, stores a cache, and returns ranked search
results with section hits, context snippets, and highlighted keywords.

The project intentionally avoids large search-engine libraries. File scanning,
Markdown/TXT parsing, tokenization, inverted-index construction, ranking, JSONL
cache serialization, and highlighting are implemented in Rust code inside this
repository.

## Features

- Recursively scan `.md`, `.markdown`, and `.txt` files.
- Extract path, file name, title, headings, body, tags, sections, and modified
  time.
- Parse Markdown headings, simple front matter `title` / `tags`, inline
  `#tags`, and plain-text `Tags:` lines.
- Ignore Markdown headings inside fenced code blocks.
- Build an inverted index from title, tags, headings, file name, and body.
- Cache parsed documents as JSON Lines in `.md-knowsearch-cache.jsonl`.
- Validate cache version when loading a saved index.
- Search one or more keywords.
- Rank results using title/tag/heading/body/file-name weights, term frequency,
  inverse document frequency, phrase bonus, multi-keyword coverage, and a small
  recency bonus.
- Show matching file, title, section, tags, modified time, score, matched
  terms, nearby context, and terminal highlighting.
- Output either plain text or JSON.

## Usage

Build an index:

```bash
cargo run -- index ./notes
```

Search the existing cache:

```bash
cargo run -- search rust ownership borrowing
```

Scan a directory and search in one command. This refreshes the cache before
running the query:

```bash
cargo run -- search rust error handling --dir ./notes
```

Require an explicit rebuild while searching:

```bash
cargo run -- search rust error handling --dir ./notes --rebuild
```

Return JSON output:

```bash
cargo run -- search rust ownership --format json
```

Use a custom cache file:

```bash
cargo run -- index ./notes --cache ./tmp/notes-cache.jsonl
cargo run -- search markdown parser --cache ./tmp/notes-cache.jsonl
```

## Commands

```text
md-knowsearch index <directory> [--cache <file>]
md-knowsearch rebuild <directory> [--cache <file>]
md-knowsearch search <keywords...> [--cache <file>] [--dir <directory>] [--rebuild] [--limit <n>] [--format text|json]
```

## Architecture

The code is organized around four major modules:

- `ingest`: scans directories, parses Markdown/TXT documents, extracts
  metadata, protects fenced code blocks, and tokenizes text.
- `index`: builds the inverted index and saves/loads the JSONL cache.
- `search`: parses queries, scores documents, chooses the best hit section,
  creates snippets, and highlights terms.
- `app`: handles command-line parsing and text/JSON output.

Shared models and error types live in `model` and `error`.

## Ranking Model

The score is not a simple string match. For every query token, the search engine
combines:

- field weight: title > tag > heading > file name > body
- term frequency in each field
- inverse document frequency
- bonus for matching all query keywords
- bonus for exact multi-word phrase matches
- small recency bonus from file modification time

This keeps title and tag matches prominent while still allowing repeated body
matches to matter.

## Engineering Practices

- Rust 2024 edition.
- Modular source layout.
- `struct`, `enum`, and `trait` are used for document models, fields, errors,
  tokenization, and search internals.
- Errors are returned with `Result` and a project-level `AppError`.
- Cache serialization is implemented without external dependencies.
- Unit tests cover tokenization, parsing, indexing, highlighting, ranking, and
  an end-to-end search flow.

## Checks

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
```
