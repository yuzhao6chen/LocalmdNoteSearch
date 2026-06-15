mod cli;
mod output;
mod tui;

use std::env;
use std::path::Path;

use crate::error::{AppError, AppResult};
use crate::index::InvertedIndex;
use crate::index::storage;
use crate::ingest::scan_documents;
use crate::search::SearchEngine;

use self::cli::{Command, Options};

pub fn run() -> AppResult<()> {
    let options = Options::parse(env::args().skip(1))?;

    match options.command {
        Command::Index { dir } => build_and_save(&dir, &options.cache),
        Command::Rebuild { dir } => build_and_save(&dir, &options.cache),
        Command::Search {
            query,
            dir,
            limit,
            format,
            rebuild,
        } => {
            let documents = load_documents_for_search(dir.as_deref(), &options.cache, rebuild)?;
            let index = InvertedIndex::build(documents);
            let engine = SearchEngine::new(&index);
            let results = engine.search(&query, limit);
            output::print_results(&results, format)
        }
        Command::Tui { dir, limit } => tui::run(&dir, &options.cache, limit),
        Command::Help => {
            output::print_help();
            Ok(())
        }
    }
}

// index 和 rebuild 都是同一件事：扫描目录后写入缓存。
fn build_and_save(dir: &Path, cache: &Path) -> AppResult<()> {
    let documents = scan_documents(dir)?;
    storage::save_cache(cache, dir, &documents)?;
    println!(
        "indexed {} documents and saved cache to {}",
        documents.len(),
        cache.display()
    );
    Ok(())
}

// search 可以直接读缓存，也可以先扫描目录刷新缓存。
fn load_documents_for_search(
    dir: Option<&Path>,
    cache: &Path,
    rebuild: bool,
) -> AppResult<Vec<crate::model::Document>> {
    if let Some(scan_dir) = dir {
        let documents = scan_documents(scan_dir)?;
        storage::save_cache(cache, scan_dir, &documents)?;
        return Ok(documents);
    }

    if rebuild {
        return Err(AppError::Cli(
            "`--rebuild` on search also needs `--dir <directory>`".to_string(),
        ));
    }

    if !cache.exists() {
        return Err(AppError::Cli(
            "cache is missing; pass --dir <directory> or run `md-knowsearch index` first"
                .to_string(),
        ));
    }

    storage::load_cache(cache)
}
