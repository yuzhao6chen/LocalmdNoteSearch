use std::io::{self, Write};
use std::path::Path;

use crate::error::{AppError, AppResult};
use crate::index::InvertedIndex;
use crate::index::storage;
use crate::ingest::scan_documents;
use crate::search::{SearchEngine, SearchResult};

// 简单的终端交互循环，输入一行就搜索一次。
pub fn run(dir: &Path, cache: &Path, limit: usize) -> AppResult<()> {
    let mut index = build_index(dir, cache)?;
    draw_home(dir, cache, limit, index.document_count())?;

    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("search> ");
        flush_stdout()?;

        input.clear();
        let bytes = stdin
            .read_line(&mut input)
            .map_err(|error| AppError::io("failed to read terminal input", error))?;
        if bytes == 0 {
            break;
        }

        match parse_action(&input) {
            TuiAction::Empty => {}
            TuiAction::Quit => {
                println!("bye");
                break;
            }
            TuiAction::Help => draw_help(),
            TuiAction::Clear => draw_home(dir, cache, limit, index.document_count())?,
            TuiAction::Rebuild => {
                index = build_index(dir, cache)?;
                println!(
                    "reindexed {} documents and refreshed {}",
                    index.document_count(),
                    cache.display()
                );
            }
            TuiAction::Search(query) => {
                let terms = split_query(&query);
                let engine = SearchEngine::new(&index);
                let results = engine.search(&terms, limit);
                draw_results(&query, &results);
            }
        }
    }

    Ok(())
}

// 启动和 :rebuild 都走这里，避免两套索引逻辑。
fn build_index(dir: &Path, cache: &Path) -> AppResult<InvertedIndex> {
    let documents = scan_documents(dir)?;
    storage::save_cache(cache, dir, &documents)?;
    Ok(InvertedIndex::build(documents))
}

fn draw_home(dir: &Path, cache: &Path, limit: usize, document_count: usize) -> AppResult<()> {
    clear_screen()?;
    println!("md-knowsearch TUI");
    println!("=================");
    println!("directory : {}", dir.display());
    println!("cache     : {}", cache.display());
    println!("documents : {document_count}");
    println!("limit     : {limit}");
    println!();
    println!("Type keywords and press Enter to search.");
    println!("Commands: :help  :rebuild  :clear  :quit");
    println!();
    Ok(())
}

fn draw_help() {
    println!();
    println!("TUI commands");
    println!("  所有权 借用          搜索 Rust 所有权与借用笔记");
    println!("  搜索 排序            搜索倒排索引和相关度排序");
    println!("  模块 架构            搜索项目模块设计");
    println!("  测试 clippy          搜索工程质量检查");
    println!("  :rebuild             重新扫描目录并刷新缓存");
    println!("  :clear               重绘界面");
    println!("  :quit                退出");
    println!();
}

fn draw_results(query: &str, results: &[SearchResult]) {
    println!();
    println!("Results for `{query}`");
    println!("---------------------");

    if results.is_empty() {
        println!("no matches");
        println!();
        return;
    }

    for (index, result) in results.iter().enumerate() {
        println!(
            "{}. {}  score {:.2}",
            index + 1,
            result.file_path,
            result.score
        );
        println!("   title   : {}", result.title);
        println!("   section : {}", result.section);
        if !result.tags.is_empty() {
            println!("   tags    : {}", result.tags.join(", "));
        }
        println!("   terms   : {}", result.matched_terms.join(", "));
        println!("   snippet : {}", result.highlighted_snippet);
        println!();
    }
}

fn clear_screen() -> AppResult<()> {
    print!("\x1b[2J\x1b[H");
    flush_stdout()
}

fn flush_stdout() -> AppResult<()> {
    io::stdout()
        .flush()
        .map_err(|error| AppError::io("failed to flush terminal output", error))
}

fn split_query(query: &str) -> Vec<String> {
    query.split_whitespace().map(str::to_string).collect()
}

fn parse_action(input: &str) -> TuiAction {
    let trimmed = input.trim();
    match trimmed {
        "" => TuiAction::Empty,
        ":q" | ":quit" | "quit" | "exit" => TuiAction::Quit,
        ":h" | ":help" | "help" => TuiAction::Help,
        ":c" | ":clear" | "clear" => TuiAction::Clear,
        ":r" | ":rebuild" | "rebuild" => TuiAction::Rebuild,
        query => TuiAction::Search(query.to_string()),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum TuiAction {
    Empty,
    Quit,
    Help,
    Clear,
    Rebuild,
    Search(String),
}

#[cfg(test)]
mod tests {
    use super::{TuiAction, parse_action, split_query};

    #[test]
    fn parses_tui_commands() {
        assert_eq!(parse_action("  "), TuiAction::Empty);
        assert_eq!(parse_action(":quit"), TuiAction::Quit);
        assert_eq!(parse_action(":rebuild"), TuiAction::Rebuild);
        assert_eq!(parse_action(":clear"), TuiAction::Clear);
        assert_eq!(parse_action(":help"), TuiAction::Help);
    }

    #[test]
    fn parses_search_action() {
        assert_eq!(
            parse_action("rust ownership"),
            TuiAction::Search("rust ownership".to_string())
        );
        assert_eq!(
            split_query("rust ownership borrowing"),
            vec!["rust", "ownership", "borrowing"]
        );
    }
}
