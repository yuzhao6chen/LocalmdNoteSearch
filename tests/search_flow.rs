use md_knowsearch::index::InvertedIndex;
use md_knowsearch::ingest::MarkdownParser;
use md_knowsearch::search::SearchEngine;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn end_to_end_search_returns_ranked_markdown_result() {
    let parser = MarkdownParser;
    let document = parser.parse(
        0,
        Path::new("notes/rust.md"),
        "# Rust Ownership\n\nTags: #rust\n\n## Borrowing\nOwnership enables memory safety.\n"
            .to_string(),
        0,
    );
    let index = InvertedIndex::build(vec![document]);
    let engine = SearchEngine::new(&index);
    let results = engine.search(&["ownership".to_string(), "rust".to_string()], 5);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].title, "Rust Ownership");
    assert!(results[0].score > 0.0);
    assert!(results[0].snippet.to_lowercase().contains("ownership"));
}

#[test]
fn cli_can_index_and_search_json_results() -> Result<(), Box<dyn std::error::Error>> {
    let root = unique_test_dir("cli-flow");
    std::fs::create_dir_all(&root)?;
    std::fs::write(
        root.join("rust.md"),
        "# Rust Ownership\n\nTags: #rust #memory\n\n## Borrowing\nOwnership supports safe borrowing.\n",
    )?;
    std::fs::write(
        root.join("other.txt"),
        "Cooking notes\nTags: #kitchen\nBody text\n",
    )?;

    let cache = root.join("cache.jsonl");
    let binary = env!("CARGO_BIN_EXE_md-knowsearch");

    let index_output = Command::new(binary)
        .arg("index")
        .arg(&root)
        .arg("--cache")
        .arg(&cache)
        .output()?;
    assert!(index_output.status.success());

    let search_output = Command::new(binary)
        .arg("search")
        .arg("rust")
        .arg("ownership")
        .arg("--cache")
        .arg(&cache)
        .arg("--format")
        .arg("json")
        .output()?;

    assert!(search_output.status.success());
    let stdout = String::from_utf8(search_output.stdout)?;
    assert!(stdout.contains("rust.md"));
    assert!(stdout.contains("\"matched_terms\":[\"ownership\",\"rust\"]"));
    assert!(stdout.contains("<mark>Ownership</mark>"));

    Ok(())
}

#[test]
fn cli_search_dir_refreshes_cache() -> Result<(), Box<dyn std::error::Error>> {
    let root = unique_test_dir("cli-dir-refresh");
    std::fs::create_dir_all(&root)?;
    std::fs::write(
        root.join("fresh.md"),
        "# Fresh Note\n\nA unique keyword appears here.\n",
    )?;

    let cache = root.join("cache.jsonl");
    let binary = env!("CARGO_BIN_EXE_md-knowsearch");

    let output = Command::new(binary)
        .arg("search")
        .arg("unique")
        .arg("--dir")
        .arg(&root)
        .arg("--cache")
        .arg(&cache)
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("fresh.md"));
    assert!(cache.exists());

    Ok(())
}

fn unique_test_dir(name: &str) -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("target")
        .join(format!("{name}-{suffix}"))
}
