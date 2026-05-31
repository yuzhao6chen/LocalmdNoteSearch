use crate::error::AppResult;
use crate::index::storage::escape_json;
use crate::search::SearchResult;

#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Text,
    Json,
}

pub fn print_results(results: &[SearchResult], format: OutputFormat) -> AppResult<()> {
    match format {
        OutputFormat::Text => {
            if results.is_empty() {
                println!("no matches");
                return Ok(());
            }

            for (index, result) in results.iter().enumerate() {
                println!(
                    "{}. {}  score {:.2}",
                    index + 1,
                    result.file_path,
                    result.score
                );
                println!("   title: {}", result.title);
                println!("   section: {}", result.section);
                if !result.tags.is_empty() {
                    println!("   tags: {}", result.tags.join(", "));
                }
                println!("   modified: {}", format_unix_time(result.modified));
                println!("   terms: {}", result.matched_terms.join(", "));
                println!("   {}", result.highlighted_snippet);
                println!();
            }
            Ok(())
        }
        OutputFormat::Json => {
            println!("{}", results_to_json(results));
            Ok(())
        }
    }
}

pub fn print_help() {
    println!(
        "md-knowsearch\n\
         \n\
         Commands:\n\
           md-knowsearch index <directory> [--cache <file>]\n\
           md-knowsearch rebuild <directory> [--cache <file>]\n\
           md-knowsearch search <keywords...> [--cache <file>] [--dir <directory>] [--rebuild] [--limit <n>] [--format text|json]\n\
           md-knowsearch tui <directory> [--cache <file>] [--limit <n>]\n\
         \n\
         Notes:\n\
           search with --dir scans that directory and refreshes the cache before querying.\n\
           tui opens a lightweight interactive terminal search interface.\n\
         \n\
         Examples:\n\
           cargo run -- index notes\n\
           cargo run -- search rust ownership borrowing\n\
           cargo run -- search \"error handling\" --dir notes --json\n\
           cargo run -- tui notes\n"
    );
}

fn results_to_json(results: &[SearchResult]) -> String {
    let mut output = String::from("[");
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push('{');
        push_json_field(&mut output, "file_path", &result.file_path);
        output.push(',');
        push_json_field(&mut output, "file_name", &result.file_name);
        output.push(',');
        push_json_field(&mut output, "title", &result.title);
        output.push(',');
        output.push_str("\"tags\":[");
        for (tag_index, tag) in result.tags.iter().enumerate() {
            if tag_index > 0 {
                output.push(',');
            }
            output.push('"');
            output.push_str(&escape_json(tag));
            output.push('"');
        }
        output.push(']');
        output.push(',');
        output.push_str("\"modified\":");
        output.push_str(&result.modified.to_string());
        output.push(',');
        push_json_field(&mut output, "section", &result.section);
        output.push_str(",\"score\":");
        output.push_str(&format!("{:.4}", result.score));
        output.push_str(",\"matched_terms\":[");
        for (term_index, term) in result.matched_terms.iter().enumerate() {
            if term_index > 0 {
                output.push(',');
            }
            output.push('"');
            output.push_str(&escape_json(term));
            output.push('"');
        }
        output.push(']');
        output.push(',');
        push_json_field(&mut output, "snippet", &result.snippet);
        output.push(',');
        push_json_field(&mut output, "highlighted_snippet", &result.marked_snippet);
        output.push('}');
    }
    output.push(']');
    output
}

fn format_unix_time(seconds: u64) -> String {
    if seconds == 0 {
        "unknown".to_string()
    } else {
        format!("{seconds}")
    }
}

fn push_json_field(output: &mut String, key: &str, value: &str) {
    output.push('"');
    output.push_str(key);
    output.push_str("\":\"");
    output.push_str(&escape_json(value));
    output.push('"');
}
