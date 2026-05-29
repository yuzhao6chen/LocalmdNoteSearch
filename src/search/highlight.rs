use crate::ingest::{SimpleTokenizer, Tokenizer};

pub fn make_snippet(text: &str, terms: &[String], radius: usize) -> String {
    let lower_text = text.to_lowercase();
    let best = terms
        .iter()
        .filter_map(|term| lower_text.find(term).map(|position| (position, term.len())))
        .min_by_key(|(position, _)| *position);

    let Some((position, length)) = best else {
        return text.chars().take(radius * 2).collect::<String>();
    };

    let start = char_floor(text, position.saturating_sub(radius));
    let end = char_ceil(text, (position + length + radius).min(text.len()));
    let mut snippet = String::new();
    if start > 0 {
        snippet.push_str("...");
    }
    if let Some(slice) = text.get(start..end) {
        snippet.push_str(slice);
    }
    if end < text.len() {
        snippet.push_str("...");
    }
    snippet
}

pub fn highlight_with_ansi(text: &str, terms: &[String]) -> String {
    highlight(text, terms, "\x1b[1;33m", "\x1b[0m")
}

pub fn highlight_with_marks(text: &str, terms: &[String]) -> String {
    highlight(text, terms, "<mark>", "</mark>")
}

fn highlight(text: &str, terms: &[String], open: &str, close: &str) -> String {
    if terms.is_empty() {
        return text.to_string();
    }

    let tokenizer = SimpleTokenizer;
    let term_set: std::collections::HashSet<String> = terms.iter().cloned().collect();
    let mut output = String::new();
    let mut current = String::new();

    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            current.push(character);
        } else {
            flush_token(
                &mut output,
                &mut current,
                &term_set,
                open,
                close,
                &tokenizer,
            );
            if character.is_alphanumeric() && !character.is_ascii() {
                let token = character.to_string();
                if term_set.contains(&token) {
                    output.push_str(open);
                    output.push(character);
                    output.push_str(close);
                } else {
                    output.push(character);
                }
            } else {
                output.push(character);
            }
        }
    }
    flush_token(
        &mut output,
        &mut current,
        &term_set,
        open,
        close,
        &tokenizer,
    );
    output
}

fn flush_token(
    output: &mut String,
    current: &mut String,
    term_set: &std::collections::HashSet<String>,
    open: &str,
    close: &str,
    tokenizer: &SimpleTokenizer,
) {
    if current.is_empty() {
        return;
    }
    let normalized = tokenizer.tokenize(current).join("");
    if term_set.contains(&normalized) {
        output.push_str(open);
        output.push_str(current);
        output.push_str(close);
    } else {
        output.push_str(current);
    }
    current.clear();
}

fn char_floor(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn char_ceil(text: &str, mut index: usize) -> usize {
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::{highlight_with_marks, make_snippet};

    #[test]
    fn highlights_ascii_terms() {
        let highlighted = highlight_with_marks("Rust ownership model", &["ownership".to_string()]);
        assert!(highlighted.contains("<mark>ownership</mark>"));
    }

    #[test]
    fn creates_context_snippet() {
        let snippet = make_snippet(
            "Rust has ownership and borrowing rules",
            &["ownership".to_string()],
            8,
        );
        assert!(snippet.contains("ownership"));
    }
}
