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

// 高亮时优先匹配长词，避免中文短语被拆碎。
fn highlight(text: &str, terms: &[String], open: &str, close: &str) -> String {
    let terms = normalized_terms(terms);
    if terms.is_empty() {
        return text.to_string();
    }

    let lower_text = text.to_lowercase();
    let mut output = String::new();
    let mut position = 0usize;

    while position < text.len() {
        if let Some(term) = find_term_at(text, &lower_text, position, &terms) {
            let end = position + term.len();
            output.push_str(open);
            output.push_str(&text[position..end]);
            output.push_str(close);
            position = end;
            continue;
        }

        if let Some(character) = text[position..].chars().next() {
            output.push(character);
            position += character.len_utf8();
        } else {
            break;
        }
    }

    output
}

fn normalized_terms(terms: &[String]) -> Vec<String> {
    let mut normalized = terms
        .iter()
        .map(|term| term.trim().to_lowercase())
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>();
    normalized.sort_by_key(|term| std::cmp::Reverse(term.len()));
    normalized.dedup();
    normalized
}

fn find_term_at<'a>(
    text: &str,
    lower_text: &str,
    position: usize,
    terms: &'a [String],
) -> Option<&'a str> {
    if !text.is_char_boundary(position) {
        return None;
    }

    for term in terms {
        let end = position + term.len();
        if end <= text.len()
            && text.is_char_boundary(end)
            && lower_text[position..].starts_with(term)
            && has_word_boundaries(text, position, end, term)
        {
            return Some(term);
        }
    }

    None
}

fn has_word_boundaries(text: &str, start: usize, end: usize, term: &str) -> bool {
    if !term
        .chars()
        .all(|character| character.is_ascii_alphanumeric())
    {
        return true;
    }

    !is_ascii_word_char_before(text, start) && !is_ascii_word_char_after(text, end)
}

fn is_ascii_word_char_before(text: &str, position: usize) -> bool {
    if position == 0 {
        return false;
    }
    text[..position]
        .chars()
        .next_back()
        .map(|character| character.is_ascii_alphanumeric())
        .unwrap_or(false)
}

fn is_ascii_word_char_after(text: &str, position: usize) -> bool {
    if position >= text.len() {
        return false;
    }
    text[position..]
        .chars()
        .next()
        .map(|character| character.is_ascii_alphanumeric())
        .unwrap_or(false)
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
    fn highlights_chinese_phrase_terms() {
        let highlighted = highlight_with_marks("Rust 所有权与借用复习", &["所有权".to_string()]);
        assert!(highlighted.contains("<mark>所有权</mark>"));
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
