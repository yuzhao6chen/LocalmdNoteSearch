pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleTokenizer;

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut ascii = String::new();
        let mut non_ascii = String::new();

        for character in text.chars() {
            if character.is_ascii_alphanumeric() {
                flush_non_ascii(&mut non_ascii, &mut tokens);
                ascii.push(character.to_ascii_lowercase());
            } else if character.is_alphanumeric() && !character.is_ascii() {
                flush_ascii(&mut ascii, &mut tokens);
                non_ascii.push(character);
            } else {
                flush_ascii(&mut ascii, &mut tokens);
                flush_non_ascii(&mut non_ascii, &mut tokens);
            }
        }

        flush_ascii(&mut ascii, &mut tokens);
        flush_non_ascii(&mut non_ascii, &mut tokens);
        tokens
    }
}

fn flush_ascii(current: &mut String, tokens: &mut Vec<String>) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

fn flush_non_ascii(current: &mut String, tokens: &mut Vec<String>) {
    if current.is_empty() {
        return;
    }

    let chars = current.chars().collect::<Vec<_>>();
    if chars.len() == 1 {
        tokens.push(current.clone());
        current.clear();
        return;
    }

    let max_len = chars.len().min(4);
    for size in 2..=max_len {
        for start in 0..=chars.len() - size {
            tokens.push(chars[start..start + size].iter().collect());
        }
    }

    current.clear();
}

#[cfg(test)]
mod tests {
    use super::{SimpleTokenizer, Tokenizer};

    #[test]
    fn tokenizes_ascii_words_and_unicode_chars() {
        let tokenizer = SimpleTokenizer;
        let tokens = tokenizer.tokenize("Rust ownership: 知识库!");
        assert_eq!(tokens, vec!["rust", "ownership", "知识", "识库", "知识库"]);
    }

    #[test]
    fn tokenizes_chinese_phrases_with_short_ngrams() {
        let tokenizer = SimpleTokenizer;
        let tokens = tokenizer.tokenize("所有权 借用");
        assert_eq!(tokens, vec!["所有", "有权", "所有权", "借用"]);
    }
}
