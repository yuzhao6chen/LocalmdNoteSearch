pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleTokenizer;

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for character in text.chars() {
            if character.is_ascii_alphanumeric() {
                current.push(character.to_ascii_lowercase());
            } else if character.is_alphanumeric() && !character.is_ascii() {
                flush(&mut current, &mut tokens);
                tokens.push(character.to_string());
            } else {
                flush(&mut current, &mut tokens);
            }
        }

        flush(&mut current, &mut tokens);
        tokens
    }
}

fn flush(current: &mut String, tokens: &mut Vec<String>) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

#[cfg(test)]
mod tests {
    use super::{SimpleTokenizer, Tokenizer};

    #[test]
    fn tokenizes_ascii_words_and_unicode_chars() {
        let tokenizer = SimpleTokenizer;
        let tokens = tokenizer.tokenize("Rust ownership: 知识库!");
        assert_eq!(tokens, vec!["rust", "ownership", "知", "识", "库"]);
    }
}
