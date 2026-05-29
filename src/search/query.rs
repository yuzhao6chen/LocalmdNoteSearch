use crate::ingest::{SimpleTokenizer, Tokenizer};

#[derive(Debug, Clone)]
pub struct Query {
    pub raw: String,
    pub terms: Vec<String>,
}

impl Query {
    pub fn new(parts: &[String]) -> Self {
        let raw = parts.join(" ");
        let tokenizer = SimpleTokenizer;
        let mut terms = tokenizer.tokenize(&raw);
        terms.sort();
        terms.dedup();
        Self { raw, terms }
    }
}
