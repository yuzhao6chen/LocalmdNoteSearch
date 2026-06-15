use std::collections::HashMap;

use crate::ingest::{SimpleTokenizer, Tokenizer};
use crate::model::{Document, Field};

#[derive(Debug, Clone, Copy, Default)]
pub struct TermStats {
    pub title: u32,
    pub tag: u32,
    pub heading: u32,
    pub body: u32,
    pub file_name: u32,
}

impl TermStats {
    pub fn add(&mut self, field: Field) {
        match field {
            Field::Title => self.title += 1,
            Field::Tag => self.tag += 1,
            Field::Heading => self.heading += 1,
            Field::Body => self.body += 1,
            Field::FileName => self.file_name += 1,
        }
    }

    pub fn weighted_frequency(self) -> f64 {
        self.title as f64 * Field::Title.weight()
            + self.tag as f64 * Field::Tag.weight()
            + self.heading as f64 * Field::Heading.weight()
            + self.file_name as f64 * Field::FileName.weight()
            + self.body as f64 * Field::Body.weight()
    }

    pub fn total(self) -> u32 {
        self.title + self.tag + self.heading + self.body + self.file_name
    }
}

#[derive(Debug, Clone)]
pub struct InvertedIndex {
    pub documents: Vec<Document>,
    postings: HashMap<String, HashMap<usize, TermStats>>,
}

impl InvertedIndex {
    // 构建倒排索引：词 -> 文档 -> 各字段命中次数。
    pub fn build(documents: Vec<Document>) -> Self {
        let tokenizer = SimpleTokenizer;
        let mut postings = HashMap::new();

        for (doc_id, document) in documents.iter().enumerate() {
            add_text(
                &mut postings,
                doc_id,
                Field::Title,
                &document.title,
                &tokenizer,
            );
            add_text(
                &mut postings,
                doc_id,
                Field::FileName,
                &document.file_name,
                &tokenizer,
            );
            add_text(
                &mut postings,
                doc_id,
                Field::Body,
                &document.body,
                &tokenizer,
            );

            for tag in &document.tags {
                add_text(&mut postings, doc_id, Field::Tag, tag, &tokenizer);
            }
            for heading in &document.headings {
                add_text(&mut postings, doc_id, Field::Heading, heading, &tokenizer);
            }
        }

        Self {
            documents,
            postings,
        }
    }

    pub fn postings_for(&self, term: &str) -> Option<&HashMap<usize, TermStats>> {
        self.postings.get(term)
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn document_frequency(&self, term: &str) -> usize {
        self.postings
            .get(term)
            .map(std::collections::HashMap::len)
            .unwrap_or(0)
    }
}

// 将一个字段的文本加入 postings，并记录字段类型。
fn add_text<T: Tokenizer>(
    postings: &mut HashMap<String, HashMap<usize, TermStats>>,
    doc_id: usize,
    field: Field,
    text: &str,
    tokenizer: &T,
) {
    for token in tokenizer.tokenize(text) {
        postings
            .entry(token)
            .or_default()
            .entry(doc_id)
            .or_default()
            .add(field);
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{Document, Section};

    use super::InvertedIndex;

    #[test]
    fn indexes_title_and_body_terms() {
        let document = Document {
            id: 0,
            path: "rust.md".to_string(),
            file_name: "rust.md".to_string(),
            title: "Rust Search".to_string(),
            headings: vec!["Ownership".to_string()],
            tags: vec!["systems".to_string()],
            modified: 0,
            body: "Borrowing and ownership".to_string(),
            sections: vec![Section {
                heading: "Ownership".to_string(),
                level: 2,
                body: "Borrowing and ownership".to_string(),
            }],
        };
        let index = InvertedIndex::build(vec![document]);
        assert_eq!(index.document_frequency("rust"), 1);
        assert_eq!(index.document_frequency("ownership"), 1);
    }
}
