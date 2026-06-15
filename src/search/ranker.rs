use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::index::{InvertedIndex, TermStats};
use crate::model::{Document, Section};

use super::Query;
use super::highlight::{highlight_with_ansi, highlight_with_marks, make_snippet};

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub file_name: String,
    pub title: String,
    pub tags: Vec<String>,
    pub modified: u64,
    pub section: String,
    pub score: f64,
    pub matched_terms: Vec<String>,
    pub snippet: String,
    pub highlighted_snippet: String,
    pub marked_snippet: String,
}

pub struct SearchEngine<'a> {
    index: &'a InvertedIndex,
}

impl<'a> SearchEngine<'a> {
    pub fn new(index: &'a InvertedIndex) -> Self {
        Self { index }
    }

    // 搜索流程：取候选文档、打分、排序、截断。
    pub fn search(&self, query_parts: &[String], limit: usize) -> Vec<SearchResult> {
        let query = Query::new(query_parts);
        if query.terms.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<usize, AccumulatedScore> = HashMap::new();
        for term in &query.terms {
            let Some(postings) = self.index.postings_for(term) else {
                continue;
            };
            let idf = inverse_document_frequency(self.index.document_count(), postings.len());
            for (doc_id, stats) in postings {
                let entry = scores.entry(*doc_id).or_default();
                entry.score += score_term(*stats, idf);
                entry.matched_terms.insert(term.clone());
            }
        }

        let mut results = scores
            .into_iter()
            .filter_map(|(doc_id, accumulated)| {
                self.index
                    .documents
                    .get(doc_id)
                    .map(|document| self.build_result(document, &query, accumulated))
            })
            .collect::<Vec<_>>();

        results.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.file_path.cmp(&right.file_path))
        });
        results.truncate(limit);
        results
    }

    fn build_result(
        &self,
        document: &Document,
        query: &Query,
        accumulated: AccumulatedScore,
    ) -> SearchResult {
        let matched_terms = display_matched_terms(document, query, accumulated.matched_terms);

        let mut score = accumulated.score;
        score += coverage_bonus(query.terms.len(), matched_terms.len());
        score += phrase_bonus(document, &query.raw);
        score += recency_bonus(document.modified);

        let section = best_section(document, &query.terms);
        let snippet_source = if section.body.trim().is_empty() {
            document.body.clone()
        } else {
            format!("{}\n{}", section.heading, section.body)
        };
        let snippet = make_snippet(&snippet_source, &query.terms, 80);
        let highlighted_snippet = highlight_with_ansi(&snippet, &query.terms);
        let marked_snippet = highlight_with_marks(&snippet, &query.terms);

        SearchResult {
            file_path: document.path.clone(),
            file_name: document.file_name.clone(),
            title: document.title.clone(),
            tags: document.tags.clone(),
            modified: document.modified,
            section: section.heading.clone(),
            score,
            matched_terms,
            snippet,
            highlighted_snippet,
            marked_snippet,
        }
    }
}

#[derive(Debug, Default)]
struct AccumulatedScore {
    score: f64,
    matched_terms: HashSet<String>,
}

// 单个词的基础分数，合并字段权重和 IDF。
fn score_term(stats: TermStats, idf: f64) -> f64 {
    let weighted = stats.weighted_frequency();
    if weighted <= 0.0 {
        0.0
    } else {
        (1.0 + weighted).ln() * idf * 4.0
    }
}

fn inverse_document_frequency(document_count: usize, document_frequency: usize) -> f64 {
    (((document_count as f64 + 1.0) / (document_frequency as f64 + 1.0)).ln() + 1.0).max(1.0)
}

fn coverage_bonus(total_terms: usize, matched_terms: usize) -> f64 {
    if total_terms == 0 {
        return 0.0;
    }
    let ratio = matched_terms as f64 / total_terms as f64;
    let all_terms_bonus = if matched_terms == total_terms {
        3.0
    } else {
        0.0
    };
    ratio * 4.0 + all_terms_bonus
}

// 完整短语直接出现时，额外加一点分。
fn phrase_bonus(document: &Document, raw_query: &str) -> f64 {
    let phrase = raw_query.trim().to_lowercase();
    if phrase.is_empty() || phrase.split_whitespace().count() <= 1 {
        return 0.0;
    }
    let title = document.title.to_lowercase();
    let body = document.body.to_lowercase();
    let headings = document.headings.join(" ").to_lowercase();
    let tags = document.tags.join(" ").to_lowercase();

    let mut bonus = 0.0;
    if title.contains(&phrase) {
        bonus += 6.0;
    }
    if tags.contains(&phrase) {
        bonus += 4.0;
    }
    if headings.contains(&phrase) {
        bonus += 3.0;
    }
    if body.contains(&phrase) {
        bonus += 1.5;
    }
    bonus
}

fn recency_bonus(modified: u64) -> f64 {
    if modified == 0 {
        return 0.0;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(modified);
    let age_days = now.saturating_sub(modified) as f64 / 86_400.0;
    (1.0 / (1.0 + age_days / 365.0)).min(1.0)
}

// 结果里展示最相关的章节，而不是只显示整篇文档。
fn best_section<'a>(document: &'a Document, terms: &[String]) -> &'a Section {
    document
        .sections
        .iter()
        .max_by(|left, right| {
            section_score(left, terms)
                .partial_cmp(&section_score(right, terms))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or_else(|| {
            static EMPTY: Section = Section {
                heading: String::new(),
                level: 0,
                body: String::new(),
            };
            &EMPTY
        })
}

fn section_score(section: &Section, terms: &[String]) -> f64 {
    let heading = section.heading.to_lowercase();
    let body = section.body.to_lowercase();
    terms
        .iter()
        .map(|term| {
            let heading_hits = heading.matches(term).count() as f64;
            let body_hits = body.matches(term).count() as f64;
            heading_hits * 4.0 + body_hits
        })
        .sum()
}

fn display_matched_terms(
    document: &Document,
    query: &Query,
    indexed_matches: HashSet<String>,
) -> Vec<String> {
    let haystack = searchable_text(document).to_lowercase();
    let mut terms = query
        .keywords
        .iter()
        .filter(|keyword| haystack.contains(&keyword.to_lowercase()))
        .cloned()
        .collect::<Vec<_>>();

    if terms.is_empty() {
        terms = indexed_matches.into_iter().collect();
    }

    terms.sort();
    terms.dedup();
    terms
}

fn searchable_text(document: &Document) -> String {
    format!(
        "{} {} {} {} {}",
        document.title,
        document.file_name,
        document.tags.join(" "),
        document.headings.join(" "),
        document.body
    )
}

#[cfg(test)]
mod tests {
    use crate::index::InvertedIndex;
    use crate::model::{Document, Section};

    use super::SearchEngine;

    #[test]
    fn title_match_scores_above_body_only_match() {
        let documents = vec![
            Document {
                id: 0,
                path: "a.md".to_string(),
                file_name: "a.md".to_string(),
                title: "Rust Ownership".to_string(),
                headings: vec!["Intro".to_string()],
                tags: vec![],
                modified: 0,
                body: "short note".to_string(),
                sections: vec![Section {
                    heading: "Intro".to_string(),
                    level: 1,
                    body: "short note".to_string(),
                }],
            },
            Document {
                id: 1,
                path: "b.md".to_string(),
                file_name: "b.md".to_string(),
                title: "Other".to_string(),
                headings: vec!["Intro".to_string()],
                tags: vec![],
                modified: 0,
                body: "ownership ownership ownership".to_string(),
                sections: vec![Section {
                    heading: "Intro".to_string(),
                    level: 1,
                    body: "ownership ownership ownership".to_string(),
                }],
            },
        ];
        let index = InvertedIndex::build(documents);
        let engine = SearchEngine::new(&index);
        let results = engine.search(&["ownership".to_string()], 10);
        assert_eq!(
            results.first().map(|result| result.file_path.as_str()),
            Some("a.md")
        );
    }

    #[test]
    fn document_matching_more_query_terms_ranks_higher() {
        let documents = vec![
            Document {
                id: 0,
                path: "one.md".to_string(),
                file_name: "one.md".to_string(),
                title: "Rust".to_string(),
                headings: vec![],
                tags: vec![],
                modified: 0,
                body: "Rust notes".to_string(),
                sections: vec![Section {
                    heading: "Document".to_string(),
                    level: 0,
                    body: "Rust notes".to_string(),
                }],
            },
            Document {
                id: 1,
                path: "both.md".to_string(),
                file_name: "both.md".to_string(),
                title: "Rust Ownership".to_string(),
                headings: vec!["Borrowing".to_string()],
                tags: vec!["memory".to_string()],
                modified: 0,
                body: "Ownership and borrowing notes".to_string(),
                sections: vec![Section {
                    heading: "Borrowing".to_string(),
                    level: 2,
                    body: "Ownership and borrowing notes".to_string(),
                }],
            },
        ];
        let index = InvertedIndex::build(documents);
        let engine = SearchEngine::new(&index);
        let results = engine.search(&["rust".to_string(), "ownership".to_string()], 10);

        assert_eq!(
            results.first().map(|result| result.file_path.as_str()),
            Some("both.md")
        );
    }
}
