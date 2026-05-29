use std::path::Path;

use crate::model::{Document, Section};

#[derive(Debug, Default, Clone, Copy)]
pub struct MarkdownParser;

impl MarkdownParser {
    pub fn parse(&self, id: usize, path: &Path, content: String, modified: u64) -> Document {
        let file_name = file_name(path);
        let file_stem = file_stem(path, &file_name);

        let mut title = String::new();
        let mut headings = Vec::new();
        let mut tags = Vec::new();
        let mut sections = Vec::new();
        let mut first_text_line = String::new();
        let mut current_heading = String::from("Document");
        let mut current_level = 0u8;
        let mut current_body = String::new();
        let mut in_front_matter = false;
        let mut in_front_matter_tags = false;
        let mut in_code_block = false;

        for (line_index, line) in content.lines().enumerate() {
            let trimmed = line.trim();

            if line_index == 0 && trimmed == "---" {
                in_front_matter = true;
                continue;
            }

            if in_front_matter {
                if trimmed == "---" {
                    in_front_matter = false;
                    in_front_matter_tags = false;
                } else {
                    collect_front_matter_line(
                        trimmed,
                        &mut title,
                        &mut tags,
                        &mut in_front_matter_tags,
                    );
                }
                continue;
            }

            if is_code_fence(trimmed) {
                in_code_block = !in_code_block;
                push_body_line(&mut current_body, line);
                continue;
            }

            let is_metadata_line = if in_code_block {
                false
            } else {
                collect_metadata_line(trimmed, &mut title, &mut tags)
            };

            let heading = if in_code_block {
                None
            } else {
                parse_heading(trimmed)
            };

            if let Some((level, heading)) = heading {
                push_section(
                    &mut sections,
                    &current_heading,
                    current_level,
                    &current_body,
                );
                current_body.clear();

                if title.is_empty() && level == 1 {
                    title = heading.to_string();
                }
                current_heading = heading.to_string();
                current_level = level;
                headings.push(heading.to_string());
                continue;
            }

            if !is_metadata_line {
                collect_inline_tags(trimmed, &mut tags);
                if first_text_line.is_empty() && !trimmed.is_empty() {
                    first_text_line = trimmed.to_string();
                }
            }
            push_body_line(&mut current_body, line);
        }

        if !current_body.trim().is_empty() || sections.is_empty() {
            sections.push(Section {
                heading: current_heading,
                level: current_level,
                body: current_body.trim().to_string(),
            });
        }

        if title.is_empty() {
            title = headings
                .first()
                .cloned()
                .or_else(|| (!first_text_line.is_empty()).then_some(first_text_line))
                .unwrap_or(file_stem);
        }

        tags.sort();
        tags.dedup();

        Document {
            id,
            path: path.to_string_lossy().to_string(),
            file_name,
            title,
            headings,
            tags,
            modified,
            body: content,
            sections,
        }
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn file_stem(path: &Path, fallback: &str) -> String {
    path.file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or(fallback)
        .to_string()
}

fn push_body_line(body: &mut String, line: &str) {
    body.push_str(line);
    body.push('\n');
}

fn push_section(sections: &mut Vec<Section>, heading: &str, level: u8, body: &str) {
    if body.trim().is_empty() && sections.is_empty() {
        return;
    }
    sections.push(Section {
        heading: heading.to_string(),
        level,
        body: body.trim().to_string(),
    });
}

fn is_code_fence(line: &str) -> bool {
    line.starts_with("```") || line.starts_with("~~~")
}

fn parse_heading(line: &str) -> Option<(u8, &str)> {
    let hashes = line
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    if !line
        .as_bytes()
        .get(hashes)
        .map(|byte| byte.is_ascii_whitespace())
        .unwrap_or(false)
    {
        return None;
    }

    let heading = line.get(hashes..)?.trim();
    if heading.is_empty() || heading == "#" {
        return None;
    }
    Some((hashes as u8, heading.trim_matches('#').trim()))
}

fn collect_metadata_line(line: &str, title: &mut String, tags: &mut Vec<String>) -> bool {
    if let Some(value) = value_after_key(line, "title") {
        if title.is_empty() && !value.is_empty() {
            *title = value.trim_matches('"').to_string();
        }
        return true;
    }

    if let Some(value) = value_after_key(line, "tags") {
        collect_tags_from_value(value, tags);
        return true;
    }

    false
}

fn collect_front_matter_line(
    line: &str,
    title: &mut String,
    tags: &mut Vec<String>,
    in_tags_list: &mut bool,
) {
    if let Some(value) = value_after_key(line, "title") {
        if title.is_empty() && !value.is_empty() {
            *title = value.trim_matches('"').to_string();
        }
        *in_tags_list = false;
        return;
    }

    if let Some(value) = value_after_key(line, "tags") {
        collect_tags_from_value(value, tags);
        *in_tags_list = value.is_empty();
        return;
    }

    let tag_list_item = if *in_tags_list {
        line.strip_prefix("- ")
    } else {
        None
    };
    if let Some(value) = tag_list_item {
        push_tag(value, tags);
        return;
    }

    *in_tags_list = false;
}

fn value_after_key<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (name, value) = line.split_once(':')?;
    name.trim()
        .eq_ignore_ascii_case(key)
        .then_some(value.trim())
}

fn collect_tags_from_value(value: &str, tags: &mut Vec<String>) {
    let cleaned = value
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .replace('"', "");
    for tag in cleaned.split([',', ' ']) {
        push_tag(tag, tags);
    }
}

fn collect_inline_tags(line: &str, tags: &mut Vec<String>) {
    let mut chars = line.char_indices().peekable();
    while let Some((_, character)) = chars.next() {
        if character != '#' {
            continue;
        }

        let mut tag = String::new();
        while let Some((_, next)) = chars.peek().copied() {
            if next.is_alphanumeric() || next == '-' || next == '_' {
                tag.push(next.to_ascii_lowercase());
                chars.next();
            } else {
                break;
            }
        }
        push_tag(&tag, tags);
    }
}

fn push_tag(raw: &str, tags: &mut Vec<String>) {
    let tag = raw
        .trim()
        .trim_start_matches('#')
        .trim_matches(|character: char| !character.is_alphanumeric() && character != '-')
        .to_ascii_lowercase();
    if !tag.is_empty() {
        tags.push(tag);
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::MarkdownParser;

    #[test]
    fn extracts_markdown_metadata() {
        let parser = MarkdownParser;
        let document = parser.parse(
            0,
            Path::new("notes/rust.md"),
            "---\ntitle: Rust Notes\ntags: [rust, search]\n---\n# Heading\n## Ownership\nBorrowing #memory\n"
                .to_string(),
            42,
        );

        assert_eq!(document.title, "Rust Notes");
        assert!(
            document
                .headings
                .iter()
                .any(|heading| heading == "Ownership")
        );
        assert!(document.tags.iter().any(|tag| tag == "rust"));
        assert!(document.tags.iter().any(|tag| tag == "memory"));
    }

    #[test]
    fn ignores_markdown_headings_inside_code_blocks() {
        let parser = MarkdownParser;
        let document = parser.parse(
            0,
            Path::new("notes/code.md"),
            "# Real Title\n```text\n# Not A Heading\n```\n## Real Section\n".to_string(),
            0,
        );

        assert!(
            document
                .headings
                .iter()
                .any(|heading| heading == "Real Title")
        );
        assert!(
            !document
                .headings
                .iter()
                .any(|heading| heading == "Not A Heading")
        );
    }

    #[test]
    fn uses_first_plain_text_line_as_txt_title() {
        let parser = MarkdownParser;
        let document = parser.parse(
            0,
            Path::new("notes/search.txt"),
            "Local search ranking notes\nTags: #search #ranking\nBody text\n".to_string(),
            0,
        );

        assert_eq!(document.title, "Local search ranking notes");
        assert!(document.tags.iter().any(|tag| tag == "search"));
    }

    #[test]
    fn reads_yaml_style_tag_lists() {
        let parser = MarkdownParser;
        let document = parser.parse(
            0,
            Path::new("notes/yaml.md"),
            "---\ntags:\n  - rust\n  - search\naliases:\n  - should-not-be-tag\n---\n# Title\n"
                .to_string(),
            0,
        );

        assert!(document.tags.iter().any(|tag| tag == "rust"));
        assert!(document.tags.iter().any(|tag| tag == "search"));
        assert!(!document.tags.iter().any(|tag| tag == "should-not-be-tag"));
    }
}
