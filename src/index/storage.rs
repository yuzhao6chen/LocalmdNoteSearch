use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{AppError, AppResult};
use crate::model::{Document, Section};

const CACHE_VERSION: u64 = 1;

// 保存解析后的文档缓存，避免每次搜索都重新解析文件。
pub fn save_cache(cache_path: &Path, root: &Path, documents: &[Document]) -> AppResult<()> {
    if let Some(parent) = cache_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            AppError::io(
                format!("failed to create cache directory {}", parent.display()),
                error,
            )
        })?;
    }

    let file = File::create(cache_path).map_err(|error| {
        AppError::io(
            format!("failed to create cache {}", cache_path.display()),
            error,
        )
    })?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "{{\"type\":\"meta\",\"version\":{},\"root\":\"{}\",\"documents\":{}}}",
        CACHE_VERSION,
        escape_json(&root.to_string_lossy()),
        documents.len()
    )
    .map_err(|error| AppError::io("failed to write cache metadata", error))?;

    for document in documents {
        writeln!(writer, "{}", document_to_json(document))
            .map_err(|error| AppError::io("failed to write document cache", error))?;
    }

    writer
        .flush()
        .map_err(|error| AppError::io("failed to flush cache", error))
}

// 读取缓存时顺便校验版本和记录类型。
pub fn load_cache(cache_path: &Path) -> AppResult<Vec<Document>> {
    let file = File::open(cache_path).map_err(|error| {
        AppError::io(
            format!("failed to open cache {}", cache_path.display()),
            error,
        )
    })?;
    let reader = BufReader::new(file);
    let mut documents = Vec::new();
    let mut saw_meta = false;

    for (line_number, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| {
            AppError::io(
                format!("failed to read cache line {}", line_number + 1),
                error,
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let value = JsonParser::new(&line).parse_complete()?;
        let JsonValue::Object(object) = value else {
            return Err(AppError::Cache(format!(
                "cache line {} is not a JSON object",
                line_number + 1
            )));
        };
        let object_type = string_field(&object, "type")?;
        if object_type == "meta" {
            validate_cache_version(&object)?;
            saw_meta = true;
            continue;
        }

        if object_type != "doc" {
            return Err(AppError::Cache(format!(
                "unsupported cache record type `{object_type}`"
            )));
        }

        let mut document = document_from_object(object)?;
        document.id = documents.len();
        documents.push(document);
    }

    if saw_meta {
        Ok(documents)
    } else {
        Err(AppError::Cache(
            "cache metadata is missing; rebuild the index".to_string(),
        ))
    }
}

pub fn escape_json(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0c}' => escaped.push_str("\\f"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn document_to_json(document: &Document) -> String {
    let mut output = String::from("{\"type\":\"doc\"");
    push_string_field(&mut output, "path", &document.path);
    push_string_field(&mut output, "file_name", &document.file_name);
    push_string_field(&mut output, "title", &document.title);
    output.push_str(",\"modified\":");
    output.push_str(&document.modified.to_string());
    push_string_array(&mut output, "headings", &document.headings);
    push_string_array(&mut output, "tags", &document.tags);
    push_string_field(&mut output, "body", &document.body);
    output.push_str(",\"sections\":[");
    for (index, section) in document.sections.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push('{');
        push_first_string_field(&mut output, "heading", &section.heading);
        output.push_str(",\"level\":");
        output.push_str(&section.level.to_string());
        push_string_field(&mut output, "body", &section.body);
        output.push('}');
    }
    output.push_str("]}");
    output
}

fn push_first_string_field(output: &mut String, key: &str, value: &str) {
    output.push('"');
    output.push_str(key);
    output.push_str("\":\"");
    output.push_str(&escape_json(value));
    output.push('"');
}

fn push_string_field(output: &mut String, key: &str, value: &str) {
    output.push(',');
    push_first_string_field(output, key, value);
}

fn push_string_array(output: &mut String, key: &str, values: &[String]) {
    output.push_str(",\"");
    output.push_str(key);
    output.push_str("\":[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push('"');
        output.push_str(&escape_json(value));
        output.push('"');
    }
    output.push(']');
}

fn document_from_object(object: BTreeMap<String, JsonValue>) -> AppResult<Document> {
    let sections = array_field(&object, "sections")?
        .iter()
        .map(section_from_value)
        .collect::<AppResult<Vec<_>>>()?;

    Ok(Document {
        id: 0,
        path: string_field(&object, "path")?.to_string(),
        file_name: string_field(&object, "file_name")?.to_string(),
        title: string_field(&object, "title")?.to_string(),
        headings: string_array_field(&object, "headings")?,
        tags: string_array_field(&object, "tags")?,
        modified: number_field(&object, "modified")?,
        body: string_field(&object, "body")?.to_string(),
        sections,
    })
}

fn validate_cache_version(object: &BTreeMap<String, JsonValue>) -> AppResult<()> {
    let version = number_field(object, "version")?;
    if version == CACHE_VERSION {
        Ok(())
    } else {
        Err(AppError::Cache(format!(
            "unsupported cache version {version}; expected {CACHE_VERSION}"
        )))
    }
}

fn section_from_value(value: &JsonValue) -> AppResult<Section> {
    let JsonValue::Object(object) = value else {
        return Err(AppError::Cache(
            "section entry is not an object".to_string(),
        ));
    };
    let level = number_field(object, "level")?;
    let level = u8::try_from(level)
        .map_err(|_| AppError::Cache(format!("invalid section level {level}")))?;
    Ok(Section {
        heading: string_field(object, "heading")?.to_string(),
        level,
        body: string_field(object, "body")?.to_string(),
    })
}

fn string_field<'a>(object: &'a BTreeMap<String, JsonValue>, key: &str) -> AppResult<&'a str> {
    match object.get(key) {
        Some(JsonValue::String(value)) => Ok(value),
        Some(_) => Err(AppError::Cache(format!(
            "cache field `{key}` is not a string"
        ))),
        None => Err(AppError::Cache(format!("cache is missing field `{key}`"))),
    }
}

fn number_field(object: &BTreeMap<String, JsonValue>, key: &str) -> AppResult<u64> {
    match object.get(key) {
        Some(JsonValue::Number(value)) => Ok(*value),
        Some(_) => Err(AppError::Cache(format!(
            "cache field `{key}` is not a number"
        ))),
        None => Err(AppError::Cache(format!("cache is missing field `{key}`"))),
    }
}

fn array_field<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> AppResult<&'a [JsonValue]> {
    match object.get(key) {
        Some(JsonValue::Array(value)) => Ok(value),
        Some(_) => Err(AppError::Cache(format!(
            "cache field `{key}` is not an array"
        ))),
        None => Err(AppError::Cache(format!("cache is missing field `{key}`"))),
    }
}

fn string_array_field(object: &BTreeMap<String, JsonValue>, key: &str) -> AppResult<Vec<String>> {
    array_field(object, key)?
        .iter()
        .map(|value| match value {
            JsonValue::String(value) => Ok(value.clone()),
            _ => Err(AppError::Cache(format!(
                "cache array `{key}` contains a non-string value"
            ))),
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonValue {
    String(String),
    Number(u64),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

// 只支持本项目缓存需要的 JSON 子集。
struct JsonParser<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            position: 0,
        }
    }

    fn parse_value(&mut self) -> AppResult<JsonValue> {
        self.skip_whitespace();
        let Some(byte) = self.peek() else {
            return Err(AppError::Cache("unexpected end of JSON".to_string()));
        };
        match byte {
            b'"' => self.parse_string().map(JsonValue::String),
            b'0'..=b'9' => self.parse_number().map(JsonValue::Number),
            b'[' => self.parse_array().map(JsonValue::Array),
            b'{' => self.parse_object().map(JsonValue::Object),
            other => Err(AppError::Cache(format!(
                "unexpected JSON byte `{}` at {}",
                other as char, self.position
            ))),
        }
    }

    fn parse_complete(&mut self) -> AppResult<JsonValue> {
        let value = self.parse_value()?;
        self.skip_whitespace();
        if self.peek().is_some() {
            return Err(AppError::Cache(format!(
                "unexpected trailing JSON at {}",
                self.position
            )));
        }
        Ok(value)
    }

    fn parse_object(&mut self) -> AppResult<BTreeMap<String, JsonValue>> {
        self.expect_byte(b'{')?;
        let mut object = BTreeMap::new();
        loop {
            self.skip_whitespace();
            if self.consume_if(b'}') {
                break;
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect_byte(b':')?;
            let value = self.parse_value()?;
            object.insert(key, value);
            self.skip_whitespace();
            if self.consume_if(b'}') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(object)
    }

    fn parse_array(&mut self) -> AppResult<Vec<JsonValue>> {
        self.expect_byte(b'[')?;
        let mut values = Vec::new();
        loop {
            self.skip_whitespace();
            if self.consume_if(b']') {
                break;
            }
            values.push(self.parse_value()?);
            self.skip_whitespace();
            if self.consume_if(b']') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(values)
    }

    fn parse_number(&mut self) -> AppResult<u64> {
        let start = self.position;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.position += 1;
        }
        let raw = std::str::from_utf8(&self.input[start..self.position])
            .map_err(|_| AppError::Cache("invalid number encoding".to_string()))?;
        raw.parse::<u64>()
            .map_err(|_| AppError::Cache(format!("invalid number `{raw}`")))
    }

    fn parse_string(&mut self) -> AppResult<String> {
        self.expect_byte(b'"')?;
        let mut output = String::new();
        while let Some(byte) = self.peek() {
            match byte {
                b'"' => {
                    self.position += 1;
                    return Ok(output);
                }
                b'\\' => {
                    self.position += 1;
                    let escaped = self
                        .next()
                        .ok_or_else(|| AppError::Cache("unterminated escape".to_string()))?;
                    match escaped {
                        b'"' => output.push('"'),
                        b'\\' => output.push('\\'),
                        b'/' => output.push('/'),
                        b'b' => output.push('\u{08}'),
                        b'f' => output.push('\u{0c}'),
                        b'n' => output.push('\n'),
                        b'r' => output.push('\r'),
                        b't' => output.push('\t'),
                        b'u' => output.push(self.parse_unicode_escape()?),
                        other => {
                            return Err(AppError::Cache(format!(
                                "unsupported escape `\\{}`",
                                other as char
                            )));
                        }
                    }
                }
                _ => {
                    let rest = std::str::from_utf8(&self.input[self.position..])
                        .map_err(|_| AppError::Cache("invalid string encoding".to_string()))?;
                    let character = rest
                        .chars()
                        .next()
                        .ok_or_else(|| AppError::Cache("unterminated string".to_string()))?;
                    self.position += character.len_utf8();
                    output.push(character);
                }
            }
        }
        Err(AppError::Cache("unterminated string".to_string()))
    }

    fn parse_unicode_escape(&mut self) -> AppResult<char> {
        if self.position + 4 > self.input.len() {
            return Err(AppError::Cache("short unicode escape".to_string()));
        }
        let raw = std::str::from_utf8(&self.input[self.position..self.position + 4])
            .map_err(|_| AppError::Cache("invalid unicode escape".to_string()))?;
        self.position += 4;
        let value = u32::from_str_radix(raw, 16)
            .map_err(|_| AppError::Cache(format!("invalid unicode escape `{raw}`")))?;
        char::from_u32(value).ok_or_else(|| AppError::Cache(format!("invalid codepoint {value}")))
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }

    fn expect_byte(&mut self, expected: u8) -> AppResult<()> {
        match self.next() {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(AppError::Cache(format!(
                "expected `{}`, got `{}` at {}",
                expected as char,
                actual as char,
                self.position.saturating_sub(1)
            ))),
            None => Err(AppError::Cache(format!(
                "expected `{}`, got end of input",
                expected as char
            ))),
        }
    }

    fn consume_if(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.position += 1;
        Some(byte)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::model::{Document, Section};

    use super::{JsonParser, JsonValue, escape_json, load_cache, save_cache};

    #[test]
    fn escapes_json_control_characters() {
        assert_eq!(escape_json("a\"b\\c\n"), "a\\\"b\\\\c\\n");
    }

    #[test]
    fn parses_generated_json_string() {
        let mut parser = JsonParser::new("{\"title\":\"Rust\\nSearch\",\"n\":2}");
        let value = parser.parse_complete();
        assert!(matches!(value, Ok(JsonValue::Object(_))));
    }

    #[test]
    fn rejects_trailing_json_data() {
        let mut parser = JsonParser::new("{\"title\":\"Rust\"} trailing");
        let value = parser.parse_complete();
        assert!(value.is_err());
    }

    #[test]
    fn saves_and_loads_document_cache() -> Result<(), Box<dyn std::error::Error>> {
        let cache_path = unique_cache_path();
        let documents = vec![Document {
            id: 0,
            path: "notes/rust.md".to_string(),
            file_name: "rust.md".to_string(),
            title: "Rust Notes".to_string(),
            headings: vec!["Ownership".to_string()],
            tags: vec!["rust".to_string()],
            modified: 123,
            body: "Ownership and borrowing".to_string(),
            sections: vec![Section {
                heading: "Ownership".to_string(),
                level: 2,
                body: "Ownership and borrowing".to_string(),
            }],
        }];

        save_cache(&cache_path, std::path::Path::new("notes"), &documents)?;
        let loaded = load_cache(&cache_path)?;

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].title, "Rust Notes");
        assert_eq!(loaded[0].sections[0].heading, "Ownership");
        Ok(())
    }

    #[test]
    fn rejects_unknown_cache_version() -> Result<(), Box<dyn std::error::Error>> {
        let cache_path = unique_cache_path();
        std::fs::write(
            &cache_path,
            "{\"type\":\"meta\",\"version\":999,\"root\":\"notes\",\"documents\":0}\n",
        )?;

        let error = load_cache(&cache_path).err();

        assert!(
            error
                .map(|error| error.to_string().contains("unsupported cache version"))
                .unwrap_or(false)
        );
        Ok(())
    }

    #[test]
    fn rejects_cache_without_metadata() -> Result<(), Box<dyn std::error::Error>> {
        let cache_path = unique_cache_path();
        std::fs::write(
            &cache_path,
            "{\"type\":\"doc\",\"path\":\"a.md\",\"file_name\":\"a.md\",\"title\":\"A\",\"modified\":0,\"headings\":[],\"tags\":[],\"body\":\"\",\"sections\":[]}\n",
        )?;

        let error = load_cache(&cache_path).err();

        assert!(
            error
                .map(|error| error.to_string().contains("metadata is missing"))
                .unwrap_or(false)
        );
        Ok(())
    }

    #[test]
    fn rejects_unknown_cache_record_type() -> Result<(), Box<dyn std::error::Error>> {
        let cache_path = unique_cache_path();
        std::fs::write(
            &cache_path,
            "{\"type\":\"meta\",\"version\":1,\"root\":\"notes\",\"documents\":0}\n{\"type\":\"other\"}\n",
        )?;

        let error = load_cache(&cache_path).err();

        assert!(
            error
                .map(|error| error.to_string().contains("unsupported cache record type"))
                .unwrap_or(false)
        );
        Ok(())
    }

    fn unique_cache_path() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("target")
            .join(format!("storage-roundtrip-{suffix}.jsonl"))
    }
}
