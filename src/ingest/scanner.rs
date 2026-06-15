use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::error::{AppError, AppResult};
use crate::model::Document;

use super::MarkdownParser;

// 扫描用户指定目录，返回可搜索的文档列表。
pub fn scan_documents(root: &Path) -> AppResult<Vec<Document>> {
    if !root.exists() {
        return Err(AppError::Cli(format!(
            "directory does not exist: {}",
            root.display()
        )));
    }
    if !root.is_dir() {
        return Err(AppError::Cli(format!(
            "path is not a directory: {}",
            root.display()
        )));
    }

    let mut files = Vec::new();
    collect_supported_files(root, &mut files)?;
    files.sort();

    let parser = MarkdownParser;
    let mut documents = Vec::with_capacity(files.len());

    for file in files {
        let content = fs::read_to_string(&file)
            .map_err(|error| AppError::io(format!("failed to read {}", file.display()), error))?;
        let modified = modified_seconds(&file)?;
        let id = documents.len();
        documents.push(parser.parse(id, &file, content, modified));
    }

    Ok(documents)
}

// 递归收集 Markdown/TXT 文件，跳过常见工程目录。
fn collect_supported_files(dir: &Path, files: &mut Vec<PathBuf>) -> AppResult<()> {
    let entries = fs::read_dir(dir)
        .map_err(|error| AppError::io(format!("failed to read {}", dir.display()), error))?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            AppError::io(format!("failed to read entry in {}", dir.display()), error)
        })?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|error| {
            AppError::io(format!("failed to inspect {}", path.display()), error)
        })?;
        if file_type.is_dir() {
            if !is_ignored_dir(&path) {
                collect_supported_files(&path, files)?;
            }
        } else if file_type.is_file() && is_supported_file(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn is_supported_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "md" | "markdown" | "txt"
            )
        })
        .unwrap_or(false)
}

fn is_ignored_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| matches!(name, ".git" | "target" | ".idea" | ".vscode"))
        .unwrap_or(false)
}

fn modified_seconds(path: &Path) -> AppResult<u64> {
    let metadata = fs::metadata(path).map_err(|error| {
        AppError::io(format!("failed to read metadata {}", path.display()), error)
    })?;
    let modified = metadata.modified().map_err(|error| {
        AppError::io(
            format!("failed to read modified time {}", path.display()),
            error,
        )
    })?;
    modified
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| AppError::Parse(format!("modified time is before epoch: {}", path.display())))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::scan_documents;

    #[test]
    fn rejects_non_directory_root() -> Result<(), Box<dyn std::error::Error>> {
        let dir = unique_test_dir("scanner-file-root");
        std::fs::create_dir_all(&dir)?;
        let file = dir.join("note.md");
        std::fs::write(&file, "# Note\n")?;

        let error = scan_documents(&file).err();

        assert!(
            error
                .map(|error| error.to_string().contains("not a directory"))
                .unwrap_or(false)
        );
        Ok(())
    }

    fn unique_test_dir(name: &str) -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("target")
            .join(format!("{name}-{suffix}"))
    }
}
