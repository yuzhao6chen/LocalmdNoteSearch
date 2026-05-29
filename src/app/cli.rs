use std::path::PathBuf;

use crate::error::{AppError, AppResult};

use super::output::OutputFormat;

#[derive(Debug)]
pub struct Options {
    pub command: Command,
    pub cache: PathBuf,
}

#[derive(Debug)]
pub enum Command {
    Index {
        dir: PathBuf,
    },
    Rebuild {
        dir: PathBuf,
    },
    Search {
        query: Vec<String>,
        dir: Option<PathBuf>,
        limit: usize,
        format: OutputFormat,
        rebuild: bool,
    },
    Help,
}

impl Options {
    pub fn parse<I>(args: I) -> AppResult<Self>
    where
        I: IntoIterator<Item = String>,
    {
        let mut args: Vec<String> = args.into_iter().collect();
        if args.is_empty() {
            return Ok(Self {
                command: Command::Help,
                cache: default_cache_path(),
            });
        }

        let command_name = args.remove(0);
        let mut cache = default_cache_path();
        let command = match command_name.as_str() {
            "index" => {
                let dir = take_required_path(&mut args, "directory")?;
                cache = take_cache(&mut args, cache)?;
                reject_extra(&args)?;
                Command::Index { dir }
            }
            "rebuild" => {
                let dir = take_required_path(&mut args, "directory")?;
                cache = take_cache(&mut args, cache)?;
                reject_extra(&args)?;
                Command::Rebuild { dir }
            }
            "search" => {
                let mut dir = None;
                let mut limit = 10usize;
                let mut format = OutputFormat::Text;
                let mut rebuild = false;
                let mut query = Vec::new();
                let mut index = 0usize;

                while index < args.len() {
                    match args[index].as_str() {
                        "--cache" => {
                            index += 1;
                            cache = parse_path_arg(args.get(index), "--cache")?;
                        }
                        "--dir" => {
                            index += 1;
                            dir = Some(parse_path_arg(args.get(index), "--dir")?);
                        }
                        "--limit" => {
                            index += 1;
                            limit = parse_limit(args.get(index))?;
                        }
                        "--format" => {
                            index += 1;
                            format = parse_format(args.get(index))?;
                        }
                        "--json" => {
                            format = OutputFormat::Json;
                        }
                        "--text" => {
                            format = OutputFormat::Text;
                        }
                        "--rebuild" => {
                            rebuild = true;
                        }
                        "--help" | "-h" => {
                            return Ok(Self {
                                command: Command::Help,
                                cache,
                            });
                        }
                        value if value.starts_with('-') => {
                            return Err(AppError::Cli(format!("unknown option `{value}`")));
                        }
                        value => query.push(value.to_string()),
                    }
                    index += 1;
                }

                if query.is_empty() {
                    return Err(AppError::Cli(
                        "search requires one or more keywords".to_string(),
                    ));
                }

                Command::Search {
                    query,
                    dir,
                    limit,
                    format,
                    rebuild,
                }
            }
            "help" | "--help" | "-h" => Command::Help,
            other => {
                return Err(AppError::Cli(format!(
                    "unknown command `{other}`; try `md-knowsearch help`"
                )));
            }
        };

        Ok(Self { command, cache })
    }
}

fn default_cache_path() -> PathBuf {
    PathBuf::from(".md-knowsearch-cache.jsonl")
}

fn take_required_path(args: &mut Vec<String>, label: &str) -> AppResult<PathBuf> {
    if args.is_empty() || args[0].starts_with('-') {
        return Err(AppError::Cli(format!("missing {label}")));
    }
    Ok(PathBuf::from(args.remove(0)))
}

fn take_cache(args: &mut Vec<String>, default: PathBuf) -> AppResult<PathBuf> {
    let mut cache = default;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--cache" => {
                args.remove(index);
                if index >= args.len() {
                    return Err(AppError::Cli("missing value after --cache".to_string()));
                }
                cache = PathBuf::from(args.remove(index));
            }
            "--help" | "-h" => {
                return Err(AppError::Cli(
                    "help is available with `md-knowsearch help`".to_string(),
                ));
            }
            _ => index += 1,
        }
    }
    Ok(cache)
}

fn reject_extra(args: &[String]) -> AppResult<()> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(AppError::Cli(format!("unexpected argument `{}`", args[0])))
    }
}

fn parse_path_arg(value: Option<&String>, flag: &str) -> AppResult<PathBuf> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| AppError::Cli(format!("missing value after {flag}")))
}

fn parse_limit(value: Option<&String>) -> AppResult<usize> {
    let raw = value.ok_or_else(|| AppError::Cli("missing value after --limit".to_string()))?;
    let limit = raw
        .parse::<usize>()
        .map_err(|_| AppError::Cli(format!("invalid --limit value `{raw}`")))?;
    if limit == 0 {
        return Err(AppError::Cli("--limit must be greater than 0".to_string()));
    }
    Ok(limit)
}

fn parse_format(value: Option<&String>) -> AppResult<OutputFormat> {
    match value.map(String::as_str) {
        Some("text") => Ok(OutputFormat::Text),
        Some("json") => Ok(OutputFormat::Json),
        Some(other) => Err(AppError::Cli(format!(
            "unsupported format `{other}`; use text or json"
        ))),
        None => Err(AppError::Cli("missing value after --format".to_string())),
    }
}
