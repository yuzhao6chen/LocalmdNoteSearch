use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::io;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug)]
pub enum AppError {
    Io { context: String, source: io::Error },
    Cli(String),
    Cache(String),
    Parse(String),
}

impl AppError {
    pub fn io(context: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}

impl Display for AppError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { context, source } => write!(formatter, "{context}: {source}"),
            Self::Cli(message) | Self::Cache(message) | Self::Parse(message) => {
                write!(formatter, "{message}")
            }
        }
    }
}

impl Error for AppError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Cli(_) | Self::Cache(_) | Self::Parse(_) => None,
        }
    }
}
