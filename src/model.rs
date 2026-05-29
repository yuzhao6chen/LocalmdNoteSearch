#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub id: usize,
    pub path: String,
    pub file_name: String,
    pub title: String,
    pub headings: Vec<String>,
    pub tags: Vec<String>,
    pub modified: u64,
    pub body: String,
    pub sections: Vec<Section>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub heading: String,
    pub level: u8,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Field {
    Title,
    Tag,
    Heading,
    Body,
    FileName,
}

impl Field {
    pub fn weight(self) -> f64 {
        match self {
            Self::Title => 6.0,
            Self::Tag => 5.0,
            Self::Heading => 4.0,
            Self::FileName => 3.0,
            Self::Body => 1.2,
        }
    }
}
