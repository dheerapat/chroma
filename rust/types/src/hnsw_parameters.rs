use crate::Metadata;
use chroma_error::{ChromaError, ErrorCodes};
use serde::{Deserialize, Serialize};
use std::num::NonZero;
use thiserror::Error;
use utoipa::ToSchema;
use validator::Validate;

#[derive(Debug, Error)]
pub enum HnswParametersFromSegmentError {
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(#[from] serde_json::Error),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(#[from] validator::ValidationErrors),
}

impl ChromaError for HnswParametersFromSegmentError {
    fn code(&self) -> ErrorCodes {
        match self {
            HnswParametersFromSegmentError::InvalidMetadata(_) => ErrorCodes::InvalidArgument,
            HnswParametersFromSegmentError::InvalidParameters(_) => ErrorCodes::InvalidArgument,
        }
    }
}

#[derive(Default, Debug, PartialEq, Serialize, Deserialize, Clone, ToSchema)]
pub enum HnswSpace {
    #[default]
    #[serde(rename = "l2")]
    L2,
    #[serde(rename = "cosine")]
    Cosine,
    #[serde(rename = "ip")]
    Ip,
}

fn default_construction_ef() -> usize {
    100
}

fn default_search_ef() -> usize {
    100
}

fn default_search_ef_distributed() -> usize {
    10
}

fn default_m() -> usize {
    16
}

fn default_num_threads() -> usize {
    std::thread::available_parallelism()
        .unwrap_or(NonZero::new(1).unwrap())
        .get()
}

fn default_resize_factor() -> f64 {
    1.2
}

fn default_sync_threshold() -> usize {
    1000
}

fn default_sync_threshold_distributed() -> usize {
    64
}

#[derive(Clone, PartialEq, ToSchema, Debug, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct DistributedHnswParameters {
    #[serde(rename = "hnsw:space", default)]
    pub space: HnswSpace,
    #[serde(rename = "hnsw:construction_ef", default = "default_construction_ef")]
    pub construction_ef: usize,
    #[serde(rename = "hnsw:search_ef", default = "default_search_ef_distributed")]
    pub search_ef: usize,
    #[serde(rename = "hnsw:M", default = "default_m")]
    pub m: usize,
    #[serde(rename = "hnsw:num_threads", default = "default_num_threads")]
    pub num_threads: usize,
    #[serde(rename = "hnsw:resize_factor", default = "default_resize_factor")]
    pub resize_factor: f64,
    #[validate(range(min = 2))]
    #[serde(
        rename = "hnsw:sync_threshold",
        default = "default_sync_threshold_distributed"
    )]
    pub sync_threshold: usize,
}

impl Default for DistributedHnswParameters {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

impl DistributedHnswParameters {
    pub fn from_legacy_segment_metadata(
        segment_metadata: &Option<Metadata>,
    ) -> Result<Self, HnswParametersFromSegmentError> {
        if let Some(metadata) = segment_metadata {
            let filtered_metadata = metadata
                .clone()
                .into_iter()
                .filter(|(k, _)| k.starts_with("hnsw:"))
                .collect::<Metadata>();

            let metadata_str = serde_json::to_string(&filtered_metadata)?;
            let parsed = serde_json::from_str::<DistributedHnswParameters>(&metadata_str)?;
            parsed.validate()?;
            Ok(parsed)
        } else {
            Ok(DistributedHnswParameters::default())
        }
    }
}

fn default_batch_size() -> usize {
    100
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Validate, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SingleNodeHnswParameters {
    #[serde(rename = "hnsw:space", default)]
    pub space: HnswSpace,
    #[serde(rename = "hnsw:construction_ef", default = "default_construction_ef")]
    pub construction_ef: usize,
    #[serde(rename = "hnsw:search_ef", default = "default_search_ef")]
    pub search_ef: usize,
    #[serde(rename = "hnsw:M", default = "default_m")]
    pub m: usize,
    #[serde(rename = "hnsw:num_threads", default = "default_num_threads")]
    pub num_threads: usize,
    #[serde(rename = "hnsw:resize_factor", default = "default_resize_factor")]
    pub resize_factor: f64,
    #[validate(range(min = 2))]
    #[serde(rename = "hnsw:sync_threshold", default = "default_sync_threshold")]
    pub sync_threshold: usize,
    #[validate(range(min = 2))]
    #[serde(rename = "hnsw:batch_size", default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for SingleNodeHnswParameters {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

impl SingleNodeHnswParameters {
    pub fn from_legacy_segment_metadata(
        segment_metadata: &Option<Metadata>,
    ) -> Result<Self, HnswParametersFromSegmentError> {
        if let Some(metadata) = segment_metadata {
            let filtered_metadata = metadata
                .clone()
                .into_iter()
                .filter(|(k, _)| k.starts_with("hnsw:"))
                .collect::<Metadata>();

            let metadata_str = serde_json::to_string(&filtered_metadata)?;
            let parsed = serde_json::from_str::<SingleNodeHnswParameters>(&metadata_str)?;
            parsed.validate()?;
            Ok(parsed)
        } else {
            Ok(SingleNodeHnswParameters::default())
        }
    }
}
