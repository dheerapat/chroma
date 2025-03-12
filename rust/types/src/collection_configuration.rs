use crate::{
    DistributedHnswParameters, HnswParametersFromSegmentError, LocalHnswParameters, Segment,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingFunctionConfiguration {
    pub name: String,
    pub config: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexConfiguration {
    LocalHnsw(LocalHnswParameters),
    DistributedHnsw(DistributedHnswParameters),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct CollectionConfiguration {
    pub vector_index_configuration: VectorIndexConfiguration,
    pub embedding_function: Option<EmbeddingFunctionConfiguration>,
}

impl CollectionConfiguration {
    pub fn default_local() -> Self {
        Self {
            vector_index_configuration: VectorIndexConfiguration::LocalHnsw(
                LocalHnswParameters::default(),
            ),
            embedding_function: None,
        }
    }

    pub fn default_distributed() -> Self {
        Self {
            vector_index_configuration: VectorIndexConfiguration::DistributedHnsw(
                DistributedHnswParameters::default(),
            ),
            embedding_function: None,
        }
    }

    fn get_distributed_hnsw_config(&self) -> Option<DistributedHnswParameters> {
        match &self.vector_index_configuration {
            VectorIndexConfiguration::DistributedHnsw(config) => Some(config.clone()),
            _ => None,
        }
    }

    pub fn get_distributed_hnsw_config_with_legacy_fallback(
        &self,
        segment: &Segment,
    ) -> Result<DistributedHnswParameters, HnswParametersFromSegmentError> {
        match self.get_distributed_hnsw_config() {
            Some(config) => Ok(config),
            None => DistributedHnswParameters::from_legacy_segment_metadata(&segment.metadata),
        }
    }

    /// todo, rename
    fn get_local_hnsw_config(&self) -> Option<LocalHnswParameters> {
        match &self.vector_index_configuration {
            VectorIndexConfiguration::LocalHnsw(config) => Some(config.clone()),
            _ => None,
        }
    }

    pub fn get_local_hnsw_config_with_legacy_fallback(
        &self,
        segment: &Segment,
    ) -> Result<LocalHnswParameters, HnswParametersFromSegmentError> {
        match self.get_local_hnsw_config() {
            Some(config) => Ok(config),
            None => LocalHnswParameters::from_legacy_segment_metadata(&segment.metadata),
        }
    }
}
