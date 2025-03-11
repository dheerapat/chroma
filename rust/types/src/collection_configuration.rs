use crate::{
    DistributedHnswParameters, HnswParametersFromSegmentError, Segment, SingleNodeHnswParameters,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingFunctionConfiguration {
    pub name: String,
    pub config: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
pub enum VectorIndexConfiguration {
    SingleNodeHnsw(SingleNodeHnswParameters),
    DistributedHnsw(DistributedHnswParameters),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct CollectionConfiguration {
    pub vector_index_configuration: VectorIndexConfiguration,
    pub embedding_function: Option<EmbeddingFunctionConfiguration>,
}

impl CollectionConfiguration {
    pub fn default_single_node() -> Self {
        Self {
            vector_index_configuration: VectorIndexConfiguration::SingleNodeHnsw(
                SingleNodeHnswParameters::default(),
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
    fn get_local_hnsw_config(&self) -> Option<SingleNodeHnswParameters> {
        match &self.vector_index_configuration {
            VectorIndexConfiguration::SingleNodeHnsw(config) => Some(config.clone()),
            _ => None,
        }
    }

    pub fn get_local_hnsw_config_with_legacy_fallback(
        &self,
        segment: &Segment,
    ) -> Result<SingleNodeHnswParameters, HnswParametersFromSegmentError> {
        match self.get_local_hnsw_config() {
            Some(config) => Ok(config),
            None => SingleNodeHnswParameters::from_legacy_segment_metadata(&segment.metadata),
        }
    }
}
