from pydantic import BaseModel, ConfigDict, Field


class NetworkMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cell_id: str = Field(min_length=1, max_length=128)
    cpu: float = Field(ge=0, le=100)
    traffic: float = Field(ge=0, le=100)
    packet_loss: float = Field(ge=0, le=100)
    latency: float = Field(ge=0)
    signal_strength: float = Field(le=0)
    throughput: float = Field(ge=0)


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)
