from pydantic import BaseModel, Field
from typing import List


class IncidentInput(BaseModel):
    description: str = Field(..., min_length=10, max_length=2000,
                             description="Description of the incident or alert")


class IncidentAnalysis(BaseModel):
    root_cause: str
    severity: str
    affected_services: List[str]
    recommended_fix: str
    estimated_resolution_time: str
