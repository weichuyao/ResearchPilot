"""所有数据表模型。

集中在这里导入，是为了让 `SQLModel.metadata` 在任何建表动作之前就已经收集到全部表。
SQLModel 的 create_all 只会为「已经被 import 过」的模型建表 —— 漏导入一个模块，
那张表就会静默地不存在。
"""

from db.models.base import DBBaseModel
from db.models.collection import Collection
from db.models.conversation import Conversation
from db.models.paper import Paper
from db.models.research import (
    ApprovalRequest,
    Conclusion,
    ConclusionEvidence,
    ConclusionObservation,
    Evidence,
    EvidenceRelation,
    EvidenceSearchAttempt,
    EvidenceSearchHit,
    Experiment,
    ExperimentHypothesis,
    Hypothesis,
    Observation,
    ObservationHypothesisRelation,
    ObservationRun,
    ResearchEvent,
    ResearchQuestion,
    ResearchRun,
    ResearchWorkflowState,
)

__all__ = [
    "DBBaseModel",
    "Collection",
    "Conversation",
    "Paper",
    "ApprovalRequest",
    "Conclusion",
    "ConclusionEvidence",
    "ConclusionObservation",
    "Evidence",
    "EvidenceRelation",
    "EvidenceSearchAttempt",
    "EvidenceSearchHit",
    "Experiment",
    "ExperimentHypothesis",
    "Hypothesis",
    "Observation",
    "ObservationHypothesisRelation",
    "ObservationRun",
    "ResearchEvent",
    "ResearchQuestion",
    "ResearchRun",
    "ResearchWorkflowState",
]
