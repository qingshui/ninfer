#pragma once

#include "models/spark2_5/frontend/frontend.h"
#include "models/spark2_5/options.h"
#include "models/spark2_5/profile.h"
#include "models/spark2_5/program/program.h"
#include "models/spark2_5/session/session.h"

namespace ninfer::models::spark2_5 {

// Names every type the Engine's generic core and executor consume.
struct RuntimeTypes {
    using ModelDefinition   = spark2_5::ModelDefinition;
    using ModelShard        = spark2_5::ModelShard;
    using Parameters        = spark2_5::RankParameters;
    using Profile           = spark2_5::ModelProfile;
    using Frontend          = spark2_5::Frontend;
    using PreparedPrompt    = spark2_5::PreparedPrompt;
    using OutputSession     = models::frontend::OutputSession;
    using PublishedOutput   = models::frontend::PublishedOutput;
    using SequencePlan      = ExecutionPlan;
    using SessionPlan       = spark2_5::SessionPlan;
    using ExecutionTypes    = spark2_5::ExecutionTypes;
    using RankProgram       = spark2_5::RankProgram;
    using Session           = spark2_5::Session;
    using PreparedSession   = spark2_5::PreparedSession;
    using ContextDemand     = spark2_5::ContextDemand;
    using ContextAssessment = spark2_5::ContextAssessment;
    using ContextReleaseSet = spark2_5::ContextReleaseSet;
    using RequestBasePlan   = spark2_5::RequestBasePlan;
    using SequenceHandle    = spark2_5::SequenceHandle;
    using CheckpointHandle  = spark2_5::CheckpointHandle;
    using CheckpointSummary = spark2_5::CheckpointSummary;
    using SourceCandidate   = spark2_5::SourceCandidate;
    using ResumeState       = spark2_5::ResumeState;
    using ExecutionUnit     = spark2_5::ExecutionUnit;
    using ExecutionUnitKind = spark2_5::ExecutionUnitKind;
    using ContextProgress   = spark2_5::ContextProgress;
    using PendingBatch      = spark2_5::PendingBatch;
    using PrefillProgress   = spark2_5::PrefillProgress;
    using MixedProgress     = spark2_5::MixedProgress;
    using ReplayProgress    = spark2_5::ReplayProgress;
    using ControlComplete   = spark2_5::ControlComplete;
    using ScoreComplete     = spark2_5::ScoreComplete;
    using SessionEvent      = spark2_5::SessionEvent;
    using CommitResult      = spark2_5::CommitResult;
    using DiscardResult     = spark2_5::DiscardResult;
    using FinishResult      = spark2_5::FinishResult;
    using AbortResult       = spark2_5::AbortResult;
    using CacheSessionKey   = models::frontend::PreparedSessionKey;
};

} // namespace ninfer::models::spark2_5
