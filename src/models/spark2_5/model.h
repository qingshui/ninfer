#pragma once

// The model's weight owner and definition: ModelShard holds the immutable placed weights that
// RankParameters prepares and RankProgram borrows.
#include "models/spark2_5/definition.h"
#include "models/spark2_5/execution/resident.h"
