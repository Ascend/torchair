#ifndef AUTOFUSE_E2E_LOAD_SUB_STORE_H
#define AUTOFUSE_E2E_LOAD_SUB_STORE_H
#include "ascir.h"

void LoadSubStore_BeforeAutofuse(ascir::HintGraph &graph);
void LoadSubStore_AfterInferOutput(ascir::HintGraph &graph);
void LoadSubStore_AfterGetApiInfo(ascir::ImplGraph &graph);
void LoadSubStore_AfterScheduler(ascir::ImplGraph &graph);
void LoadSubStore_AfterQueBufAlloc(ascir::ImplGraph &graph);
#endif
