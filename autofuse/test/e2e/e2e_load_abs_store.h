#ifndef __TEST_E2E_LOAD_ABS_STORE_H__
#define __TEST_E2E_LOAD_ABS_STORE_H__

#include "ascir.h"

void LoadAbsStore_BeforeAutofuse(ascir::HintGraph &graph);
void LoadAbsStore_AfterGetApiInfo(ascir::ImplGraph &graph);
void LoadAbsStore_AfterScheduler(ascir::ImplGraph &graph);
void LoadAbsStore_AfterQueBufAlloc(ascir::ImplGraph &graph);

#endif
