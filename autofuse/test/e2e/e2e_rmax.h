#ifndef E2E_RMAX_H
#define E2E_RMAX_H

#include "ascir.h"

void LoadRmaxStore_BeforeAutofuse(ascir::HintGraph &graph);
void LoadRmaxStore_AfterAutofuse(ascir::ImplGraph &graph);

#endif

