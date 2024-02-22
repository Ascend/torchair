#ifndef E2E_RMAX_H
#define E2E_RMAX_H

#include "ascir.h"

void LoadRmaxStore_BeforeAutofuse(ascir::HintGraph &graph, bool is_f16 = true);
void LoadRmaxStore_AfterAutofuse(ascir::ImplGraph &graph, bool is_f16 = true);

#endif

