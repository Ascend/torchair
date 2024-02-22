#ifndef __E2E_BROADCAST_H__
#define __E2E_BROADCAST_H__

#include "ascir.h"

void LoadBroadcastStore_BeforeAutofuse(ascir::HintGraph &graph, bool is_f16 = true);
void LoadBroadcastStore_AfterAutofuse(ascir::ImplGraph &graph, bool is_f16 = true);

#endif
