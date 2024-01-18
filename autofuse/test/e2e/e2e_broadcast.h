#ifndef __E2E_BROADCAST_H__
#define __E2E_BROADCAST_H__

#include "ascir.h"

void LoadBroadcastStore_BeforeAutofuse(ascir::HintGraph &graph);
void LoadBroadcastStore_AfterAutofuse(ascir::ImplGraph &graph);

#endif
