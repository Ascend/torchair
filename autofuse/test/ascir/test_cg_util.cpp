/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>
#include "cg_utils.h"
#include "ascir_ops.h"
namespace ascir {
namespace cg {
TEST(CgUtils, LoopGuardContextOk) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  int64_t count = 0;
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  LOOP(a, b, c) {
    ++count;
    ASSERT_NE(CgContext::GetThreadLocalContext(), nullptr);
  }
  ASSERT_EQ(count, 1);
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
}
TEST(CgUtils, LoopGuardAxisOk) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  LOOP(a, b, c) {
    ASSERT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes().size(), 3UL);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].name, a.name);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].id, a.id);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].name, b.name);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].id, b.id);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].name, c.name);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].id, c.id);
  }
}
TEST(CgUtils, LoopGuard_SchedAxis_Ok) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  LOOP(a, b, c) {
    auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT, {a, b});
    auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT, {b, c});
    auto mm = MatMul("mm", data0.y, data1.y);
  }

  auto data0 = graph.Find("data0");
  auto data1 = graph.Find("data1");
  auto mm = graph.Find("mm");
  ASSERT_EQ(std::vector<AxisId>(data0.attr.sched.axis), std::vector<AxisId>({a.id, b.id, c.id}));
  ASSERT_EQ(std::vector<AxisId>(data1.attr.sched.axis), std::vector<AxisId>({a.id, b.id, c.id}));
  ASSERT_EQ(std::vector<AxisId>(mm.attr.sched.axis), std::vector<AxisId>({a.id, b.id, c.id}));
}
}  // namespace cg
}  // namespace ascir