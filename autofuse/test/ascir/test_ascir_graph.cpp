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
#include "gtest/gtest.h"
#include "ascir.h"
#include "ascir_ops.h"

#include "node_utils_ex.h"
#include "graph_utils_ex.h"
using namespace ascir;

TEST(Ascir_Graph, AddStartNode_Ok) {
  Graph g("graph");
  ops::Data data("data");
  g.AddNode(data);

  auto cg = ge::GraphUtilsEx::GetComputeGraph(g);
  ASSERT_NE(cg, nullptr);
  auto data_node = cg->FindNode("data");
  ASSERT_NE(data_node, nullptr);
  auto data_node_in_op = ge::NodeUtilsEx::GetNodeFromOperator(data);
  ASSERT_NE(data_node_in_op, nullptr);
  ASSERT_EQ(data_node, data_node_in_op);
}

TEST(Ascir_Graph_Bg, AddStartNode_Ok) {
  Graph g("graph");
  auto data = cg::Data("data", g);

  auto cg = ge::GraphUtilsEx::GetComputeGraph(g);
  ASSERT_NE(cg, nullptr);
  auto data_node = cg->FindNode("data");
  ASSERT_NE(data_node, nullptr);
  auto data_node_in_op = ge::NodeUtilsEx::GetNodeFromOperator(data);
  ASSERT_NE(data_node_in_op, nullptr);
  ASSERT_EQ(data_node, data_node_in_op);
}