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

#ifndef AUTOFUSE_TEST_UTIL_H
#define AUTOFUSE_TEST_UTIL_H

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const int64_t &expect) {
  int64_t value = -1;
  ge::AttrUtils::GetInt(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const ge::DataType &expect) {
  ge::DataType value = ge::DT_UNDEFINED;
  ge::AttrUtils::GetDataType(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const vector<int64_t> &expect) {
  vector<int64_t> value;
  ge::AttrUtils::GetListInt(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const vector<vector<int64_t>> &expect) {
  vector<vector<int64_t>> value;
  ge::AttrUtils::GetListListInt(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

#endif  // AUTOFUSE_TEST_UTIL_H
