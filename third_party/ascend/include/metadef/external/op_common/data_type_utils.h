/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*!
 * \file data_type_utils.h
 * \brief
 */

#ifndef EXTERNAL_OP_COMMON_DATA_TYPE_UTILS_H_
#define EXTERNAL_OP_COMMON_DATA_TYPE_UTILS_H_

#include "graph/types.h"

namespace opcommon {
inline bool IsComplexType(const ge::DataType type) {
  return (type == ge::DataType::DT_COMPLEX32 || type == ge::DataType::DT_COMPLEX64 ||
          type == ge::DataType::DT_COMPLEX128);
}

inline bool IsFloatingType(const ge::DataType type)
{
    return (type == ge::DataType::DT_DOUBLE || type == ge::DataType::DT_FLOAT || type == ge::DataType::DT_BF16
        || type == ge::DataType::DT_FLOAT16);
}

inline bool IsIntegralType(const ge::DataType type)
{
    return (type == ge::DataType::DT_INT8 || type == ge::DataType::DT_INT16 || type == ge::DataType::DT_INT32
        || type == ge::DataType::DT_INT64 || type == ge::DataType::DT_UINT8 || type == ge::DataType::DT_UINT16
        || type == ge::DataType::DT_UINT32 || type == ge::DataType::DT_UINT64);
}

inline bool IsIntegralType(const ge::DataType type, const bool include_bool)
{
    bool is_integral = IsIntegralType(type);
    return include_bool ? (is_integral || (type == ge::DataType::DT_BOOL)) : is_integral;
}

inline bool CanCast(const ge::DataType from, const ge::DataType to)
{
    if (IsComplexType(from) && !IsComplexType(to)) {
        return false;
    }

    if (IsFloatingType(from) && IsIntegralType(to, false)) {
        return false;
    }

    if (from != ge::DataType::DT_BOOL && to == ge::DataType::DT_BOOL) {
        return false;
    }

    return true;
}

inline ge::DataType PromoteType(ge::DataType type_a, ge::DataType type_b)
{
    if (type_a < 0 || type_b < 0 || type_a >= ge::DataType::DT_MAX || type_b >= ge::DataType::DT_MAX) {
        return ge::DataType::DT_UNDEFINED;
    }

    if (type_a == type_b) {
        return type_a;
    }

    constexpr auto u1 = ge::DataType::DT_UINT8;
    constexpr auto i1 = ge::DataType::DT_INT8;
    constexpr auto i2 = ge::DataType::DT_INT16;
    constexpr auto i4 = ge::DataType::DT_INT32;
    constexpr auto i8 = ge::DataType::DT_INT64;
    constexpr auto f2 = ge::DataType::DT_FLOAT16;
    constexpr auto f4 = ge::DataType::DT_FLOAT;
    constexpr auto f8 = ge::DataType::DT_DOUBLE;
    constexpr auto c2 = ge::DataType::DT_COMPLEX32;
    constexpr auto c4 = ge::DataType::DT_COMPLEX64;
    constexpr auto c8 = ge::DataType::DT_COMPLEX128;
    constexpr auto b1 = ge::DataType::DT_BOOL;
    constexpr auto bf = ge::DataType::DT_BF16;
    constexpr auto ud = ge::DataType::DT_UNDEFINED;
    // @formatter:off
    static constexpr ge::DataType kPromoteTypesLookup[static_cast<int>(
        ge::DataType::DT_MAX)][static_cast<int>(ge::DataType::DT_MAX)] = {
        /*          f4  f2  i1  i4  u1  xx  i2  u2  u4  i8  u8  f8  b1  sv  d1  D1  c4  c8  q1  q2  q4  Q1  Q2  rs  sr  du  va  bf, ud  t4  T1  t2  T2  c2*/
        /* f4 0 */ {f4, f4, f4, f4, f4, ud, f4, ud, ud, f4, ud, f8, f4, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, f4, ud, ud, ud, ud, ud, c4},
        /* f2 1 */ {f4, f2, f2, f2, f2, ud, f2, ud, ud, f2, ud, f8, f2, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, f4, ud, ud, ud, ud, ud, c2},
        /* i1 2 */ {f4, f2, i1, i4, i2, ud, i2, ud, ud, i8, ud, f8, i1, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, bf, ud, ud, ud, ud, ud, c2},
        /* i4 3 */ {f4, f2, i4, i4, i4, ud, i4, ud, ud, i8, ud, f8, i4, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, bf, ud, ud, ud, ud, ud, c2},
        /* u1 4 */ {f4, f2, i2, i4, u1, ud, i2, ud, ud, i8, ud, f8, u1, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, bf, ud, ud, ud, ud, ud, c2},
        /* xx 5 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* i2 6 */ {f4, f2, i2, i4, i2, ud, i2, ud, ud, i8, ud, f8, i2, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, bf, ud, ud, ud, ud, ud, c2},
        /* u2 7 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* u4 8 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* i8 9 */ {f4, f2, i8, i8, i8, ud, i8, ud, ud, i8, ud, f8, i8, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, bf, ud, ud, ud, ud, ud, c2},
        /* u8 10*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, c8, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* f8 11*/ {f8, f8, f8, f8, f8, ud, f8, ud, ud, f8, ud, f8, f8, ud, ud, ud, c8, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, f8, ud, ud, ud, ud, ud, c8},
        /* b1 12*/ {f4, f2, i1, i4, u1, ud, i2, ud, ud, i8, ud, f8, b1, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, bf, ud, ud, ud, ud, ud, c2},
        /* sv 13*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* d1 14*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* D1 15*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* c4 16*/ {c4, c4, c4, c4, c4, ud, c4, ud, ud, c4, ud, c8, c4, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, c4, ud, ud, ud, ud, ud, c4},
        /* c8 17*/ {c8, c8, c8, c8, c8, ud, c8, ud, ud, c8, ud, c8, c8, ud, ud, ud, c8, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, c8, ud, ud, ud, ud, ud, c8},
        /* q1 18*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* q2 19*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* q4 20*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* Q1 21*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* Q2 22*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* rs 23*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* sr 24*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* du 25*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* va 26*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* bf 27*/ {f4, f4, bf, bf, bf, ud, bf, ud, ud, bf, ud, f8, bf, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, bf, ud, ud, ud, ud, ud, c4},
        /* ud 28*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* t4 29*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* T1 30*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* t2 31*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* T2 32*/ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* c2 33*/ {c4, c2, c2, c2, c2, ud, c2, ud, ud, c2, ud, c8, c2, ud, ud, ud, c4, c8, ud, ud, ud, ud, ud, ud, ud, ud, ud, c4, ud, ud, ud, ud, ud, c2},
    };
    // @formatter:on
    return kPromoteTypesLookup[static_cast<int>(type_a)][static_cast<int>(type_b)];
}
} // namespace opcommon

#endif // EXTERNAL_OP_COMMON_DATA_TYPE_UTILS_H_
