/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: __clang_cce_runtime.h
 * Create: 2020-01-01
 */

#ifndef __CLANG_CCE_RUNTIME_H__
#define __CLANG_CCE_RUNTIME_H__
#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

// This interface is provided by runtime, it needs to be kept the same as their.
/**
 * @ingroup dvrt_clang_cce_runtime
 * @brief Config kernel launch parameters
 * @param [in] numBlocks block dimemsions
 * @param [in|out] smDesc  L2 memory usage control information
 * @param [in|out] stream associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
#ifdef __cplusplus
uint32_t rtConfigureCall(uint32_t numBlocks, void *smDesc = nullptr, void *stream = nullptr);
#else  // __cplusplus
uint32_t rtConfigureCall(uint32_t numBlocks, void *smDesc, void *stream);
#endif

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // __CLANG_CCE_RUNTIME_H__
