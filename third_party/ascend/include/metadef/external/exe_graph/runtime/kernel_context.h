/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_KERNEL_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_KERNEL_CONTEXT_H_
#include <type_traits>
#include "kernel_run_context.h"
namespace gert {
class Chain {
 public:
  using Deleter = void (*)(void *);
  /**
   * 获取Chain中保存的数据的指针
   * @tparam T 数据类型
   * @return 指向数据的指针
   */
  template<typename T, typename std::enable_if<(sizeof(T) <= sizeof(void *)), int>::type = 0>
  const T *GetPointer() const {
    return reinterpret_cast<const T *>(any_value_.data.inplace);
  }
  /**
   * 获取Chain中保存的数据的指针
   * @tparam T 数据类型
   * @return 指向数据的指针
   */
  template<typename T, typename std::enable_if<(sizeof(T) > sizeof(void *)), int>::type = 0>
  const T *GetPointer() const {
    return reinterpret_cast<const T *>(any_value_.data.pointer);
  }
  /**
   * 获取Chain中保存的数据的指针
   * @tparam T 数据类型
   * @return 指向数据的指针
   */
  template<typename T, typename std::enable_if<(sizeof(T) <= sizeof(void *)), int>::type = 0>
  auto GetPointer() -> T* {
    return reinterpret_cast<T *>(any_value_.data.inplace);
  }
  /**
   * 获取Chain中保存的数据的指针
   * @tparam T 数据类型
   * @return 指向数据的指针
   */
  template<typename T, typename std::enable_if<(sizeof(T) > sizeof(void *)), int>::type = 0>
  auto GetPointer() -> T* {
    return reinterpret_cast<T *>(any_value_.data.pointer);
  }
  /**
   * 获取Chain中保存的数据的值
   * @tparam T 数据类型
   * @return 数据的值的引用
   */
  template<typename T, typename std::enable_if<(sizeof(T) <= sizeof(void *)), int>::type = 0>
  const T &GetValue() const {
    return *reinterpret_cast<const T *>(any_value_.data.inplace);
  }
  /**
   * 获取Chain中保存的数据的值
   * @tparam T 数据类型
   * @return 数据的值的引用
   */
  template<typename T, typename std::enable_if<(sizeof(T) <= sizeof(void *)), int>::type = 0>
  auto GetValue() -> T& {
    return *reinterpret_cast<T *>(any_value_.data.inplace);
  }
  /**
   * 将数据设置到Chain中。设置数据指针时，会尝试调用deleter将原有保存在Chain的数据删除
   * @param data 指向数据的指针
   * @param deleter 释放数据的接口，空指针的含义为不需要释放
   */
  void Set(void * const data, const Chain::Deleter deleter) {
    FreeResource();
    any_value_.data.pointer = data;
    any_value_.deleter = deleter;
  }
  /**
   * 将数据设置到Chain中。设置数据指针时，会尝试调用deleter将原有保存在Chain的数据删除
   * @tparam T 数据的类型
   * @param data 数据的指针
   */
  template<typename T, typename std::enable_if<(!std::is_array<T>::value), int>::type = 0>
  void SetWithDefaultDeleter(T *data) {
    Set(data, reinterpret_cast<FreeCallback>(DefaultDeleter<T>));
  }
  /**
   * 将数据设置到Chain中。设置数据指针时，会尝试调用deleter将原有保存在Chain的数据删除
   * @tparam T 数据的类型
   * @param data 数据的指针
   */
  template<typename T, typename PureT = typename std::remove_extent<T>::type,
           typename std::enable_if<std::is_array<T>::value, int>::type = 0>
  void SetWithDefaultDeleter(PureT *data) {
    Set(data, reinterpret_cast<FreeCallback>(DefaultArrayDeleter<PureT>));
  }
  /**
   * 判断当前Chain中保存的数据是否有deleter
   * @return true代表含有deleter
   */
  bool HasDeleter() const {
    return any_value_.deleter != nullptr;
  }

 private:
  template<typename T>
  static void DefaultArrayDeleter(T *data) {
    delete[] data;
  }

  template<typename T>
  static void DefaultDeleter(T *data) {
    delete data;
  }

  void FreeResource() {
    if (any_value_.deleter != nullptr) {
      any_value_.deleter(any_value_.data.pointer);
    }
  }

  AsyncAnyValue any_value_;
};
static_assert(std::is_standard_layout<Chain>::value, "The class Chain must be a POD");

class KernelContext {
 public:
  /**
   * 获取kernel的输入数量
   * @return kernel的输入数量
   */
  size_t GetInputNum() const {
    return context_.input_size;
  }
  /**
   * 获取kernel的输出数量
   * @return kernel的输出数量
   */
  size_t GetOutputNum() const {
    return context_.output_size;
  }
  /**
   * 获取输入的Chain指针
   * @param i kernel的输入index
   * @return 输入Chain的指针
   */
  const Chain *GetInput(const size_t i) const {
    if (i >= context_.input_size) {
      return nullptr;
    }
    return reinterpret_cast<const Chain *>(context_.values[i]);
  }
    /**
   * 获取输入的Chain指针
   * @param i kernel的输入index
   * @return 输入Chain的指针
   */
  Chain *MutableInput(const size_t i) const {
    if (i >= context_.input_size) {
      return nullptr;
    }
    return reinterpret_cast<Chain *>(context_.values[i]);
  }
  /**
   * 获取输出的Chain指针
   * @param i kernel的输出index
   * @return 输出Chain的指针
   */
  Chain *GetOutput(const size_t i) {
    if (i >= context_.output_size) {
      return nullptr;
    }
    return reinterpret_cast<Chain *>(context_.values[context_.input_size + i]);
  }
  /**
   * 获取输出的Chain指针
   * @param i kernel的输出index
   * @return 输出Chain的指针
   */
  const Chain *GetOutput(const size_t i) const {
    if (i >= context_.output_size) {
      return nullptr;
    }
    return reinterpret_cast<const Chain *>(context_.values[context_.input_size + i]);
  }
  Chain *GetOutput2(const size_t i) {
    if (i >= context_.output_size) {
      return nullptr;
    }
    return reinterpret_cast<Chain *>(context_.output_start[i]);
  }
  /**
   * 获取输入数据的值，本函数首先获取输入Chain，然后从输入Chain中获取值
   * @tparam T 值的类型
   * @param i kernel的输入index
   * @return 输入的值
   */
  template<typename T>
  const T GetInputValue(size_t i) const {
    const auto av = GetInput(i);
    if (av == nullptr) {
      return {};
    }
    return av->GetValue<T>();
  }
  /**
   * 获取输入数据的指针，本函数首先获取输入Chain，然后从输入Chain中获取指针
   * @tparam T 值的类型
   * @param i kernel的输入index
   * @return 输入数据的指针
   */
  template<typename T>
  const T *GetInputPointer(size_t i) const {
    const auto av = GetInput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetPointer<T>();
  }
  /**
   * 获取输入数据的指针，本函数首先获取输入Chain，然后从输入Chain中获取指针
   * @tparam T 值的类型
   * @param i kernel的输入index
   * @return 输入数据的指针
   */
  template<typename T>
  auto MutableInputPointer(size_t i) const -> T* {
    const auto av = MutableInput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetPointer<T>();
  }
  /**
   * 获取输入字符串的指针，本函数首先获取输入Chain，然后从输入Chain中获取指针
   *
   * todo 特化一个模板就可以了
   * @param i kernel的输入index
   * @return 字符串的指针
   */
  const char *GetInputStrPointer(const size_t i) const {
    const auto av = GetInput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<const char *>();
  }
  /**
   * 获取计算节点信息的指针
   * @return 计算节点信息的指针
   */
  const void *GetComputeNodeExtend() const {
    return context_.compute_node_info;
  }
  /**
   * 获取kernel扩展信息的指针
   * @return
   */
  const void *GetKernelExtend() const {
    return context_.kernel_extend_info;
  }
  /**
   * 获取输出数据的指针，本函数首先获取输出Chain，然后从Chain中获取指针
   * @tparam T 数据的类型
   * @param i kernel的输出index
   * @return 输出数据的指针
   */
  template<typename T>
  auto GetOutputPointer(size_t i) -> T* {
    const auto av = GetOutput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetPointer<T>();
  }
  /**
   * 获取输出数据的指针，本函数首先获取输出Chain，然后从Chain中获取指针
   * @tparam T 数据的类型
   * @param i kernel的输出index
   * @return 输出数据的指针
   */
  template<typename T>
  const T *GetOutputPointer(size_t i) const {
    const auto av = GetOutput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetPointer<T>();
  }
  /**
   * 获取底层的context结构体，非框架代码请勿直接操作此结构体
   * @return 指向context结构体的指针
   */
  KernelRunContext *GetContext() {
    return &context_;
  }
  /**
   * 获取底层的context结构体，非框架代码请勿直接操作此结构体
   * @return 指向context结构体的指针
   */
  const KernelRunContext *GetContext() const {
    return &context_;
  }
  /**
   * 根据数据的长度判断一个数据是否会被Inline存储，所谓Inline存储是指此数据保存在context中时不需要单独分配内存
   * @param size 数据的长度
   * @return true代表会被inline存储
   */
  static bool IsInlineSize(const size_t size) {
    return size <= sizeof(void *);
  }

 private:
  KernelRunContext context_;
};
static_assert(std::is_standard_layout<KernelContext>::value, "The class KernelContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_KERNEL_CONTEXT_H_
