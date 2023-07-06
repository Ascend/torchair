#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_TNG_STATUS_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_TNG_STATUS_H_

namespace tng {
class Status {
 public:
  bool IsSuccess() const { return status_ == nullptr; }
  const char *GetErrorMessage() const noexcept { return (status_ == nullptr) ? "" : status_; }
  ~Status() { delete[] status_; }

  Status() : status_(nullptr) {}
  Status(const Status &other);
  Status(Status &&other) noexcept;
  Status &operator=(const Status &other);
  Status &operator=(Status &&other) noexcept;

  static Status Success();
  static Status Error(const char *message, ...);

 private:
  char *status_;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_TNG_STATUS_H_