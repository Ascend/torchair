#include <future>
#include <utility>

#include "checker.h"

#include "acl/acl_rt.h"
#include "acl/acl_tdt.h"

namespace tng {
class ChannelData {
public:
  struct Item {
    aclDataType dtype = aclDataType::ACL_DT_UNDEFINED;
    std::vector<int64_t> dims;
    void *data{};
    size_t data_len{};
  };

  explicit ChannelData(acltdtDataset *dataset = nullptr, std::vector<Item> items = {}) : dataset(dataset),
                                                                                         items(std::move(items)) {}

  ChannelData &operator=(const ChannelData &) = delete;

  ChannelData &operator=(ChannelData &&) = delete;

  ChannelData(const ChannelData &) = delete;

  ChannelData(ChannelData &&other) noexcept : dataset(other.dataset), items(std::move(other.items)) {
    other.items.clear();
    other.dataset = nullptr;
  }

  ChannelData Move() {
    acltdtDataset *owned_dataset = dataset;
    auto owned_items = items;
    dataset = nullptr;
    items.clear();
    return ChannelData(owned_dataset, owned_items);
  }

  ~ChannelData() {
    if (dataset != nullptr) {
      acltdtDestroyDataset(dataset);
    }
    dataset = nullptr;
  }

  acltdtDataset *dataset;
  std::vector<Item> items;
};

class Channel {
public:
  static std::unique_ptr<Channel> Create(int32_t device, const std::string &name, size_t capacity) {
    acltdtChannelHandle *handle = acltdtCreateChannelWithCapacity(device, name.c_str(), capacity);
    if (handle == nullptr) {
      TNG_LOG(ERROR) << "Failed to create channel " << name;
      return nullptr;
    }
    return std::unique_ptr<Channel>(new Channel(handle));
  }

  ChannelData Receive() {
    if (handle_ == nullptr) {
      return ChannelData();
    }
    auto acl_dataset = acltdtCreateDataset();
    if (acl_dataset == nullptr) {
      return ChannelData();
    }

    ChannelData channel_data(acl_dataset);
    auto acl_status = acltdtReceiveTensor(handle_, acl_dataset, 1000);
    if (acl_status != ACL_ERROR_NONE) {
      return ChannelData();
    }

    size_t num_items = acltdtGetDatasetSize(acl_dataset);
    channel_data.items.resize(num_items);

    for (size_t i = 0u; i < num_items; i++) {
      auto &readable_item = channel_data.items[i];
      auto item = acltdtGetDataItem(acl_dataset, i);
      if (item == nullptr) {
        return ChannelData();
      }

      readable_item.dtype = acltdtGetDataTypeFromItem(item);
      size_t dims_num = acltdtGetDimNumFromItem(item);
      readable_item.dims.resize(dims_num);
      if (acltdtGetDimsFromItem(item, readable_item.dims.data(), dims_num) != ACL_ERROR_NONE) {
        return ChannelData();
      }
      readable_item.data_len = acltdtGetDataSizeFromItem(item);
      readable_item.data = acltdtGetDataAddrFromItem(item);
      if (readable_item.data == nullptr && readable_item.data_len != 0) {
        return ChannelData();
      }
    }

    return channel_data.Move();
  }

  void Destroy() {
    if (handle_ != nullptr) {
      (void) acltdtDestroyChannel(handle_);
    }
    handle_ = nullptr;
  }

  ~Channel() {
    Destroy();
  }

private:
  explicit Channel(acltdtChannelHandle *handle) : handle_(handle) {};
  acltdtChannelHandle *handle_;
};

class DeviceStdout {
public:
  static DeviceStdout &GetInstance(int32_t device) {
    static DeviceStdout instance(device);
    return instance;
  }

  ~DeviceStdout() {
    Stop();
  }

  tng::Status Start() {
    std::unique_lock<std::mutex> lock(mu_);
    if (worker_ != nullptr) {
      return tng::Status::Success();
    }
    if (channel_ == nullptr) {
      constexpr size_t kChannelCapacity = 2;
      channel_ = Channel::Create(device_, "_npu_log", kChannelCapacity);
    }
    if (channel_ == nullptr) {
      return tng::Status::Error("Failed to create device stdout channel");
    }
    worker_ = std::make_unique<std::thread>([this]() {
      size_t index = 0u;
      while (running_) {
        TNG_LOG(DEBUG) << "Start to receive npu device stdout index " << index;
        ChannelData data = channel_->Receive();
        TNG_LOG(DEBUG) << "Received npu device stdout " << index++;
        for (auto &item: data.items) {
          if (item.dtype != aclDataType::ACL_STRING) {
            continue;
          }
          if (item.data == nullptr || item.data_len == 0) {
            continue;
          }
          std::cerr << std::string(static_cast<char *>(item.data), item.data_len) << std::endl;
        }
      }
    });
    return tng::Status::Success();
  }

  tng::Status Stop() {
    running_ = false;
    if (channel_ != nullptr) {
      channel_->Destroy(); // tdt need destroy for relive block
    }
    if (worker_ != nullptr) {
      worker_->join();
    }
    worker_.reset(nullptr);
    channel_.reset(nullptr);
    return tng::Status::Success();
  }

private:
  explicit DeviceStdout(int32_t device) : device_(device) {}

  int32_t device_;
  std::unique_ptr<std::thread> worker_;
  std::unique_ptr<Channel> channel_;
  std::atomic<bool> running_{true};
  std::mutex mu_;
};

Status StartStdoutChannel(int32_t device) {
  return DeviceStdout::GetInstance(device).Start();
}

void StopStdoutChannel(int32_t device) {
  (void)DeviceStdout::GetInstance(device).Stop();
}
}  // namespace tng
