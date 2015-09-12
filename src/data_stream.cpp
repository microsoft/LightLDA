#include "data_stream.h"
#include "common.h"
#include "data_block.h"

#include <vector>
#include <thread>

#include <multiverso/double_buffer.h>

namespace multiverso { namespace lightlda
{
    class MemoryDataStream :public IDataStream
    {
    public:
        MemoryDataStream(int32_t num_blocks, std::string data_path);
        virtual ~MemoryDataStream();
        virtual void BeforeDataAccess() override;
        virtual void EndDataAccess() override;
        virtual DataBlock& CurrDataBlock() override;
    private:
        std::vector<DataBlock*> data_buffer_;
        std::string data_path_;
        int32_t index_;

        // No copying allowed
        MemoryDataStream(const MemoryDataStream&);
        void operator=(const MemoryDataStream&);
    };

    class DiskDataStream : public IDataStream
    {
        typedef DoubleBuffer<DataBlock> DataBuffer;
    public:
        DiskDataStream(int32_t num_blocks, std::string data_path,
            int32_t num_iterations);
        virtual ~DiskDataStream();
        virtual void BeforeDataAccess() override;
        virtual void EndDataAccess() override;
        virtual DataBlock& CurrDataBlock() override;
    private:
        /*! \brief Background data thread entrance function */
        void DataPreloadMain();
        /*! \brief buffer for data */
        DataBlock* buffer_0;
        DataBlock* buffer_1;
        DataBuffer* data_buffer_;
        /*! \brief current block id to be accessed */
        int32_t block_id_;
        /*! \brief number of data blocks in disk */
        int32_t num_blocks_;
        /*! \brief number of training iterations */
        int32_t num_iterations_;
        /*! \brief data path */
        std::string data_path_;
        /*! \brief backend thread for data preload */
        std::thread preload_thread_;
        bool working_;

        // No copying allowed
        DiskDataStream(const DiskDataStream&);
        void operator=(const DiskDataStream&);
    };

    MemoryDataStream::MemoryDataStream(int32_t num_blocks, std::string data_path)
        : data_path_(data_path), index_(0)
    {
        data_buffer_.resize(num_blocks, nullptr);
        for (int32_t i = 0; i < num_blocks; ++i)
        {
            data_buffer_[i] = new DataBlock();
            data_buffer_[i]->Read(data_path_ + "/block."
                + std::to_string(i));
        }
    }
    MemoryDataStream::~MemoryDataStream()
    {
        for (auto& data : data_buffer_)
        {
            data->Write();
            delete data;
            data = nullptr;
        }
    }
    void MemoryDataStream::BeforeDataAccess()
    {
        index_ %= data_buffer_.size();
    }
    void MemoryDataStream::EndDataAccess()
    {
        ++index_;
    }

    DataBlock& MemoryDataStream::CurrDataBlock()
    {
        return *data_buffer_[index_];
    }

    DiskDataStream::DiskDataStream(int32_t num_blocks,
        std::string data_path, int32_t num_iterations) :
        num_blocks_(num_blocks), data_path_(data_path),
        num_iterations_(num_iterations), working_(false)
    {
        block_id_ = 0;
        buffer_0 = new DataBlock();
        buffer_1 = new DataBlock();
        data_buffer_ = new DataBuffer(1, buffer_0, buffer_1);
        preload_thread_ = std::thread(&DiskDataStream::DataPreloadMain, this);
        while (!working_)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    DiskDataStream::~DiskDataStream()
    {
        preload_thread_.join();
        if (data_buffer_ != nullptr)
        {
            delete data_buffer_;
            data_buffer_ = nullptr;
            delete buffer_1;
            buffer_1 = nullptr;
            delete buffer_0;
            buffer_0 = nullptr;
        }
    }

    DataBlock& DiskDataStream::CurrDataBlock()
    {
        return data_buffer_->WorkerBuffer();
    }

    void DiskDataStream::BeforeDataAccess()
    {
        data_buffer_->Start(1);
    }

    void DiskDataStream::EndDataAccess()
    {
        data_buffer_->End(1);
    }

    void DiskDataStream::DataPreloadMain()
    {
        int32_t block_id = 0;
        std::string block_file = data_path_ + "/block."
            + std::to_string(block_id);
        data_buffer_->Start(0);
        data_buffer_->IOBuffer().Read(block_file);
        data_buffer_->End(0);
        working_ = true;
        for (int32_t iter = 0; iter <= num_iterations_; ++iter)
        {
            for (int32_t block_id = 0; block_id < num_blocks_; ++block_id)
            {
                data_buffer_->Start(0);

                DataBlock& data_block = data_buffer_->IOBuffer();
                if (data_block.HasLoad())
                {
                    data_block.Write();
                }
                if (iter == num_iterations_ && block_id == num_blocks_ - 1)
                {
                    break;
                }
                // Load New data;
                int32_t next_block_id = (block_id + 1) % num_blocks_;
                block_file = data_path_ + "/block." + 
                    std::to_string(next_block_id);
                data_block.Read(block_file);
                data_buffer_->End(0);
            }
        }
    }

    IDataStream* CreateDataStream()
    {
        if (Config::out_of_core && Config::num_blocks != 1)
        {
            return new DiskDataStream(Config::num_blocks, Config::input_dir,
                Config::num_iterations);
        }
        else
        {
            return new MemoryDataStream(Config::num_blocks, Config::input_dir);
        }
    }
} // namespace lightlda
} // namespace multiverso
