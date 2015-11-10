/*!
 * \file data_block.h
 * \brief Defines the training data block
 */

#ifndef LIGHTLDA_DATA_BLOCK_H_
#define LIGHTLDA_DATA_BLOCK_H_

#include "common.h"

#include <multiverso/multiverso.h>

#include <memory>
#include <string>
#include <vector>

namespace multiverso { namespace lightlda
{
    class Document;
    class LocalVocab;
    /*!
     * \brief DataBlock is the an unit of the training dataset, 
     *  it correspond to a data block file in disk. 
     */
    class DataBlock
    {
    public:
        DataBlock();
        ~DataBlock();
        /*! \brief Reads a block of data into data block from disk */
        void Read(std::string file_name);
        /*! \brief Writes a block of data to disk */
        void Write();
        
        bool HasLoad() const;

        /*! \brief Gets the size (number of documents) of data block */
        DocNumber Size() const;
        /*!
         * \brief Gets one document
         * \param index index of document
         * \return pointer to document
         */
        Document* GetOneDoc(int32_t index);

        // mutator and accessor methods
        const LocalVocab& meta() const;
        void set_meta(const LocalVocab* local_vocab);
    private:
        void GenerateDocuments();
        bool has_read_;
        /*! \brief size of memory pool for document offset */
        int64_t max_num_document_;
        /*! \brief size of memory pool for documents */
        int64_t memory_block_size_;
        /*! \brief index to each document */
        std::vector<std::shared_ptr<Document>> documents_;
        /*! \brief number of document in this block */
        DocNumber num_document_;
        /*! \brief memory pool to store the document offset */
        int64_t* offset_buffer_;
        /*! \brief actual memory size used */
        int64_t corpus_size_;
        /*! \brief memory pool to store the documents */
        int32_t* documents_buffer_;
        /*! \brief meta(vocabs) information of current data block */
        const LocalVocab* vocab_;
        /*! \brief file name in disk */
        std::string file_name_;
        // No copying allowed
        DataBlock(const DataBlock&);
        void operator=(const DataBlock&);
    };

    /*!
     * \brief LDADataBlock is a logic data block that multiverso used to 
     *  train lightlda
     */
    class LDADataBlock : public DataBlockBase
    {
    public:
        // mutator and accessor methods
        int32_t block() const;
        void set_block(int32_t block);
        int32_t slice() const;
        void set_slice(int32_t slice);
        int32_t iteration() const;
        void set_iteration(int32_t iteration);
        DataBlock& data();
        void set_data(DataBlock* data);
    private:
        /*! \brief the actual data block */
        DataBlock* data_;
        /*! \brief the data block id */
        int32_t block_;
        /*! \brief the slice id */
        int32_t slice_;
        /*! \brief the i-th iteration */
        int32_t iteration_;
    };

    // -- inline functions definition area --------------------------------- //

    inline bool DataBlock::HasLoad() const { return has_read_; }
    inline Document* DataBlock::GetOneDoc(int32_t index)
    { 
        return documents_[index].get(); 
    }
    inline const LocalVocab& DataBlock::meta() const  { return *vocab_; }
    inline void DataBlock::set_meta(const LocalVocab* local_vocab)
    {
        vocab_ = local_vocab;
    }
    inline int32_t LDADataBlock::block() const { return block_; }
    inline void LDADataBlock::set_block(int32_t block) { block_ = block; }
    inline int32_t LDADataBlock::slice() const { return slice_; }
    inline void LDADataBlock::set_slice(int32_t slice) { slice_ = slice; }
    inline int32_t LDADataBlock::iteration() const { return iteration_; }
    inline void LDADataBlock::set_iteration(int32_t iteration)
    { 
        iteration_ = iteration; 
    }

    inline DataBlock& LDADataBlock::data() { return *data_; }
    inline void LDADataBlock::set_data(DataBlock* data) { data_ = data; }
    inline DocNumber DataBlock::Size() const { return num_document_; }

    // -- inline functions definition area --------------------------------- //

} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_DATA_BLOCK_H_
