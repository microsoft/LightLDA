/*!
 * \file meta.h
 * \brief This file defines meta information for training dataset
 */

#ifndef _LIGHTLDA_META_H_
#define _LIGHTLDA_META_H_

#include <string>
#include <vector>
#include <cstdint>

namespace multiverso { namespace lightlda
{
    /*!
     * \brief LocalVocab defines the meta information of a data block. 
     *  It containes 1) which words occurs in this block, 2) slice information
     */
    class LocalVocab
    {
    public:
        friend class Meta;
        LocalVocab();
        ~LocalVocab();
        /*! \brief Get the last word of current slice */
        int32_t LastWord(int32_t slice) const;
        /*! \brief Get the number of slice */
        int32_t num_slice() const;
        /*! \brief Get the pointer to first word in this slice */
        const int* begin(int32_t slice) const;
        /*! \brief Get the pointer to last word + 1 in this slice */
        const int32_t* end(int32_t slice) const;
    private:
        int32_t num_slices_;
        int32_t* vocabs_;
        int32_t size_;
        bool own_memory_;
        std::vector<int32_t> slice_index_;
    };


    struct WordEntry
    {
        bool is_dense;
        int64_t begin_offset;
        int32_t capacity;
    };

    class AliasTableIndex
    {
    public:
        AliasTableIndex();
        WordEntry& word_entry(int32_t word);
        void PushWord(int32_t word, bool is_dense,
            int64_t begin_offset, int32_t capacity);
    private:
        std::vector<WordEntry> index_;
        std::vector<int32_t> index_map_;
    };

    /*!
     * \brief Meta containes all the meta information of training data in 
     *  current process. It containes 1) all the local vacabs for all data
     *  blocks, 2) the global tf for the training dataset
     */
    class Meta
    {
    public:
        Meta();
        ~Meta();
        /*! \brief Initialize the Meta information */
        void Init();
        /*! \brief Get the tf of word in the whole dataset */
        int32_t tf(int32_t word) const;
        /*! \brief Get the tf of word in local dataset */
        int32_t local_tf(int32_t word) const;
        /*! \brief Get the local vocab based on block id */
        const LocalVocab& local_vocab(int32_t id) const;

        AliasTableIndex* alias_index(int32_t block, int32_t slice);
    private:
        /*! \brief Schedule the model and split as slices based on memory */
        void ModelSchedule();
        /*! \brief Build index for alias table */
        void BuildAliasIndex();
    private:
        /*! \brief meta information for all data block */
        std::vector<LocalVocab> local_vocabs_;
        /*! \breif tf information for all word in the dataset */
        std::vector<int32_t> tf_;
        /*! \brief local tf information for all word in this machine */
        std::vector<int32_t> local_tf_;

        std::vector<std::vector<AliasTableIndex*> > alias_index_;
        // No copying allowed
        Meta(const Meta&);
        void operator=(const Meta&);
    };

    // -- inline functions definition area --------------------------------- //
    inline int32_t LocalVocab::LastWord(int32_t slice) const
    {
        return vocabs_[slice_index_[slice + 1] - 1];
    }
    inline int32_t LocalVocab::num_slice() const { return num_slices_; }
    inline const int32_t* LocalVocab::begin(int32_t slice) const
    {
        return vocabs_ + slice_index_[slice];
    }
    inline const int32_t* LocalVocab::end(int32_t slice) const
    {
        return vocabs_ + slice_index_[slice + 1];
    }
    inline int32_t Meta::tf(int32_t word) const { return tf_[word]; }
    inline int32_t Meta::local_tf(int32_t word) const { return local_tf_[word]; }
    inline const LocalVocab& Meta::local_vocab(int32_t id) const
    {
        return local_vocabs_[id]; 
    }
    inline AliasTableIndex* Meta::alias_index(int32_t block, int32_t slice)
    {
        return alias_index_[block][slice];
    }
    // -- inline functions definition area --------------------------------- //

} // namespace lightlda
} // namespace multiverso

#endif // _LIGHTLDA_META_H_
