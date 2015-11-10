/*!
 * \file document.h
 * \brief Defines Document data structure
 */

#ifndef LIGHTLDA_DOCUMENT_H_
#define LIGHTLDA_DOCUMENT_H_

#include "common.h"

namespace multiverso
{
    template <typename T> class Row;
}

namespace multiverso { namespace lightlda
{
    /*!
     * \brief Document presents a document. Document doesn't own memory, but   
     *  would interpret a contiguous piece of extern memory as a document
     *  with the format :
     *  #cursor, word1, topic1, word2, topic2, ..., wordn, topicn.#
     */
    class Document
    {
    public:
        /*!
         * \brief Constructs a document based on the start and end pointer
         */
        Document(int32_t* begin, int32_t* end);
        /*! \brief Get the length of the document */
        int32_t Size() const;
        /*! \brief Get the word based on the index */
        int32_t Word(int32_t index) const;
        /*! \brief Get the topic based on the index */
        int32_t Topic(int32_t index) const;
        /*! \brief Get the cursor */
        int32_t& Cursor();
        /*! \brief Set the topic based on the index */
        void SetTopic(int32_t index, int32_t topic);
        /*! \brief Get the doc-topic vector */
        void GetDocTopicVector(Row<int32_t>& vec);
    private:
        int32_t* begin_;
        int32_t* end_;
        int32_t& cursor_;

        // No copying allowed
        Document(const Document&);
        void operator=(const Document&);
    };

    // -- inline functions definition area --------------------------------- //
    inline int32_t Document::Size() const
    {
        return static_cast<int32_t>((end_ - begin_) / 2);
    }
    inline int32_t Document::Word(int32_t index) const
    {
        return *(begin_ + 1 + index * 2);
    }
    inline int32_t Document::Topic(int32_t index) const
    {
        return *(begin_ + 2 + index * 2);
    }
    inline int32_t& Document::Cursor() { return cursor_; }
    inline void Document::SetTopic(int32_t index, int32_t topic)
    {
        *(begin_ + 2 + index * 2) = topic;
    }
    // -- inline functions definition area --------------------------------- //

} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_DOCUMENT_H_
