#include "document.h"

#include <multiverso/row.h>

namespace multiverso { namespace lightlda
{
    Document::Document(int32_t* begin, int32_t* end)
        : begin_(begin), end_(end), cursor_(*begin_)
    {}

    void Document::GetDocTopicVector(Row<int32_t>& topic_counter)
    {
        int32_t* p = begin_ + 2;
        int32_t num = 0;
        while (p < end_)
        {
            topic_counter.Add(*p, 1);
            ++p; ++p;
            if (++num == topic_counter.Capacity())
                return;
        }
    }
} // namespace lightlda
} // namespace multiverso
