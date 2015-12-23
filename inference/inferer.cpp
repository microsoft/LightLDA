#include "inferer.h"

#include "alias_table.h"
#include "common.h"
#include "data_block.h"
#include "meta.h"
#include "sampler.h"
#include <multiverso/barrier.h>
#include <multiverso/stop_watch.h>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
{
    void Inferer::TrainIteration(DataBlockBase* data_block)
    {
        StopWatch watch; watch.Start();
        LDADataBlock* lda_data_block =
            reinterpret_cast<LDADataBlock*>(data_block);

        DataBlock& data = lda_data_block->data();
        int32_t block = lda_data_block->block();
        int32_t slice = lda_data_block->slice();
        int32_t iter = lda_data_block->iteration();
        const LocalVocab& local_vocab = data.meta();
        int32_t lastword = local_vocab.LastWord(slice);
        if (id_ == 0)
        {
            Log::Info("Iter = %d, Block = %d, Slice = %d\n",
                lda_data_block->iteration(),
                lda_data_block->block(), lda_data_block->slice());
        }
        //determin alias table
        if (id_ == 0) alias_->Init(meta_->alias_index(block, slice));
        pthread_barrier_wait(barrier_);
        // build alias table 
        if(0 == iter)
        {
            for (const int32_t* pword = local_vocab.begin(slice) + id_;
                pword < local_vocab.end(slice);
                pword += thread_num_)
            {
                alias_->Build(*pword, this);
            }
            if (id_ == 0) alias_->Build(-1, this);
            pthread_barrier_wait(barrier_);
            if (id_ == 0)
            {
                Log::Info("Alias Time used: %.2f s \n", watch.ElapsedSeconds());
            }
        }
        
        // Inference with lightlda sampler
        int32_t num_token = 0;
        for (int32_t doc_id = id_; doc_id < data.Size(); doc_id += thread_num_)
        {
            Document* doc = data.GetOneDoc(doc_id);
            num_token += sampler_->SampleOneDoc(doc, slice, lastword, this, alias_);
        }
        if (iter == Config::num_iterations - 1) alias_->Clear();
    }

    Row<int32_t>& Inferer::GetWordTopicRow(integer_t word_id)
    {
        return *(static_cast<Row<int32_t>*>(model_->GetWordTopicRow(word_id)));
    }

    void Inferer::UpdateWordTopic(integer_t word_id, integer_t topic_id, int32_t delta)
    {
    }

    Row<int64_t>& Inferer::GetSummaryRow()
    {
        return *(static_cast<Row<int64_t>*>(model_->GetSummaryRow(0)));
    }

    void Inferer::UpdateSummary(integer_t topic_id, int64_t delta)
    {
    }
} // namespace lightlda
} // namespace multiverso
