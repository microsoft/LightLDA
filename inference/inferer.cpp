#include "inferer.h"

#include "alias_table.h"
#include "common.h"
#include "data_block.h"
#include "meta.h"
#include "sampler.h"
#include "model.h"
#include <multiverso/stop_watch.h>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
{
    Inferer::Inferer(AliasTable* alias_table, Meta* meta, LocalModel * model,
        pthread_barrier_t* barrier, 
        int32_t id, int32_t thread_num):
        alias_(alias_table), meta_(meta), model_(model),
        barrier_(barrier), 
        id_(id), thread_num_(thread_num) 
    {
        sampler_ = new LightDocSampler();
    }

    Inferer::~Inferer()
    {
        delete sampler_;
    }

    void Inferer::InferenceIteration(DataBlockBase* data_block)
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
                alias_->Build(*pword, model_);
            }
            if (id_ == 0) alias_->Build(-1, model_);
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
            num_token += sampler_->SampleOneDoc(doc, slice, lastword, model_, alias_);
        }
        if (iter == Config::num_iterations - 1) alias_->Clear();
    }
} // namespace lightlda
} // namespace multiverso
