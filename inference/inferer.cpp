#include "inferer.h"

#include "alias_table.h"
#include "common.h"
#include "data_block.h"
#include "meta.h"
#include "sampler.h"
#include "model.h"
#include "data_stream.h"
#include <multiverso/stop_watch.h>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
{
    Inferer::Inferer(AliasTable* alias_table,
        IDataStream * data_stream,
        Meta* meta, LocalModel * model,
        pthread_barrier_t* barrier, 
        int32_t id, int32_t thread_num):
        alias_(alias_table), data_stream_(data_stream),
        meta_(meta), model_(model),
        barrier_(barrier), 
        id_(id), thread_num_(thread_num) 
    {
        sampler_ = new LightDocSampler();
        lda_data_block_ = new LDADataBlock();
    }

    Inferer::~Inferer()
    {
        delete sampler_;
        delete lda_data_block_;
    }

    void Inferer::BeforeIteration(int32_t block)
    {
        //get data block
        if(id_ == 0) data_stream_->BeforeDataAccess();
        pthread_barrier_wait(barrier_);
        DataBlock& data = data_stream_->CurrDataBlock();
        data.set_meta(&(meta_->local_vocab(block)));
        lda_data_block_->set_data(&data);
        lda_data_block_->set_block(block);
        lda_data_block_->set_slice(0);        

        StopWatch watch; watch.Start();
        const LocalVocab& local_vocab = data.meta();
        //determin alias index
        if (id_ == 0) alias_->Init(meta_->alias_index(block, 0));
        pthread_barrier_wait(barrier_);
        // build alias table 
        for (const int32_t* pword = local_vocab.begin(0) + id_;
            pword < local_vocab.end(0);
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

    void Inferer::DoIteration(int32_t iter)
    {
        lda_data_block_->set_iteration(iter);
        DataBlock& data = lda_data_block_->data();
        int32_t block = lda_data_block_->block();
        const LocalVocab& local_vocab = data.meta();
        int32_t lastword = local_vocab.LastWord(0);
        if (id_ == 0)
        {
            Log::Info("Iter = %d, Block = %d\n", iter, block);
        }
        // wait for all threads
        pthread_barrier_wait(barrier_);
        // Inference with lightlda sampler
        for (int32_t doc_id = id_; doc_id < data.Size(); doc_id += thread_num_)
        {
            Document* doc = data.GetOneDoc(doc_id);
            sampler_->SampleOneDoc(doc, 0, lastword, model_, alias_);
        }
    }

    void Inferer::EndIteration()
    {
        pthread_barrier_wait(barrier_);
        if(id_ == 0)
        {
            data_stream_->EndDataAccess();
            alias_->Clear();
        }
    }

} // namespace lightlda
} // namespace multiverso
