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
#include <multiverso/barrier.h>

namespace multiverso { namespace lightlda
{
    Inferer::Inferer(AliasTable* alias_table,
        IDataStream * data_stream,
        Meta* meta, LocalModel * model,
        Barrier* barrier, 
        int32_t id, int32_t thread_num):
        alias_(alias_table), data_stream_(data_stream),
        meta_(meta), model_(model),
        barrier_(barrier), 
        id_(id), thread_num_(thread_num) 
    {
        sampler_ = new LightDocSampler();
    }

    Inferer::~Inferer()
    {
        delete sampler_;
    }

    void Inferer::BeforeIteration(int32_t block)
    {
        //init current data block
        if(id_ == 0)
        {
	    data_stream_->BeforeDataAccess();
            DataBlock& data = data_stream_->CurrDataBlock();
            data.set_meta(&(meta_->local_vocab(block)));
            alias_->Init(meta_->alias_index(block, 0));
            alias_->Build(-1, model_);
	}
        barrier_->Wait();

        // build alias table 
	DataBlock& data = data_stream_->CurrDataBlock();
        const LocalVocab& local_vocab = data.meta();
        StopWatch watch; watch.Start();
        for (const int32_t* pword = local_vocab.begin(0) + id_;
            pword < local_vocab.end(0);
            pword += thread_num_)
        {
            alias_->Build(*pword, model_);
        }
        barrier_->Wait();
        if (id_ == 0)
        {
            Log::Info("block=%d, Alias Time used: %.2f s \n", block, watch.ElapsedSeconds());
        }
    }

    void Inferer::DoIteration(int32_t iter)
    {
        if (id_ == 0)
        {
            Log::Info("iter=%d\n", iter);
        }
	DataBlock& data = data_stream_->CurrDataBlock();
        const LocalVocab& local_vocab = data.meta();
        int32_t lastword = local_vocab.LastWord(0);
        // Inference with lightlda sampler
        for (int32_t doc_id = id_; doc_id < data.Size(); doc_id += thread_num_)
        {
            Document* doc = data.GetOneDoc(doc_id);
            //TODO: Asymmeric prior
            sampler_->SampleOneDoc(doc, 0, lastword, model_, alias_, nullptr);
        }
    }

    void Inferer::EndIteration()
    {
        barrier_->Wait();
        if(id_ == 0)
        {
            data_stream_->EndDataAccess();
            alias_->Clear();
        }
    }

} // namespace lightlda
} // namespace multiverso
