#include "common.h"
#include "trainer.h"
#include "alias_table.h"
#include "asym_alpha.h"
#include "data_stream.h"
#include "data_block.h"
#include "document.h"
#include "meta.h"
#include "model.h"
#include "util.h"
#include <vector>
#include <iostream>
#include <multiverso/barrier.h>
#include <multiverso/log.h>
#include <multiverso/row.h>

namespace multiverso { namespace lightlda
{     
    class LightLDA
    {
    public:
        static void Run(int argc, char** argv)
        {
            Config::Init(argc, argv);
            
            AliasTable* alias_table = new AliasTable();
            AsymAlpha* asym_alpha = nullptr;
            if(Config::asymmetric_prior)
            {
                asym_alpha = new AsymAlpha();
            }
            Barrier* barrier = new Barrier(Config::num_local_workers);
            meta.Init();
            std::vector<TrainerBase*> trainers;
            for (int32_t i = 0; i < Config::num_local_workers; ++i)
            {
                Trainer* trainer = new Trainer(alias_table, asym_alpha, barrier, &meta);
                trainers.push_back(trainer);
            }

            ParamLoader* param_loader = new ParamLoader();
            multiverso::Config config;
            config.num_servers = Config::num_servers;
            config.num_aggregator = Config::num_aggregator;
            config.server_endpoint_file = Config::server_file;

            Multiverso::Init(trainers, param_loader, config, &argc, &argv);

            Log::ResetLogFile("LightLDA."
                + std::to_string(clock()) + ".log");

            data_stream = CreateDataStream();
            InitDocTopic();
            PSModel::Init(&meta, data_stream);
            Train();

            Multiverso::Close();
            
            for (auto& trainer : trainers)
            {
                delete trainer;
            }
            delete param_loader;
            
            DumpDocTopic();

            delete data_stream;
            delete barrier;
            delete alias_table;
            if(Config::asymmetric_prior) delete asym_alpha;
        }
    private:
        static void Train()
        {
            Multiverso::BeginTrain();
            for (int32_t i = 0; i < Config::num_iterations; ++i)
            {
                Multiverso::BeginClock();
                // Train corpus block by block
                for (int32_t block = 0; block < Config::num_blocks; ++block)
                {
                    data_stream->BeforeDataAccess();
                    DataBlock& data_block = data_stream->CurrDataBlock();
                    data_block.set_meta(&meta.local_vocab(block));
                    int32_t num_slice = meta.local_vocab(block).num_slice();
                    std::vector<LDADataBlock> data(num_slice);
                    // Train datablock slice by slice
                    for (int32_t slice = 0; slice < num_slice; ++slice)
                    {
                        LDADataBlock* lda_block = &data[slice];
                        lda_block->set_data(&data_block);
                        lda_block->set_iteration(i);
                        lda_block->set_block(block);
                        lda_block->set_slice(slice);
                        Multiverso::PushDataBlock(lda_block);
                    }
                    Multiverso::Wait();
                    data_stream->EndDataAccess();
                }
                Multiverso::EndClock();
            }
            Multiverso::EndTrain();
        }

        static void InitDocTopic()
        {
            xorshift_rng rng;
            for (int32_t block = 0; block < Config::num_blocks; ++block)
            {
                data_stream->BeforeDataAccess();
                DataBlock& data_block = data_stream->CurrDataBlock();
                int32_t num_slice = meta.local_vocab(block).num_slice();
                for (int32_t slice = 0; slice < num_slice; ++slice)
                {
                    for (int32_t i = 0; i < data_block.Size(); ++i)
                    {
                        Document* doc = data_block.GetOneDoc(i);
                        int32_t& cursor = doc->Cursor();
                        if (slice == 0) cursor = 0;
                        int32_t last_word = meta.local_vocab(block).LastWord(slice);
                        for (; cursor < doc->Size(); ++cursor)
                        {
                            if (doc->Word(cursor) > last_word) break;
                            // Init the latent variable
                            if (!Config::warm_start)
                                doc->SetTopic(cursor, rng.rand_k(Config::num_topics));
                        }
                    }
                    Multiverso::Flush();
                }
                data_stream->EndDataAccess();
            }
        }

        static void DumpDocTopic()
        {
            Row<int32_t> doc_topic_counter(0, Format::Sparse, kMaxDocLength); 
            for (int32_t block = 0; block < Config::num_blocks; ++block)
            {
                std::ofstream fout("doc_topic." + std::to_string(block));
                data_stream->BeforeDataAccess();
                DataBlock& data_block = data_stream->CurrDataBlock();
                for (int i = 0; i < data_block.Size(); ++i)
                {
                    Document* doc = data_block.GetOneDoc(i);
                    doc_topic_counter.Clear();
                    doc->GetDocTopicVector(doc_topic_counter);
                    fout << i << " ";  // doc id
                    Row<int32_t>::iterator iter = doc_topic_counter.Iterator();
                    while (iter.HasNext())
                    {
                        fout << " " << iter.Key() << ":" << iter.Value();
                        iter.Next();
                    }
                    fout << std::endl;
                }
                data_stream->EndDataAccess();
            }
        }

    private:
        /*! \brief training data access */
        static IDataStream* data_stream;
        /*! \brief training data meta information */
        static Meta meta;
    };
    IDataStream* LightLDA::data_stream = nullptr;
    Meta LightLDA::meta;

} // namespace lightlda
} // namespace multiverso


int main(int argc, char** argv)
{
    multiverso::lightlda::LightLDA::Run(argc, argv);
    return 0;
}
