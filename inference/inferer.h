/*!
 * \file inferer.h
 * \brief data inference
 */
#ifndef LIGHTLDA_INFERER_H_
#define LIGHTLDA_INFERER_H_

// #include <pthread.h>
#include <multiverso/multiverso.h>
#include <multiverso/log.h>
#include <multiverso/barrier.h>

namespace multiverso 
{ 
    class Barrier;

namespace lightlda
{
    class AliasTable;
    class LDADataBlock;
    class LightDocSampler;
    class Meta;
    class LocalModel;
    class IDataStream;
    
    class Inferer
    {
    public:
        Inferer(AliasTable* alias_table, 
                IDataStream * data_stream,
                Meta* meta, LocalModel * model,
                Barrier* barrier, 
                int32_t id, int32_t thread_num);

        ~Inferer();
        void BeforeIteration(int32_t block);
        void DoIteration(int32_t iter);
        void EndIteration();
    private:
        AliasTable* alias_;
        IDataStream * data_stream_;
        Meta* meta_;
        LocalModel * model_;
        Barrier* barrier_;
        int32_t id_;
        int32_t thread_num_;
        LightDocSampler* sampler_;
    };
} // namespace lightlda
} // namespace multiverso


 #endif //LIGHTLDA_INFERER_H_
