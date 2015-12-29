/*!
 * \file inferer.h
 * \brief data inference
 */
#ifndef LIGHTLDA_INFERER_H_
#define LIGHTLDA_INFERER_H_

#include <pthread.h>
#include <multiverso/multiverso.h>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
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
                pthread_barrier_t* barrier, 
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
        pthread_barrier_t* barrier_;
        int32_t id_;
        int32_t thread_num_;
        LightDocSampler* sampler_;
        LDADataBlock * lda_data_block_;
    };
} // namespace lightlda
} // namespace multiverso


 #endif //LIGHTLDA_INFERER_H_