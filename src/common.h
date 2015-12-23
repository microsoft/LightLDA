/*! 
 * \file common.h
 * \brief Defines common settings in LightLDA
 */

#ifndef LIGHTLDA_COMMON_H_
#define LIGHTLDA_COMMON_H_

#include <cstdint>
#include <string>
#include <unordered_map>

namespace multiverso { namespace lightlda
{
    /*! \brief constant variable for table id */
    const int32_t kWordTopicTable = 0;
    /*! \brief constant variable for table id */
    const int32_t kSummaryRow = 1;
    /*! \brief load factor for sparse hash table */
    const int32_t kLoadFactor = 2;
    /*! \brief max length of a document */
    const int32_t kMaxDocLength = 8192;

    // 
    typedef int64_t DocNumber;

    /*!
     * \brief Defines LightLDA configs
     */
    struct Config
    {
    public:
        /*! \brief Inits configs from command line arguments */
        static void Init(int argc, char* argv[]);
        /*! \brief size of vocabulary */
        static int32_t num_vocabs;
        /*! \brief number of topics */
        static int32_t num_topics;
        /*! \brief number of iterations for trainning */
        static int32_t num_iterations;
        /*! \brief number of metropolis-hastings steps */
        static int32_t mh_steps;
        /*! \brief number of servers for Multiverso setting */
        static int32_t num_servers;
        /*! \brief server endpoint file */
        static std::string server_file;
        /*! \brief number of worker threads */
        static int32_t num_local_workers;
        /*! \brief number of local aggregation threads */
        static int32_t num_aggregator;
        /*! \brief number of blocks to train in disk */
        static int32_t num_blocks;
        /*! \brief maximum number of documents in a block */
        static int64_t max_num_document;
        /*! \brief hyper-parameter for symmetric dirichlet prior */
        static float alpha;
        /*! \brief hyper-parameter for symmetric dirichlet prior */
        static float beta;
        /*! \brief path of input directory */
        static std::string input_dir;
        /*! \brief option specify whether warm_start */
        static bool warm_start;
        /*! \brief option specity whether use out of core computation */
        static bool out_of_core;
        /*! \brief memory capacity settings, for memory pools */
        static int64_t data_capacity;
        static int64_t model_capacity;
        static int64_t delta_capacity;
        static int64_t alias_capacity;
    private:
        /*! \brief Print usage */
		static void PrintUsage();
        /*! \brief Check if the configs are valid */
        static void Check();
    };
} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_COMMON_H_
