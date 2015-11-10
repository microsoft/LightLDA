/*!
* \file eval.h
* \brief Defines utility for evaluating likelihood of lda
*/

#ifndef LIGHTLDA_EVAL_H_
#define LIGHTLDA_EVAL_H_

#include "common.h"

namespace multiverso
{
    template <typename T>
    class Row;
}

namespace multiverso { namespace lightlda
{
    class Document;
    class Trainer;

    /*!
     * \brief Eval defines functions to compute the likelihood of lightlda
     *  Likelihood is split into doc-likelihood and word-likelihood. 
     *  The total likelihood can be get by adding these values.
     */
    class Eval
    {
    public:
        /*!
         * \brief Compute doc-likelihood for one document
         * \param doc input document for evaluation
         */
        static double ComputeOneDocLLH(Document* doc, 
            Row<int32_t>& doc_topic_counter);

        /*!
         * \brief Compute word-likelihood for one word
         * \param word input word for evaluation
         * \param trainer for multiverso parameter access 
         */
        static double ComputeOneWordLLH(int32_t word, Trainer* trainer);

        /*!
         * \brief Compute normalization item for word-likelihood
         * \param trainer for multiverso parameter access
         */
        static double NormalizeWordLLH(Trainer* trainer);
    };
} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_EVAL_H_
