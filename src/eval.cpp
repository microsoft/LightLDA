#include "eval.h"

#include <cmath>

#include "common.h"
#include "document.h"
#include "trainer.h"

#include <multiverso/row.h>
#include <multiverso/row_iter.h>

namespace
{
    const double cof[6] = 
    { 
        76.18009172947146, -86.50532032941677,
        24.01409824083091, -1.231739572450155,
        0.1208650973866179e-2, -0.5395239384953e-5
    };

    double LogGamma(double xx)
    {
        int j;
        double x, y, tmp1, ser;
        y = xx;
        x = xx;
        tmp1 = x + 5.5;
        tmp1 -= (x + 0.5)*log(tmp1);
        ser = 1.000000000190015;
        for (j = 0; j < 6; j++) ser += cof[j] / ++y;
        return -tmp1 + log(2.5066282746310005*ser / x);
    }
}

namespace multiverso { namespace lightlda
{ 
    double Eval::ComputeOneDocLLH(Document* doc, Row<int32_t>& doc_topic_counter)
    {
        if (doc->Size() == 0) return 0.0;
        double one_doc_llh = LogGamma(Config::num_topics * Config::alpha)
            - Config::num_topics * LogGamma(Config::alpha);
        int32_t nonzero_num = 0;
        doc_topic_counter.Clear();
        doc->GetDocTopicVector(doc_topic_counter);
        Row<int32_t>::iterator iter = doc_topic_counter.Iterator();
        while (iter.HasNext())
        {
            one_doc_llh += LogGamma(iter.Value() + Config::alpha);
            ++nonzero_num;
            iter.Next();
        }
        one_doc_llh += (Config::num_topics - nonzero_num)
            * LogGamma(Config::alpha);
        one_doc_llh -= LogGamma(doc->Size() + 
            Config::alpha * Config::num_topics);
        return one_doc_llh;
    }

    double Eval::ComputeOneWordLLH(int32_t word, Trainer* trainer)
    {
        Row<int32_t>& params = trainer->GetRow<int32_t>(
            kWordTopicTable, word);
        if (params.NonzeroSize() == 0) return 0.0;
        double word_llh = 0.0;
        int32_t nonzero_num = 0;
        RowIterator<int32_t> iter = params.Iterator();
        while (iter.HasNext())
        {
            word_llh += LogGamma(iter.Value() + Config::beta);
            ++nonzero_num;
            iter.Next();
        }
        word_llh += (Config::num_topics - nonzero_num)
            * LogGamma(Config::beta);
        return word_llh;
    }

    double Eval::NormalizeWordLLH(Trainer* trainer)
    {
        Row<int64_t>& params = trainer->GetRow<int64_t>(kSummaryRow, 0);
        double llh = Config::num_topics *
            (LogGamma(Config::beta * Config::num_vocabs) -
            Config::num_vocabs * LogGamma(Config::beta));
        for (int32_t k = 0; k < Config::num_topics; ++k)
        {
            llh -= LogGamma(params.At(k)
                + Config::num_vocabs * Config::beta);
        }
        return llh;
    }
} // namespace lightlda
} // namespace multiverso
