#include "sampler.h"

#include "alias_table.h"
#include "common.h"
#include "document.h"
#include "model.h"

#include <multiverso/log.h>
#include <multiverso/row.h>

namespace multiverso { namespace lightlda
{
    LightDocSampler::LightDocSampler()
    {
        alpha_ = Config::alpha;
        beta_ = Config::beta;
        num_vocab_ = Config::num_vocabs;
        num_topic_ = Config::num_topics;
        mh_steps_ = Config::mh_steps;

        alpha_sum_ = num_topic_ * alpha_;
        beta_sum_ = num_vocab_ * beta_;

        subtractor_ = Config::inference ? 0 : 1;

        doc_topic_counter_.reset(new Row<int32_t>(0, 
            multiverso::Format::Sparse, kMaxDocLength));
    }

    int32_t LightDocSampler::SampleOneDoc(Document* doc, int32_t slice,
        int32_t lastword, ModelBase* model, AliasTable* alias)
    {
        DocInit(doc);
        int32_t num_tokens = 0;
        int32_t& cursor = doc->Cursor();
        if (slice == 0) cursor = 0;
        for (; cursor != doc->Size(); ++cursor)
        {
            int32_t word = doc->Word(cursor);
            if (word > lastword) break;
            int32_t old_topic = doc->Topic(cursor);
            int32_t new_topic = Sample(doc, word, old_topic, old_topic,
                model, alias);
            if (old_topic != new_topic)
            {
                doc->SetTopic(cursor, new_topic);
                doc_topic_counter_->Add(old_topic, -1);
                doc_topic_counter_->Add(new_topic, 1);
                if(!Config::inference)
                {
                    model->AddWordTopicRow(word, old_topic, -1);
                    model->AddSummaryRow(old_topic, -1);
                    model->AddWordTopicRow(word, new_topic, 1);
                    model->AddSummaryRow(new_topic, 1);
                }
            }
            ++num_tokens;
        }
        return num_tokens;
    }

    void LightDocSampler::DocInit(Document* doc)
    {
        doc_topic_counter_->Clear();
        doc->GetDocTopicVector(*doc_topic_counter_);
    }

    int32_t LightDocSampler::Sample(Document* doc,
        int32_t word, int32_t old_topic, int32_t s,
        ModelBase* model, AliasTable* alias)
    {
        int32_t t, w_t_cnt, w_s_cnt;
        int64_t n_t, n_s;
        float n_td_alpha, n_sd_alpha;
        float n_tw_beta, n_sw_beta, n_t_beta_sum, n_s_beta_sum;
        float proposal_t, proposal_s;
        float nominator, denominator;
        double rejection, pi;
        int32_t m;

        Row<int32_t>& word_topic_row = model->GetWordTopicRow(word);
        Row<int64_t>& summary_row = model->GetSummaryRow();

        for (int32_t i = 0; i < mh_steps_; ++i)
        {
            // Word proposal
            t = alias->Propose(word, rng_);
            if (t < 0 || t >= num_topic_)
            {
                Log::Fatal("Invalid topic assignment %d from word proposal\n", t);
            }
            if (t != s)
            {
                rejection = rng_.rand_double();

                w_t_cnt = word_topic_row.At(t);
                w_s_cnt = word_topic_row.At(s);
                n_t = summary_row.At(t);
                n_s = summary_row.At(s);

                n_td_alpha = doc_topic_counter_->At(t) + alpha_;
                n_sd_alpha = doc_topic_counter_->At(s) + alpha_;
                n_tw_beta = w_t_cnt + beta_;
                n_t_beta_sum = n_t + beta_sum_;
                n_sw_beta = w_s_cnt + beta_;
                n_s_beta_sum = n_s + beta_sum_;
                if (s == old_topic)
                {
                    --n_sd_alpha;
                    n_sw_beta -= subtractor_;
                    n_s_beta_sum -= subtractor_;
                }
                if (t == old_topic)
                {
                    --n_td_alpha;
                    n_tw_beta -= subtractor_;
                    n_t_beta_sum -= subtractor_;
                }

                proposal_s = (w_s_cnt + beta_) / (n_s + beta_sum_);
                proposal_t = (w_t_cnt + beta_) / (n_t + beta_sum_);

                nominator = n_td_alpha * n_tw_beta * n_s_beta_sum * proposal_s;
                denominator = n_sd_alpha * n_sw_beta * n_t_beta_sum * proposal_t;

                pi = nominator / denominator;

                m = -(rejection < pi);
                s = (t & m) | (s & ~m);
            }
            // Doc proposal
            double n_td_or_alpha = rng_.rand_double() *
                (doc->Size() + alpha_sum_);
            if (n_td_or_alpha < doc->Size())
            {
                int32_t t_idx = static_cast<int32_t>(n_td_or_alpha);
                t = doc->Topic(t_idx);
            }
            else
            {
                t = rng_.rand_k(num_topic_);
            }
            if (t != s)
            {
                rejection = rng_.rand_double();

                w_t_cnt = word_topic_row.At(t);
                w_s_cnt = word_topic_row.At(s);
                n_t = summary_row.At(t);
                n_s = summary_row.At(s);

                n_td_alpha = doc_topic_counter_->At(t) + alpha_;
                n_sd_alpha = doc_topic_counter_->At(s) + alpha_;
                n_tw_beta = w_t_cnt + beta_;
                n_t_beta_sum = n_t + beta_sum_;
                n_sw_beta = w_s_cnt + beta_;
                n_s_beta_sum = n_s + beta_sum_;
                if (s == old_topic)
                {
                    --n_sd_alpha;
                    n_sw_beta -= subtractor_;
                    n_s_beta_sum -= subtractor_;
                }
                if (t == old_topic)
                {
                    --n_td_alpha;
                    n_tw_beta -= subtractor_;
                    n_t_beta_sum -= subtractor_;
                    
                }

                proposal_s = (doc_topic_counter_->At(s) + alpha_);
                proposal_t = (doc_topic_counter_->At(t) + alpha_);

                nominator = n_td_alpha * n_tw_beta * n_s_beta_sum * proposal_s;
                denominator = n_sd_alpha * n_sw_beta * n_t_beta_sum * proposal_t;

                pi = nominator / denominator;

                m = -(rejection < pi);
                s = (t & m) | (s & ~m);
            }
        }
        return s;
    }

    int32_t LightDocSampler::ApproxSample(Document* doc,
        int32_t word, int32_t old_topic, int32_t s,
        ModelBase* model, AliasTable* alias)
    {
        float n_tw_beta, n_sw_beta, n_t_beta_sum, n_s_beta_sum;
        float nominator, denominator;
        double rejection, pi;
        int32_t m, t;
        
        Row<int32_t>& word_topic_row = model->GetWordTopicRow(word);
        Row<int64_t>& summary_row = model->GetSummaryRow();

        for (int32_t i = 0; i < mh_steps_; ++i)
        {
            // word proposal
            t = alias->Propose(word, rng_);
            if (t != s)
            {
                nominator = doc_topic_counter_->At(t) + alpha_;
                denominator = doc_topic_counter_->At(s) + alpha_;
                if (t == old_topic)
                {
                    nominator -= 1;
                }
                if (s == old_topic)
                {
                    denominator -= 1;
                }
                pi = nominator / denominator;
                rejection = rng_.rand_double();
                m = -(rejection < pi);
                s = (t & m) | (s & ~m);
            }
            // doc proposal
            double n_td_or_alpha = rng_.rand_double() *
                (doc->Size() + alpha_sum_);
            if (n_td_or_alpha < doc->Size())
            {
                int32_t t_idx = static_cast<int32_t>(n_td_or_alpha);
                t = doc->Topic(t_idx);
            }
            else
            {
                t = rng_.rand_k(num_topic_);
            }
            if (t != s)
            {
                n_tw_beta = word_topic_row.At(t) + beta_;
                n_sw_beta = word_topic_row.At(s) + beta_;
                n_t_beta_sum = summary_row.At(t) + beta_sum_;
                n_s_beta_sum = summary_row.At(s) + beta_sum_;

                if (t == old_topic)
                {
                    n_tw_beta -= subtractor_;
                    n_t_beta_sum -= subtractor_;
                }
                if (s == old_topic)
                {
                    n_sw_beta -= subtractor_;
                    n_s_beta_sum -= subtractor_;
                }
                
                nominator = n_tw_beta * n_s_beta_sum;
                denominator = n_sw_beta * n_t_beta_sum;
                pi = nominator / denominator;
                rejection = rng_.rand_double();
                m = -(rejection < pi);
                s = (t & m) | (s & ~m);
            }
        }
        return s;
    }
} // namespace lightlda
} // namespace multiverso
