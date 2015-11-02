/*!
 * \file data_stream.h
 * \brief Defines interface for data access
 */

#ifndef _LIGHTLDA_DATA_STREAM_H_
#define _LIGHTLDA_DATA_STREAM_H_

namespace multiverso { namespace lightlda 
{
    class DataBlock;
    /*! \brief interface of data stream */
    class IDataStream
    {
    public:
        virtual ~IDataStream() {}
        /*! \brief Should call this method before access a data block */
        virtual void BeforeDataAccess() = 0;
        /*! \brief Should call this method after access a data block */
        virtual void EndDataAccess() = 0;
        /*! 
         * \brief Get one data block 
         * \return reference to data block 
         */
        virtual DataBlock& CurrDataBlock() = 0;
    };

    /*! \brief Factory method to create data stream */
    IDataStream* CreateDataStream();
    
} // namespace lightlda
} // namespace multiverso

#endif // _LIGHTLDA_DATA_STREAM_
