#ifndef KMERHASH_HASHMAP_RADIXSORT_HPP_
#define KMERHASH_HASHMAP_RADIXSORT_HPP_
#include <stdlib.h>
#include <stdint.h>
//#include "immintrin.h"  // emm: _mm_stream_si64

#include <x86intrin.h>

#include <math.h>
#include <functional>
#ifdef VTUNE_ANALYSIS
#include <ittnotify.h>
#endif

//#include "MurmurHash3.h"
#include "math_utils.hpp"
#include "mem_utils.hpp"
#include "hash_new.hpp"

#include "iterators/transform_iterator.hpp"

namespace fsc {

template <class Key, class V, template <typename> class Hash = ::std::hash,
          template <typename> class Equal = ::std::equal_to,
          	typename Reducer = ::std::plus<V>
         >
class hashmap_radixsort {

protected:
    using t_bin_size = uint16_t;
    using t_bin_count = size_t;
    using t_bucket_count = size_t;   // if really careful, maybe can use 32 bit for the immediate test.
#if 1
    using t_buffer_size = uint16_t;  // could be bin_size * 2.
#else
    using t_buffer_size = uint32_t;  // could be bin_size ^ 2 / 2.
#endif
    using t_key_count = size_t;
public:
    using t_pfd_size = size_t;  // uint8_t may be enough


	static constexpr t_pfd_size PFD = 16;
    using key_type              = Key;
    using mapped_type           = V;
    using value_type            = ::std::pair<const Key, V>;
    using hasher                = Hash<Key>;
    using key_equal             = Equal<Key>;
    using size_type             = size_t;
    using difference_type       = ptrdiff_t;


    // for generating padding https://stackoverflow.com/questions/1239855/pad-a-c-structure-to-a-power-of-two
    // larger size reduced performance. turn off for now.
    // // template <class T1, class T2, class T3>
    // // struct PadSize
    // // {
    // //     enum { val = ::utils::mem::P<sizeof(T1) + sizeof(T2) + sizeof(T3) - 1>::val - sizeof(T1) - sizeof(T2) - sizeof(T3) }; 
    // // };
    // template <typename KK = Key, typename VV = V, typename ID = t_bucket_count,
    //  int N = ::utils::mem::P<sizeof(KK) + sizeof(VV) + sizeof(ID) - 1>::val - sizeof(KK) - sizeof(VV) - sizeof(ID)>
    // struct PossiblyPadded
    // {
    //     KK key;
    //     VV val;
    //     ID bucketId;
    //     char    pad[N]; 
    // };
    // template <typename KK, typename VV, typename ID>
    // struct PossiblyPadded<KK, VV, ID, 0>
    // {
    //     KK key;
    //     VV val;
    //     ID bucketId;
    // };

    // //using HashElement = PossiblyPadded<PadSize<Key, V, t_bucket_count>::val>;
    // using HashElement = 
    // PossiblyPadded<Key, V, t_bucket_count, 
    //     ::utils::mem::P<sizeof(Key) + sizeof(V) + sizeof(t_bucket_count) - 1>::val - sizeof(Key) - sizeof(V) - sizeof(t_bucket_count)>;

    struct HashElement
    {
        Key key;
        V val;
        t_bucket_count bucketId;
    } 

    template <typename KV>
    class IndexedRangesIterator :
        public std::iterator<
            std::random_access_iterator_tag,
            KV, std::ptrdiff_t>
    {
        protected:
            using Distance = std::ptrdiff_t;

            HashElement const * table;
            t_bin_size const * counts;

            t_bin_count numBins;
            t_bin_size binSize;

            t_bin_count binId;
            t_bin_size idx;   // id within a bin

        public:

            // contructor.  construct end iterator.
            IndexedRangesIterator(HashElement const * _table, t_bin_size const * _counts,
                t_bin_count num_bins, t_bin_size bin_size, t_bin_count bin_id = 0, t_bin_size _idx = 0) :
                table(_table), counts(_counts), numBins(num_bins), binSize(bin_size), binId(bin_id), idx(_idx) {}

            IndexedRangesIterator(IndexedRangesIterator const & other) : 
                IndexedRangesIterator(other.table, other.counts, other.numBins, other.binSize, other.binId, other.idx) {}
            IndexedRangesIterator(IndexedRangesIterator && other) : 
                IndexedRangesIterator(other.table, other.counts, other.numBins, other.binSize, other.binId, other.idx) {}
            // assignement operators
            IndexedRangesIterator& operator=(IndexedRangesIterator const & other) {
                table = other.table;
                counts = other.counts;
                numBins = other.numBins;
                binSize = other.binSize;
                binId = other.binId;
                idx = other.idx;
            }
            IndexedRangesIterator& operator=(IndexedRangesIterator && other) {
                table = other.table;
                counts = other.counts;
                numBins = other.numBins;
                binSize = other.binSize;
                binId = other.binId;
                idx = other.idx;
            }

            virtual ~IndexedRangesIterator() {}

            // increment and decrement operators
            IndexedRangesIterator& operator++() {
                if (binId >= numBins) return *this;
                // should always be dereferenceable, before incremented.
                ++idx;
                while (idx == counts[binId]) {
                    ++binId;
                    if (binId >= numBins) return *this;
                    idx = 0;
                }
                return *this;
            }

            IndexedRangesIterator operator++(int) const {
                IndexedRangesIterator it = *this;
                it.operator++();
                return it;
            }
            IndexedRangesIterator& operator--() {
                if (binId < 0) return *this;
                --idx;
                while (idx < 0) {
                    --binId;
                    if (binId < 0) return *this;
                    idx = counts[binId] - 1;
                }
                return *this;
            }

            IndexedRangesIterator operator--(int) const {
                IndexedRangesIterator it = *this;
                it.operator--();
                return it;
            }

            IndexedRangesIterator& operator+=(Distance const & n) {
                if (n < 0) return this->operator-=(-n);

                Distance dist;
                Distance nn = n;
                while (nn > 0) {
                    if (binId >= numBins) return *this;

                    dist = std::min(nn, static_cast<Distance>(counts[binId] - idx));
                    nn -= dist;
                    idx += dist;
                    if (idx == counts[binId]) {
                        ++binId;
                        idx = 0;
                    }
                }
            }

            IndexedRangesIterator operator+(Distance const & n) const {
                IndexedRangesIterator it = *this;
                it += n;
                return it;
            }

            IndexedRangesIterator& operator-=(Distance const & n) {
                if (n < 0) return this->operator+=(-n);

                Distance dist;
                Distance nn = n;
                while (nn > 0) {
                    if (binId < 0) return *this;

                    dist = std::min(nn, static_cast<Distance>(idx + 1));
                    nn -= dist;
                    idx -= dist;
                    if (idx < 0) {
                        --binId;
                        idx = counts[binId] - 1;
                    }
                }
            }

            IndexedRangesIterator operator-(Distance const & n) const {
                IndexedRangesIterator it = *this;
                it -= n;
                return it;
            }

            Distance operator-(IndexedRangesIterator const & other) const {
                if (binId == other.binId) return idx - other.idx;
                else if (binId > other.binId) return other.operator-(*this);
                else { // binId < other.binId
                    Distance dist = other.idx;
                    t_bin_size j = idx;
                    for (t_bin_count i = binId; i < other.binId; ++i) {
                        dist += counts[binId] - j;
                        j = 0;
                    }
                    return -dist;
                }
            }

            // comparison operators
            bool operator==(IndexedRangesIterator const & other) const {
                return (binId == other.binId) && (idx == other.idx);
            }
            bool operator!=(IndexedRangesIterator const & other) const {
                return (binId != other.binId) || (idx != other.idx);
            }

            bool operator<(IndexedRangesIterator const & other) const {
                return (binId == other.binId) ? (idx < other.idx) : (binId < other.binId);
            }
            bool operator>(IndexedRangesIterator const & other) const {
                return (binId == other.binId) ? (idx > other.idx) : (binId > other.binId);
            }
            bool operator>=(IndexedRangesIterator const & other) const {
                return !(binId < other.binId);
            }
            bool operator<=(IndexedRangesIterator const & other) const {
                return !(binId > other.binId);
            }

            // dereference operator
            KV operator[](Distance const & i) const {
                IndexedRangesIterator it(table, counts, 0, 0);
                return *(it + i);
            }

            KV operator*() const {
                HashElement const* it = table + binId * binSize + idx;
                return std::make_pair((*it).key, (*it).val);
            }
        
            friend IndexedRangesIterator operator+(std::ptrdiff_t n, IndexedRangesIterator const & right)
            {
                // reduced to + operator
                return right + n;
            }
    };

	using iterator              = IndexedRangesIterator<value_type>;
	using const_iterator        = IndexedRangesIterator<const value_type>;


#define COHERENT 0
#define INSERT 1
#define ERASE 2

    protected:
    t_bucket_count numBuckets;  // each bucket holds 2 elements.
    t_bucket_count bucketMask;

    t_bin_count numBins;   // numBuckets / (binsize/2)
    t_bin_count binMask;

    uint8_t binShift;   // binSize in bits
    t_bin_size binSize;    // limited to 2^16 or 2^15?  if 32K, then t_buffer_size can be 16 bit.

    t_bin_count overflowBufSize; 
    t_bin_count curOverflowBufId;

    t_buffer_size sortBufSize;   // binsize^2 / 2.  really only need binSize * 2.

    mutable V noValue;
    int8_t  coherence;
    int32_t seed;

    t_key_count totalKeyCount;   // max is numBin * binSize, or numBucket * 2.

    t_bin_size *countArray;   // count per bin.  at most binSize.
    HashElement *hashTable;
    HashElement *overflowBuf;
    HashElement *sortBuf;          // binSize^2/2 elements, so need at most 2*binSize elements (during merge)
    t_bin_size *countSortBuf;  // radixsort sorts at most binSize number of elements at a time.  each element in countSortBuf should hold 2*binSize, at most t_bin_size.
                                //  2*binSize during merge.

    t_bin_size *info_container;  // max value is countArray max val, so 2^16, which is max of binSize?

    Equal<Key> eq;
    Hash<Key> hash;
	hyperloglog64<Key, Hash<Key>, 12> hll;  // precision of 12bits  error rate : 1.04/(2^6)

	template <typename S>
	struct modulus2 {
		static constexpr size_t batch_size = 1;

		S mask;

		modulus2(S const & _mask, int) :	mask(_mask) {}  // dummy

		template <typename IN>
		inline IN operator()(IN const & x) const { return (x & mask); }

	};

	// mod 2 okay since hashtable size is always power of 2.t_buffer_size
	using InternalHash = ::fsc::hash::TransformedHash<Key, Hash, ::bliss::transform::identity, modulus2>;
	using hash_val_type = typename InternalHash::HASH_VAL_TYPE;
    InternalHash hash_mod2; 

    Reducer reduc;

    template <typename R = Reducer, typename VV = V,
        typename std::enable_if<!::std::is_same<R, std::plus<VV> >::value, int>::type = 0>
    t_bin_size radixSort(HashElement *A,
                  t_bin_size size)
    {
        HashElement *sortBuf = this->sortBuf;
        t_bin_size *countBuf = this->countSortBuf;
        //int32_t shift = this->binShift;
        t_buffer_size bufSize = this->sortBufSize;
        //int32_t binSize = this->binSize;
#if 1
        if(size <= 1) return size;
        t_buffer_size mask = bufSize - 1;
        memset(countBuf, 0, bufSize * sizeof(t_bin_size));
        t_bin_size i;
        for(i = 0; i < size; i++)
        {
            HashElement he = A[i];
            t_buffer_size id = he.bucketId & mask;
            //printf("%d] bucketId = %d, id = %d\n", i, he.bucketId, id);
            countBuf[id]++;
        }
        t_bin_size cumulSum = 0;
        for(t_buffer_size j = 0; j < bufSize; j++)
        {
            t_bin_size c = countBuf[j];
            countBuf[j] = cumulSum;
            //printf("%d] count = %d, cumulSum = %d\n", i, c, cumulSum);
            cumulSum += c;
        }

        for(i = 0; i < size; i++)
        {
            HashElement he = A[i];
            t_buffer_size id = he.bucketId & mask;
            t_bin_size pos = countBuf[id];
            // printf("%d] bucketId = %u, id = %d, pos = %d, sortBuf size = %d, numBuckets %u, numBins %d, binSize %d\n", i, he.bucketId, id, pos, bufSize, numBuckets, numBins, binSize);
            sortBuf[pos] = he;
            countBuf[id]++;
        }

#ifndef NDEBUG
        for(i = 1; i < size; i++)
        {
            if(sortBuf[i].bucketId < sortBuf[i - 1].bucketId)
            {
                printf("ERROR! %d] %u, %u\n", i, sortBuf[i - 1].bucketId, sortBuf[i].bucketId);
                exit(0);
            }
        }
#endif

        t_bucket_count curBid = sortBuf[0].bucketId;
        t_bin_size curStart = 0;
        A[0] = sortBuf[0];
        t_bin_size count = 1;
        for(i = 1; i < size; i++)
        {
            t_bucket_count bid = sortBuf[i].bucketId;
            Key key = sortBuf[i].key;
            if(bid == curBid)
            {
                t_bin_size j;
                for(j = curStart; j < count; j++)
                {
                    if(eq(key, A[j].key))
                    {
//                        A[j].val++;
                        A[j].val = reduc(A[j].val, sortBuf[i].val);
                        break;
                    }
                }
                if(j == count)
                {
                    A[count] = sortBuf[i];
                    count++;
                }
            }
            else
            {
                A[count] = sortBuf[i];
                curBid = bid;
                curStart = count;
                count++;
            }

        }
        return count;
#else
        if(size <= 1) return size;
        int32_t mask = bufSize - 1;

        memset(countBuf, 0, bufSize * sizeof(uint8_t));
        int i, j;
        for(i = 0; i < size; i++)
        {
            HashElement he = A[i];
            uint32_t id = he.bucketId & mask;
            int32_t count = countBuf[id];
            for(j = 0; j < count; j++)
            {
                HashElement he2 = sortBuf[id * bucketSize + j];
                if(he2.key == he.key)
                {
                    sortBuf[id * bucketSize + j].val++;
                    break;
                }
            }
            if(j == count)
            {
                sortBuf[id * bucketSize + count] = he;
                countBuf[id]++;
            }
        }

        int count = 0;
        for(i = 0; i < bufSize; i++)
        {
            int c = countBuf[i];
            for(j = 0; j < c; j++)
                A[count++] = sortBuf[i * bucketSize + j];
        }
        return count;
#endif
    }


    template <typename R = Reducer, typename VV = V,
        typename std::enable_if<::std::is_same<R, std::plus<VV> >::value, int>::type = 0>
    t_bin_size radixSort(HashElement *A,
                  t_bin_size size)
    {
        HashElement *sortBuf = this->sortBuf;
        t_bin_size *countBuf = this->countSortBuf;
        //int32_t shift = this->binShift;
        t_buffer_size bufSize = this->sortBufSize;
        //int32_t binSize = this->binSize;
#if 1
        if(size <= 1) return size;
        t_buffer_size mask = bufSize - 1;
        memset(countBuf, 0, bufSize * sizeof(t_bin_size));
        t_bin_size i;
        for(i = 0; i < size; i++)
        {
            HashElement he = A[i];
            t_buffer_size id = he.bucketId & mask;
            //printf("%d] bucketId = %d, id = %d\n", i, he.bucketId, id);t_buffer_size
            countBuf[id]++;
        }
        t_bin_size cumulSum = 0;
        for(t_buffer_size j = 0; j < bufSize; j++)
        {
            t_bin_size c = countBuf[j];
            countBuf[j] = cumulSum;
            //printf("%d] count = %d, cumulSum = %d\n", i, c, cumulSum);
            cumulSum += c;
        }

        for(i = 0; i < size; i++)
        {
            HashElement he = A[i];
            t_buffer_size id = he.bucketId & mask;
            t_buffer_size pos = countBuf[id];
            // printf("%d] bucketId = %u, id = %d, pos = %d, sortBuf size = %d, numBuckets %u, numBins %d, binSize %d\n", i, he.bucketId, id, pos, bufSize, numBuckets, numBins, binSize);
            sortBuf[pos] = he;
            countBuf[id]++;
        }

#ifndef NDEBUG
        for(i = 1; i < size; i++)
        {
            if(sortBuf[i].bucketId < sortBuf[i - 1].bucketId)
            {
                printf("ERROR! %d] %u, %u\n", i, sortBuf[i - 1].bucketId, sortBuf[i].bucketId);
                exit(0);
            }
        }
#endif

        t_bucket_count curBid = sortBuf[0].bucketId;
        t_bin_size curStart = 0;
        A[0] = sortBuf[0];
        t_bin_size count = 1;
        for(i = 1; i < size; i++)
        {
            t_bucket_count bid = sortBuf[i].bucketId;
            Key key = sortBuf[i].key;
            if(bid == curBid)
            {
                t_bin_size j;
                for(j = curStart; j < count; j++)
                {
                    if(eq(key, A[j].key))
                    {
                        A[j].val++;
                        break;
                    }
                }
                if(j == count)
                {
                    A[count] = sortBuf[i];
                    count++;
                }
            }
            else
            {
                A[count] = sortBuf[i];
                curBid = bid;
                curStart = count;
                count++;
            }

        }
        return count;
#else
        if(size <= 1) return size;
        int32_t mask = bufSize - 1;

        memset(countBuf, 0, bufSize * sizeof(uint8_t));
        int i, j;
        for(i = 0; i < size; i++)
        {
            HashElement he = A[i];
            uint32_t id = he.bucketId & mask;
            int32_t count = countBuf[id];
            for(j = 0; j < count; j++)
            {
                HashElement he2 = sortBuf[id * bucketSize + j];
                if(he2.key == he.key)
                {
                    sortBuf[id * bucketSize + j].val++;
                    break;
                }
            }
            if(j == count)
            {
                sortBuf[id * bucketSize + count] = he;
                countBuf[id]++;
            }
        }

        int count = 0;
        for(i = 0; i < bufSize; i++)
        {
            int c = countBuf[i];
            for(j = 0; j < c; j++)
                A[count++] = sortBuf[i * bucketSize + j];
        }
        return count;
#endif
    }

    template <typename R = Reducer>
    t_bin_size merge(HashElement *A, t_bin_size sizeA, HashElement *B, t_bin_size sizeB)
    {

        //printf("sizeA = %d, sizeB = %d\n", sizeA, sizeB);
        t_bin_size pA, pB;

        pA = pB = 0;

        t_buffer_size count = 0;

        while((pA < sizeA) && (pB < sizeB))
        {
            t_bucket_count bidA = A[pA].bucketId;
            t_bucket_count bidB = B[pB].bucketId;
            //printf("pA = %d, pB = %d, bidA = %d, bidB = %d\n", pA, pB, bidA, bidB);
            if(bidA <= bidB)
            {
                sortBuf[count] = A[pA];
                pA++;
                count++;
            }
            else
            {
                sortBuf[count] = B[pB];
                pB++;
                count++;
            }
        }
        while(pA < sizeA)
        {
            sortBuf[count] = A[pA];
            pA++;
            count++;
        }
        while(pB < sizeB)
        {
            sortBuf[count] = B[pB];
            pB++;
            count++;
        }

        //printf("count = %d, pA = %d, pB = %d\n", count, pA, pB);
        t_buffer_size i;
        t_bucket_count curBid = sortBuf[0].bucketId;
        t_buffer_size curStart = 0;
        t_buffer_size size = count;
        count = 1;
        HashElement *newBuf = (HashElement *)_mm_malloc(size * sizeof(HashElement), 64);
        newBuf[0] = sortBuf[0];
        for(i = 1; i < size; i++)
        {
            t_bucket_count bid = sortBuf[i].bucketId;
            Key key = sortBuf[i].key;
            if(bid == curBid)
            {
                t_buffer_size j;
                for(j = curStart; j < count; j++)
                {
                    if(eq(key, newBuf[j].key))
                    {
//                        newBuf[j].val++;
                        newBuf[j].val = reduc(newBuf[j].val, sortBuf[i].val);
                        break;
                    }
                }
                if(j == count)
                {
                    newBuf[count] = sortBuf[i];
                    count++;
                }
            }
            else
            {
                newBuf[count] = sortBuf[i];
                curBid = bid;
                curStart = count;
                count++;
            }

        }
#ifndef NDEBUG
        for(i = 1; i < count; i++)
        {
            if(newBuf[i].bucketId < newBuf[i - 1].bucketId)
            {
                printf("ERROR! %d] %u, %u\n", i, newBuf[i - 1].bucketId, newBuf[i].bucketId);
                exit(0);
            }
        }
#endif
        for(i = 0; i < sizeA; i++)
            A[i] = newBuf[i];

        for(; i < count; i++)
            B[i - sizeA] = newBuf[i];

		_mm_free(newBuf);
        return count;
    }


    inline HashElement *find_internal(Key key, t_bucket_count bucketId) const
    {

        t_bin_count binId = bucketId >> binShift;
        t_bin_size start = info_container[bucketId];
        t_bin_size end;

        if (bucketId == std::numeric_limits<t_bucket_count>::max()) {
            end = countArray[binId];
        } else {
            t_bin_count binId2 = (bucketId + 1) >> binShift;
            if(binId == binId2)
                end = info_container[bucketId + 1];
            else
                end = countArray[binId];
        }

        t_bin_size j;
        HashElement *he;
        for(j = start; j < end; j++)
        {
            if(j < (binSize - 1))
            {
                he = hashTable + static_cast<size_t>(binId) * binSize + j;
            }
            else
            {
                t_bucket_count overflowBufId = hashTable[static_cast<size_t>(binId) * binSize + binSize - 1].bucketId;
                he = overflowBuf + static_cast<size_t>(overflowBufId) * binSize + j - (binSize - 1);
            }
            if(eq(key, he->key))
            {
                return he;
            }
        }
        return NULL;
    }

    public:
    hashmap_radixsort(t_bucket_count _numBuckets = 1048576,
            t_bin_size _binSize = 4096,
            V _noValue = 0) :
            	numBuckets(next_power_of_2(_numBuckets)),
				bucketMask(numBuckets - 1),
				totalKeyCount(0),
				hash_mod2(hash, ::bliss::transform::identity<Key>(), modulus2<hash_val_type>(bucketMask, 0))
    {
	
        binSize = next_power_of_2(_binSize);

        if (binSize > (std::numeric_limits<t_bin_size>::max() >> 1) )
            std::invalid_argument("ERROR: constructor.  binSize is limited to log(sizeof(t_bin_size))-1 bit..");

        numBins = std::max(static_cast<t_bin_count>(1), 
                    numBuckets / std::max(static_cast<t_bin_size>(1), static_cast<t_bin_size>(binSize >> 1)));  // both powers of 2.  this is more safe...
        binMask = numBins - 1;
        overflowBufSize = std::max(static_cast<t_bin_count>(1), numBins >> 3);
        noValue = _noValue;

        countArray = (t_bin_size *)_mm_malloc(numBins * sizeof(t_bin_size), 64);
        memset(countArray, 0, numBins * sizeof(t_bin_size));

        hashTable = (HashElement *)_mm_malloc(numBins * binSize * sizeof(HashElement), 64);
        overflowBuf = (HashElement *)_mm_malloc(overflowBufSize * binSize * sizeof(HashElement), 64);

        sortBufSize = numBuckets / numBins;  // == binSize / 2
#if 1
        sortBuf = (HashElement *)_mm_malloc(2 * binSize * sizeof(HashElement), 64);  // binSize ^2 in size
#else
        sortBuf = (HashElement *)_mm_malloc(sortBufSize * binSize * sizeof(HashElement), 64);  // binSize ^2 in size
#endif
        countSortBuf = (t_buffer_size *)_mm_malloc(sortBufSize * sizeof(t_buffer_size), 64);

        binShift = log2(sortBufSize);

        info_container = (t_bin_size *)_mm_malloc(numBuckets * sizeof(t_bin_size), 64);
        memset(info_container, 0, numBuckets * sizeof(t_bin_size));

        curOverflowBufId = 0;
#ifndef NDEBUG
        printf("numBuckets = %u, numBins = %d, binSize = %d, overflowBufSize = %d, sortBufSize = %d, binShift = %d\n",
                numBuckets, numBins, binSize, overflowBufSize, sortBufSize, binShift);
#endif
        coherence = COHERENT;
        seed = 43;
    }

	~hashmap_radixsort()
	{
		_mm_free(countArray);
		_mm_free(hashTable);
		_mm_free(overflowBuf);
		_mm_free(sortBuf);
		_mm_free(countSortBuf);
		_mm_free(info_container);
	}

    hashmap_radixsort(hashmap_radixsort const & other) : 
        numBuckets(other.numBuckets),
        bucketMask(other.bucketMask),
        numBins(other.numBins),
        binMask(other.binMask),
        binShift(other.binShift),
        binSize(other.binSize),
        overflowBufSize(other.overflowBufSize),
        curOverflowBufId(other.curOverflowBufId),
        sortBufSize(other.sortBufSize),
        noValue(other.noValue),
        coherence(other.coherence),
        seed(other.seed),
        totalKeyCount(other.totalKeyCount),
        eq(other.eq),
        hash(other.hash),
        hll(other.hll),
        hash_mod2(other.hash_mod2)
    {
        countArray = (t_bin_size *)_mm_malloc(numBins * sizeof(t_bin_size), 64);
        memcpy(countArray, other.countArray, numBins * sizeof(t_bin_size));

        hashTable = (HashElement *)_mm_malloc(numBins * binSize * sizeof(HashElement), 64);
        memcpy(hashTable, other.hashTable, numBins * binSize * sizeof(HashElement));
        overflowBuf = (HashElement *)_mm_malloc(overflowBufSize * binSize * sizeof(HashElement), 64);
        memcpy(overflowBuf, other.overflowBuf, overflowBufSize * binSize * sizeof(HashElement));
#if 1
        sortBuf = (HashElement *)_mm_malloc(2 * binSize * sizeof(HashElement), 64);  // binSize ^2 in size
        memcpy(sortBuf, other.sortBuf, 2 * binSize * sizeof(HashElement));
#else
        sortBuf = (HashElement *)_mm_malloc(sortBufSize * binSize * sizeof(HashElement), 64);  // binSize ^2 in size
        memcpy(sortBuf, other.sortBuf, sortBufSize * binSize * sizeof(HashElement));
#endif

        countSortBuf = (t_buffer_size *)_mm_malloc(sortBufSize * sizeof(t_buffer_size), 64);
        memcpy(countSortBuf, other.countSortBuf, sortBufSize * sizeof(t_buffer_size));

        info_container = (t_bin_size *)_mm_malloc(numBuckets * sizeof(t_bin_size), 64);
        memcpy(info_container, other.info_container, numBuckets * sizeof(t_bin_size));
    }
    hashmap_radixsort(hashmap_radixsort && other) : 
        numBuckets(other.numBuckets),
        bucketMask(other.bucketMask),
        numBins(other.numBins),
        binMask(other.binMask),
        binShift(other.binShift),
        binSize(other.binSize),
        overflowBufSize(other.overflowBufSize),
        curOverflowBufId(other.curOverflowBufId),
        sortBufSize(other.sortBufSize),
        noValue(other.noValue),
        coherence(other.coherence),
        seed(other.seed),
        totalKeyCount(other.totalKeyCount),
        eq(std::move(other.eq)),
        hash(std::move(other.hash)),
        hash_mod2(std::move(other.hash_mod2))
    {
        hll.swap(other.hll),
        std::swap(countArray, other.countArray);
        std::swap(hashTable, other.hashTable);
        std::swap(overflowBuf, other.overflowBuf);
        std::swap(sortBuf, other.sortBuf);
        std::swap(countSortBuf, other.countSortBuf);
        std::swap(info_container, other.info_container);
    }
    hashmap_radixsort & operator=(hashmap_radixsort const & other) {
        numBuckets = other.numBuckets;
        bucketMask = other.bucketMask;
        numBins = other.numBins;
        binMask = other.binMask;
        binShift = other.binShift;
        binSize = other.binSize;
        overflowBufSize = other.overflowBufSize;
        curOverflowBufId = other.curOverflowBufId;
        sortBufSize = other.sortBufSize;
        noValue = other.noValue;
        coherence = other.coherence;
        seed = other.seed;
        totalKeyCount = other.totalKeyCount;

        eq = other.eq;
        hash = other.hash;
        hll = other.hll;
        hash_mod2 = other.hash_mod2;

        _mm_free(countArray);
        countArray = (t_bin_size *)_mm_malloc(numBins * sizeof(t_bin_size), 64);
        memcpy(countArray, other.countArray, numBins * sizeof(t_bin_size));
        _mm_free(hashTable);
        hashTable = (HashElement *)_mm_malloc(numBins * binSize * sizeof(HashElement), 64);
        memcpy(hashTable, other.hashTable, numBins * binSize * sizeof(HashElement));
        _mm_free(overflowBuf);
        overflowBuf = (HashElement *)_mm_malloc(overflowBufSize * binSize * sizeof(HashElement), 64);
        memcpy(overflowBuf, other.overflowBuf, overflowBufSize * binSize * sizeof(HashElement));
        _mm_free(sortBuf);
#if 1
        sortBuf = (HashElement *)_mm_malloc(2 * binSize * sizeof(HashElement), 64);
        memcpy(sortBuf, other.sortBuf, 2 * binSize * sizeof(HashElement));
#else 
        sortBuf = (HashElement *)_mm_malloc(sortBufSize * binSize * sizeof(HashElement), 64);
        memcpy(sortBuf, other.sortBuf, sortBufSize * binSize * sizeof(HashElement));
#endif
        _mm_free(countSortBuf);
        countSortBuf = (t_buffer_size *)_mm_malloc(sortBufSize * sizeof(t_buffer_size), 64);
        memcpy(countSortBuf, other.countSortBuf, sortBufSize * sizeof(t_buffer_size));
        _mm_free(info_container);
        info_container = (t_bin_size *)_mm_malloc(numBuckets * sizeof(t_bin_size), 64);
        memcpy(info_container, other.info_container, numBuckets * sizeof(t_bin_size));

        return *this;
    }
    hashmap_radixsort & operator=(hashmap_radixsort && other) {
        numBuckets = other.numBuckets;
        bucketMask = other.bucketMask;
        numBins = other.numBins;
        binMask = other.binMask;
        binShift = other.binShift;
        binSize = other.binSize;
        overflowBufSize = other.overflowBufSize;
        curOverflowBufId = other.curOverflowBufId;
        sortBufSize = other.sortBufSize;
        noValue = other.noValue;
        coherence = other.coherence;
        seed = other.seed;
        totalKeyCount = other.totalKeyCount;

        eq = std::move(other.eq);
        hash = std::move(other.hash);
        hll.swap(other.hll);
        hash_mod2 = std::move(other.hash_mod2);

        std::swap(countArray, other.countArray);
        std::swap(hashTable, other.hashTable);
        std::swap(overflowBuf, other.overflowBuf);
        std::swap(sortBuf, other.sortBuf);
        std::swap(countSortBuf, other.countSortBuf);
        std::swap(info_container, other.info_container);

        return *this;
    }


    void swap(hashmap_radixsort && other) {
        std::swap(numBuckets, other.numBuckets);
        std::swap(bucketMask, other.bucketMask);
        std::swap(numBins   , other.numBins);
        std::swap(binMask   , other.binMask);
        std::swap(binShift  , other.binShift);
        std::swap(binSize   , other.binSize);
        std::swap(overflowBufSize, other.overflowBufSize);
        std::swap(curOverflowBufId, other.curOverflowBufId);
        std::swap(sortBufSize, other.sortBufSize);
        std::swap(noValue, other.noValue);
        std::swap(coherence, other.coherence);
        std::swap(seed, other.seed);
        std::swap(totalKeyCount, other.totalKeyCount);

        std::swap(eq, other.eq);
        std::swap(hash, other.hash);
        hll.swap(std::move(other.hll));
        std::swap(hash_mod2, other.hash_mod2);

        std::swap(countArray, other.countArray);
        std::swap(hashTable, other.hashTable);
        std::swap(overflowBuf, other.overflowBuf);
        std::swap(sortBuf, other.sortBuf);
        std::swap(countSortBuf, other.countSortBuf);
        std::swap(info_container, other.info_container);
    }



	void set_novalue(V _noValue) const { noValue = _noValue; }

	const_iterator cbegin() const {
		return const_iterator(hashTable, countArray, numBins, binSize, 0, 0);
	}

	const_iterator cend() const {
		return const_iterator(hashTable, countArray, numBins, binSize, numBins, 0);
	}


	void reserve(t_key_count _newElementCount) {
		resize(next_power_of_2(_newElementCount));
	}

	// return fail or success
	bool resize(t_bucket_count _newNumBuckets )
	{
		if (next_power_of_2(_newNumBuckets) == numBuckets) return true;

#ifndef NDEBUG
        int64_t preStart = __rdtsc();
#endif
		if(coherence == INSERT)
		{
//			printf("WARNING! The hashtable is in INSERT mode at the moment. finalize_insert will be called \n");
			finalize_insert();
		} else if (coherence == ERASE) {
//		  printf("WARNING! The hashtable is in ERASE mode at the moment. finalize_erase will be called \n");
		  finalize_erase();
		}
#ifndef NDEBUG
        int64_t preTicks = __rdtsc() - preStart;
        int64_t initStart = __rdtsc();
#endif

       // shortcutting, if starting out with empty.
       if (totalKeyCount == 0) {
            resize_alloc(_newNumBuckets, binSize);
            return true;
       }


//		::std::cout << "RESIZING:  size=" << totalKeyCount
//				<< " prev capacity=" << capacity()
//				<< " new capacity=" << _newNumBuckets << std::endl;
		//::std::cout << "key count " << totalKeyCount << " elem size " << sizeof(HashElement) << std::endl;
        std::pair<Key, V> *keyArray = (std::pair<Key, V> *)_mm_malloc((totalKeyCount + PFD) * sizeof(std::pair<Key, V>), 64);
		//printf("CURR numBuckets = %u, numBins = %d, binSize = %d, overflowBufSize = %d, sortBufSize = %d, binShift = %d\n",
		//        numBuckets, numBins, binSize, overflowBufSize, sortBufSize, binShift);

		t_bin_count i;
        t_bin_size j;
		t_key_count elemCount = 0;
		for(i = 0; i < numBins; i++)
		{
			t_bin_size count = countArray[i];
			t_bin_size y = std::min(count, static_cast<t_bin_size>(binSize - 1));

			for(j = 0; j < y; j++)
			{
				keyArray[elemCount++].first = hashTable[i * binSize + j].key;
				keyArray[elemCount++].second = hashTable[i * binSize + j].val;
			}
            t_bin_count overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                keyArray[elemCount++].first = overflowBuf[overflowBufId * binSize + j - (binSize - 1)].key;
                keyArray[elemCount++].second = overflowBuf[overflowBufId * binSize + j - (binSize - 1)].val;
            }
		}
#ifndef NDEBUG
        int64_t initTicks = __rdtsc() - initStart;
		printf("elemCount = %ld\n", elemCount);

        int64_t allocStart = __rdtsc();
#endif
		while (elemCount > _newNumBuckets)  _newNumBuckets <<= 1;
		//ssstd::cout << "before " << numBuckets << std::endl;
		resize_alloc(_newNumBuckets, binSize);
#ifndef NDEBUG
        int64_t allocTicks = __rdtsc() - allocStart;
        int64_t insertStart = __rdtsc();
#endif
		//std::cout << "allocated " << _newNumBuckets << std::endl;
		int resize_result = resize_insert(keyArray, elemCount);
		//std::cout << "inserted " << std::endl;
#ifndef NDEBUG
        int64_t insertTicks = __rdtsc() - insertStart;

        
        int64_t whileStart = __rdtsc();
#endif
		int tries = 2;
		while ( (resize_result > 0) && (tries > 0)) {
			std::cout << "resizing again and reinserting " << std::endl;

			if (resize_result == 1) {
			  // failed resize.  happens when binSize is not big enough, or overflow is not big enough.
				resize_alloc(_newNumBuckets << 1, binSize);
				std::cout << "try to resize increasing buckets" << std::endl;
			} else if (resize_result == 2) {
				resize_alloc(_newNumBuckets, binSize << 1);
				std::cout << "try to resize increasing binsize" << std::endl;
			}

			--tries;
			resize_result = resize_insert(keyArray, elemCount);  // double binSize, which returns to old number of bin.
		}
        _mm_free(keyArray);
#ifndef NDEBUG
        int64_t whileTicks = __rdtsc() - whileStart;

        int64_t finalizeStart = __rdtsc();
#endif

    if (resize_result == 0) resize_finalize_insert();
		else throw std::logic_error("ERROR: failed to resize, binSize doubled and still failed.");

#ifndef NDEBUG
        int64_t finalizeTicks = __rdtsc() - finalizeStart;

        int myrank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        printf("%d] %ld, %ld, %ld, %ld, %ld, %ld\n", myrank, preTicks, initTicks, allocTicks, insertTicks, whileTicks, finalizeTicks);
#endif
		return true;
	}

	  // return fail or success
	  void resize_alloc(t_bucket_count _newNumBuckets, t_bin_size _binSize)
	  {
        numBuckets = next_power_of_2(_newNumBuckets);
        bucketMask = numBuckets - 1;
	
        if (_binSize > (std::numeric_limits<t_bin_size>::max() >> 1))
            throw std::invalid_argument("ERROR: resize_alloc: _binSize exceeded max.");
            
        binSize = _binSize;    // allow bin size resize.  this is for new one...

        numBins = std::max(static_cast<t_bin_count>(1), 
                    numBuckets / std::max(static_cast<t_bin_size>(1), static_cast<t_bin_size>(binSize >> 1)));  // both powers of 2.  this is more safe...
        binMask = numBins - 1;
        overflowBufSize = std::max(static_cast<t_bin_count>(1), numBins >> 3);

    _mm_free(countArray);
        countArray = (t_bin_size *)_mm_malloc(numBins * sizeof(t_bin_size), 64);
        memset(countArray, 0, numBins * sizeof(t_bin_size));

    _mm_free(hashTable);
        hashTable = (HashElement *)_mm_malloc(numBins * binSize * sizeof(HashElement), 64);

    _mm_free(overflowBuf);
        overflowBuf = (HashElement *)_mm_malloc(overflowBufSize * binSize * sizeof(HashElement), 64);
        sortBufSize = numBuckets / numBins;

    _mm_free(sortBuf);
#if 1
        sortBuf = (HashElement *)_mm_malloc(2 * binSize * sizeof(HashElement), 64);  // binSize ^2 in size
#else
        sortBuf = (HashElement *)_mm_malloc(sortBufSize * binSize * sizeof(HashElement), 64);  // binSize ^2 in size
#endif

    _mm_free(countSortBuf);
        countSortBuf = (t_buffer_size *)_mm_malloc(sortBufSize * sizeof(t_buffer_size), 64);
        binShift = log2(sortBufSize);

    _mm_free(info_container);
        info_container = (t_bin_size *)_mm_malloc(numBuckets * sizeof(t_bin_size), 64);
        memset(info_container, 0, numBuckets * sizeof(t_bin_size));
        curOverflowBufId = 0;
#ifndef NDEBUG
        printf("numBuckets = %u, numBins = %d, binSize = %d, overflowBufSize = %d, sortBufSize = %d, binShift = %d\n",
                numBuckets, numBins, binSize, overflowBufSize, sortBufSize, binShift);
#endif

        hash_mod2.posttrans.mask = bucketMask;

	  }

        // return fail or success
        template <typename T>
        int resize_insert(T * keyArray, t_key_count numKeys)
        {
            t_key_count i;
    // insert the elements again
        coherence = INSERT;

        
		int32_t hash_batch_size = 512;
        hash_val_type bucketIdArray[2 * hash_batch_size];
		memset(bucketIdArray, 0, 2 * hash_batch_size * sizeof(hash_val_type));
		t_key_count hash_mask = 2 * hash_batch_size - 1;

		t_key_count hash_count = std::min(numKeys, static_cast<t_key_count>(hash_batch_size));
        hash_mod2(keyArray, hash_count, bucketIdArray);

        for(i = 0; i < numKeys; i += hash_batch_size)
        {
			t_key_count hash_first = i + hash_batch_size;
			if(hash_first > numKeys) hash_first = numKeys;
			t_key_count hash_last = i + 2 * hash_batch_size;
			if(hash_last > numKeys) hash_last = numKeys;
			t_key_count hash_count = hash_last - hash_first;
			hash_mod2(keyArray + hash_first, hash_count, bucketIdArray + ((hash_first) & hash_mask));
			t_key_count last = i + hash_batch_size;
			if(last > numKeys) last = numKeys;
			t_key_count j;
			for(j = i; j < last; j++)
			{
				HashElement he;
				// he.key = keyArray[j];
				// he.val = 1;
                init_hash_element(he, keyArray[j]);
                he.bucketId = bucketIdArray[j & hash_mask];
				t_bucket_count binId = he.bucketId >> binShift;
				t_bin_size count = countArray[binId];
				if(count < binSize)
				{
					if(count == (binSize - 1))
					{
						if(curOverflowBufId == overflowBufSize)
						{
							printf("ERROR! Ran out of overflowBuf, curOverflowBufId = %d.\n"
									"Try increasing numBins, binSize or overflowBufSize\n", 
									curOverflowBufId);
							return 1;
						}
						t_bucket_count overflowBufId = curOverflowBufId;
						printf("Have to use overflow buf, overflowBufId = %d\n", overflowBufId);
						curOverflowBufId++;
						hashTable[binId * binSize + binSize - 1].bucketId = overflowBufId;
						overflowBuf[overflowBufId * binSize] = he;
					}
					else
					{
						hashTable[binId * binSize + count] = he;                
					}

					countArray[binId]++;
				}
				else
				{
					t_bucket_count overflowBufId;
					overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
					if(count == (2 * binSize - 1))
					{
						printf("ERROR! binId = %ld, count = 2 * binSize - 1. Please use larger binSize or numBins\n", binId);
						return 2;
					}
					overflowBuf[overflowBufId * binSize + count - (binSize - 1)] = he;
					countArray[binId]++;
				}
			}
        }

    return 0;
  }

    void resize_finalize_insert()
    {
        if(coherence != INSERT)
        {
//            printf("ERROR! The hashtable coherence is not set to INSERT at the moment. finalize_insert() can not be serviced\n");
            return;
        }
        t_bin_count i;
        t_bin_size j;
        totalKeyCount = 0;
        for(i = 0; i < numBins; i++)
        {
            t_bin_size count = countArray[i];
            t_bucket_count firstBucketId = i << binShift;
            t_bucket_count lastBucketId = firstBucketId + sortBufSize - 1;
            t_bucket_count prevBucketId = firstBucketId - 1;
            t_bucket_count k;
            t_bin_size y = std::min(count, static_cast<t_bin_size>(binSize - 1));
            for(j = 0; j < y; j++)
            {
                t_bucket_count bucketId = hashTable[i * binSize + j].bucketId;
                if(bucketId != prevBucketId)
                {
                    for(k = prevBucketId; k < bucketId; k++)
                        info_container[k + 1] = j;
                    prevBucketId = bucketId;
                }
            }
            t_bucket_count overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                t_bucket_count bucketId = overflowBuf[overflowBufId * binSize + j - (binSize - 1)].bucketId;
                if(bucketId != prevBucketId)
                {
                    for(k = prevBucketId; k < bucketId; k++)
                        info_container[k + 1] = j;
                    prevBucketId = bucketId;
                }
            }
            for(k = prevBucketId; k < lastBucketId; k++)
                info_container[k + 1] = count;
            countArray[i] = count;
            totalKeyCount += count;
        }
        coherence = COHERENT;
    }

    inline void init_hash_element(HashElement & he, Key const & x) {
        he.key = x;
        he.val = 1;
    }
    inline void init_hash_element(HashElement & he, std::pair<Key, V> const & x) {
        he.key = x.first;
        he.val = x.second;
    }
    inline void init_hash_element(HashElement & he, std::pair<const Key, V> const & x) {
        he.key = x.first;
        he.val = x.second;
    }

    //uint64_t hash(uint64_t x)
    //{
    //    uint64_t y[2];
    //    MurmurHash3_x64_128(&x, sizeof(uint64_t), seed, y);
    //    return y[0];
    //}
	/// return number of successful inserts
    template < typename T > 
    t_key_count insert_impl(T *keyArray, t_key_count numKeys)
    {
        if((coherence != COHERENT) && (coherence != INSERT))
        {
//            printf("ERROR! The hashtable is not coherent at the moment. insert() can not be serviced\n");
            return 0;
        }
        t_key_count i;
        coherence = INSERT;
		int32_t hash_batch_size = 512;
        hash_val_type bucketIdArray[2 * hash_batch_size];
		memset(bucketIdArray, 0, 2 * hash_batch_size * sizeof(hash_val_type));
		t_key_count hash_mask = 2 * hash_batch_size - 1;
        //int64_t hashTicks = 0;
        //int64_t startTick, endTick;
        //startTick = __rdtsc();

		t_key_count hash_count = std::min(numKeys, static_cast<t_key_count>(hash_batch_size));
        hash_mod2(keyArray, hash_count, bucketIdArray);
//        for(i = 0; i < PFD; i++)
//        {
//            bucketIdArray[i] = hash(keyArray[i]) & bucketMask;
//        }
        //endTick = __rdtsc();
        //hashTicks += (endTick - startTick);

        for(i = 0; i < numKeys; i += hash_batch_size)
        {
            //startTick = __rdtsc();
			t_key_count hash_first = i + hash_batch_size;
			if(hash_first > numKeys) hash_first = numKeys;
			t_key_count hash_last = i + 2 * hash_batch_size;
			if(hash_last > numKeys) hash_last = numKeys;
			t_key_count hash_count = hash_last - hash_first;
			hash_mod2(keyArray + hash_first, hash_count, bucketIdArray + ((hash_first) & hash_mask));
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
			t_key_count last = i + hash_batch_size;
			if(last > numKeys) last = numKeys;
			t_key_count j;
			for(j = i; j < last; j++)
			{
				HashElement he;
				// he.key = keyArray[j];
				// he.val = 1;
                init_hash_element(he, keyArray[j]);
				he.bucketId = bucketIdArray[j & hash_mask];
				t_bin_count binId = he.bucketId >> binShift;
				t_bin_size count = countArray[binId];
#if ENABLE_PREFETCH
				t_bucket_count f_bucketId = bucketIdArray[(j + PFD) & hash_mask]; // = (hash(keyArray[i + PFD]) & bucketMask);
				t_bin_count f_binId = f_bucketId >> binShift;
				t_bin_size f_count = countArray[f_binId];
				_mm_prefetch((const char *)(hashTable + f_binId * binSize + f_count), _MM_HINT_T0);
#endif
				if(count < binSize)
				{
					if(count == (binSize - 1))
					{
						count = radixSort(hashTable + binId * binSize,
								count);
						countArray[binId] = count;
					}

					if(count == (binSize - 1))
					{
						if(curOverflowBufId == overflowBufSize)
						{
							printf("ERROR! Ran out of overflowBuf, curOverflowBufId = %d.\n"
									"Try increasing numBins, binSize or overflowBufSize\n", 
									curOverflowBufId);
							return j;
//							exit(1);
						}
						t_bucket_count overflowBufId = curOverflowBufId;
						printf("Have to use overflow buf, overflowBufId = %d\n", overflowBufId);
						curOverflowBufId++;
						hashTable[binId * binSize + binSize - 1].bucketId = overflowBufId;
						overflowBuf[overflowBufId * binSize] = he;
					}
					else
					{
						hashTable[binId * binSize + count] = he;                
					}

					countArray[binId]++;
				}
				else
				{
					t_bucket_count overflowBufId;
					overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
					if(count == (binSize + ( binSize - 1)) )
					{
						t_bin_size c = radixSort(overflowBuf + overflowBufId * binSize,
								count - (binSize - 1));
						count = merge(hashTable + binId * binSize, binSize - 1,
								overflowBuf + overflowBufId * binSize, c);
						countArray[binId] = count;
					}
					if(count == (binSize + ( binSize - 1)) )
					{
						printf("ERROR! binId = %ld, count = 2 * binSize - 1. Please use larger binSize or numBins\n", binId);
						return j;
						// return exit(1);
					}
					overflowBuf[overflowBufId * binSize + count - (binSize - 1)] = he;
					countArray[binId]++;
				}
			}
        }

        //printf("hashTicks = %ld\n", hashTicks);

        return numKeys;
    }

    // T may be a key-value pair
    template <typename T>
    t_key_count insert(T *keyArray, t_key_count numKeys) {
      t_key_count inserted = 0;
      bool resize_succeeded = true;

      //std::cout << "inserting " << numKeys << std::endl;

      t_key_count delta = insert_impl(keyArray + inserted, numKeys - inserted);
      inserted += delta;
      while ((inserted < numKeys) && resize_succeeded) {
  		std::cout << "insert try to resize.  last iter inserted " << delta << std::endl;
        // did not complete, so must need to resize.
        if (! resize(numBuckets << 1)) {  // failed resizing.
          // try 1 more time.
          resize_succeeded = resize(numBuckets << 1);
        }


        if (resize_succeeded)   // successful resize, so keep inserting.
          delta = insert_impl(keyArray + inserted, numKeys - inserted);
        else
          throw ::std::logic_error("FAILED TO RESIZE 2 consecutive times.  There is probably a full bin.");

        inserted += delta;
      }

      return inserted;
    }

    template <typename T>
    t_key_count estimate_and_insert(T *keyArray, t_key_count numKeys) {

      // local hash computation and hll update.
        hash_val_type* hvals = (hash_val_type *)_mm_malloc((numKeys + PFD) * sizeof(hash_val_type), 64);
        memset(hvals + numKeys, 0, sizeof(hash_val_type) * PFD);
        this->hll.update(keyArray, numKeys, hvals);

        t_key_count est = this->hll.estimate();
    
        if (est > this->capacity())
          // add 10% just to be safe.
          this->reserve(static_cast<size_t>(static_cast<double>(est) * (1.0 + this->hll.est_error_rate + 0.1)));

        t_key_count inserted = insert(keyArray, hvals, numKeys);

        _mm_free(hvals);

        return inserted;
    }

     // return number of successful inserts.
	template <typename T, class HashType>
    t_key_count insert_impl(T *keyArray, HashType *hashArray, t_key_count numKeys)
    {
        if((coherence != COHERENT) && (coherence != INSERT))
        {
//            printf("ERROR! The hashtable is not coherent at the moment. insert() can not be serviced\n");
            return 0;
        }
        t_key_count i;
        coherence = INSERT;
        hash_val_type bucketIdArray[32];
        //int64_t hashTicks = 0;
        //int64_t startTick, endTick;
        //startTick = __rdtsc();
        for(i = 0; i < PFD; i++)
        {
            bucketIdArray[i] = hashArray[i] & bucketMask;
        }
        //endTick = __rdtsc();
        //hashTicks += (endTick - startTick);
        for(i = 0; i < (numKeys); i++)
        {
            HashElement he;
            // he.key = keyArray[i];
            // he.val = 1;
            init_hash_element(he, keyArray[i]);
            he.bucketId = bucketIdArray[i & 31];
            t_bin_count binId = he.bucketId >> binShift;
            t_bin_size count = countArray[binId];
            //startTick = __rdtsc();
            bucketIdArray[(i + PFD) & 31] = (hashArray[i + PFD] & bucketMask);
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
#if ENABLE_PREFETCH
            t_bucket_count f_bucketId = bucketIdArray[(i + PFD) & 31];
			t_bin_count f_binId = f_bucketId >> binShift;
            t_bin_size f_count = countArray[f_binId];
            _mm_prefetch((const char *)(hashTable + f_binId * binSize + f_count), _MM_HINT_T0);
#endif
            if(count < binSize)
            {
                if(count == (binSize - 1))
                {
                    count = radixSort(hashTable + binId * binSize,
                            count);
                    countArray[binId] = count;
                }

                if(count == (binSize - 1))
                {
                    if(curOverflowBufId == overflowBufSize)
                    {
                        printf("ERROR! Ran out of overflowBuf, curOverflowBufId = %d.\n"
                                "Try increasing numBins, binSize or overflowBufSize\n", 
                                curOverflowBufId);
                        return i;
//                        exit(1);
                    }
                    t_bucket_count overflowBufId = curOverflowBufId;
					printf("Have to use overflow buf, overflowBufId = %d\n", overflowBufId);
                    curOverflowBufId++;
                    hashTable[binId * binSize + binSize - 1].bucketId = overflowBufId;
                    overflowBuf[overflowBufId * binSize] = he;
                }
                else
                {
                    hashTable[binId * binSize + count] = he;                
                }

                countArray[binId]++;
            }
            else
            {
                t_bucket_count overflowBufId;
                overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
                if(count == (binSize + (binSize - 1)))
                {
                    t_bin_size c = radixSort(overflowBuf + overflowBufId * binSize,
                                          count - (binSize - 1));
                    count = merge(hashTable + binId * binSize, binSize - 1,
                          overflowBuf + overflowBufId * binSize, c);
                    countArray[binId] = count;
                }
                if(count == (binSize + (binSize - 1)))
                {
                    printf("ERROR! binId = %ld, count = 2 * binSize - 1. Please use larger binSize or numBins\n", binId);
                    return i;
                    //exit(1);
                }
                overflowBuf[overflowBufId * binSize + count - (binSize - 1)] = he;
                countArray[binId]++;
            }
        }
        //printf("hashTicks = %ld\n", hashTicks);

        return numKeys;
    }

  template <class T, class HashType>
    t_key_count insert(T *keyArray, HashType *hashArray, t_key_count numKeys) {
      t_key_count inserted = 0;
      bool resize_succeeded = true;

      //std::cout << "inserting " << numKeys << std::endl;


      t_key_count delta = insert_impl(keyArray + inserted, hashArray + inserted, numKeys - inserted);
      inserted += delta;
      while ((inserted < numKeys) && resize_succeeded) {
  		std::cout << "insert try to resize.  last iter inserted " << delta << std::endl;
        // did not complete, so must need to resize.
        if (! resize(numBuckets << 1)) {  // failed resizing.
          // try 1 more time.
          resize_succeeded = resize(numBuckets << 1);
        }


        if (resize_succeeded)   // successful resize, so keep inserting.
          delta = insert_impl(keyArray + inserted, hashArray + inserted, numKeys - inserted);
        else
          throw ::std::logic_error("FAILED TO RESIZE 2 consecutive times.  There is probably a full bin.");

        inserted += delta;
      }

      return inserted;
  }

    void finalize_insert()
    {
        if(coherence != INSERT)
        {
//            printf("ERROR! The hashtable coherence is not set to INSERT at the moment. finalize_insert() can not be serviced\n");
            return;
        }
        t_bin_count i;
        t_bin_size j;
        totalKeyCount = 0;
        for(i = 0; i < numBins; i++)
        {
            t_bin_size count = countArray[i];
            if(count < binSize)
            {
                count = radixSort(hashTable + i * binSize,
                        count);
            }
            else
            {
                t_bucket_count overflowBufId;
                overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
                t_bin_size c = radixSort(overflowBuf + overflowBufId * binSize,
                        count - (binSize - 1));
                count = merge(hashTable + i * binSize, binSize - 1,
                        overflowBuf + overflowBufId * binSize, c);
            }
            t_bucket_count firstBucketId = i << binShift;
            t_bucket_count lastBucketId = firstBucketId + sortBufSize - 1;
            t_bucket_count prevBucketId = firstBucketId - 1;
            t_bucket_count k;
			t_bin_size y = std::min(count, static_cast<t_bin_size>(binSize - 1));
            for(j = 0; j < y; j++)
            {
                t_bucket_count bucketId = hashTable[i * binSize + j].bucketId;
                if(bucketId != prevBucketId)
                {
                    for(k = prevBucketId; k < bucketId; k++)
                        info_container[k + 1] = j;
                    prevBucketId = bucketId;
                }
            }
            t_bucket_count overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                t_bucket_count bucketId = overflowBuf[overflowBufId * binSize + j - (binSize - 1)].bucketId;
                if(bucketId != prevBucketId)
                {
                    for(k = prevBucketId; k < bucketId; k++)
                        info_container[k + 1] = j;
                    prevBucketId = bucketId;
                }
            }
            for(k = prevBucketId; k < lastBucketId; k++)
                info_container[k + 1] = count;
            countArray[i] = count;
            totalKeyCount += count;
        }
        coherence = COHERENT;
    }

    t_key_count find(Key *keyArray, t_key_count numKeys, uint32_t *findResult) const
    {
        if(coherence != COHERENT)
        {
//            printf("ERROR! The hashtable is not coherent at the moment. find() can not be serviced\n");
            return 0ULL;
        }
        t_key_count foundCount = 0;
#if ENABLE_PREFETCH
        t_pfd_size PFD_INFO = PFD;
        t_pfd_size PFD_HASH = PFD >> 1;
#endif
        t_key_count i;
		t_key_count hash_batch_size = 1024;
        hash_val_type bucketIdArray[2 * hash_batch_size];
		memset(bucketIdArray, 0, 2 * hash_batch_size * sizeof(hash_val_type));
		t_key_count hash_mask = 2 * hash_batch_size - 1;

		t_key_count hash_count = std::min(numKeys, hash_batch_size);
        hash_mod2(keyArray, hash_count, bucketIdArray);

//        for(i = 0; i < PFD_INFO; i++)
//        {
//            bucketIdArray[i] = hash(keyArray[i]) & bucketMask;
//        }
        for(i = 0; i < numKeys; i += hash_batch_size)
        {
            //startTick = __rdtsc();
			t_key_count hash_first = i + hash_batch_size;
			if(hash_first > numKeys) hash_first = numKeys;
			t_key_count hash_last = i + 2 * hash_batch_size;
			if(hash_last > numKeys) hash_last = numKeys;
			t_key_count hash_count = hash_last - hash_first;
			hash_mod2(keyArray + hash_first, hash_count, bucketIdArray + ((hash_first) & hash_mask));
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
			t_key_count last = i + hash_batch_size;
			if(last > numKeys) last = numKeys;
			t_key_count j;

			for(j = i; j < last; j++)
			{
#if ENABLE_PREFETCH
				t_bucket_count f_info_bucketId = bucketIdArray[(j + PFD_INFO) & hash_mask];
				_mm_prefetch((const char *)(info_container + f_info_bucketId), _MM_HINT_T0);
				t_bucket_count f_bucketId = bucketIdArray[(j + PFD_HASH) & hash_mask];
				t_bin_count f_binId = f_bucketId >> binShift;
				t_bin_count f_start = f_binId * binSize + info_container[f_bucketId];
				_mm_prefetch((const char *)(hashTable + f_start), _MM_HINT_T0);
#endif
				t_bucket_count bucketId = bucketIdArray[j & hash_mask];
				HashElement *he = find_internal(keyArray[j], bucketId);
				if(he != NULL)
				{
					findResult[j] = he->val;
					foundCount++;
				}
				else
					findResult[j] = 0;
			}
        }

        return foundCount;
    }

    t_key_count count(Key *keyArray, t_key_count numKeys, uint8_t *countResult) const
    {
        if(coherence != COHERENT)
        {
//            printf("ERROR! The hashtable is not coherent at the moment. count() can not be serviced\n");
            return 0ULL;
        }
        t_key_count foundCount = 0;
#if ENABLE_PREFETCH
        t_pfd_size PFD_INFO = PFD;
        t_pfd_size PFD_HASH = PFD >> 1;
#endif
        t_key_count i;
		t_key_count hash_batch_size = 1024;
        hash_val_type bucketIdArray[2 * hash_batch_size];
		memset(bucketIdArray, 0, 2 * hash_batch_size * sizeof(hash_val_type));
		t_key_count hash_mask = 2 * hash_batch_size - 1;

		t_key_count hash_count = std::min(numKeys, hash_batch_size);
        hash_mod2(keyArray, hash_count, bucketIdArray);

//        for(i = 0; i < PFD_INFO; i++)
//        {
//            bucketIdArray[i] = hash(keyArray[i]) & bucketMask;
//        }

        for(i = 0; i < numKeys; i += hash_batch_size)
        {
            //startTick = __rdtsc();
			t_key_count hash_first = i + hash_batch_size;
			if(hash_first > numKeys) hash_first = numKeys;
			t_key_count hash_last = i + 2 * hash_batch_size;
			if(hash_last > numKeys) hash_last = numKeys;
			t_key_count hash_count = hash_last - hash_first;
			hash_mod2(keyArray + hash_first, hash_count, bucketIdArray + ((hash_first) & hash_mask));
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
			t_key_count last = i + hash_batch_size;
			if(last > numKeys) last = numKeys;
			t_key_count j;


			for(j = i; j < last; j++)
			{
#if ENABLE_PREFETCH
				t_bucket_count f_info_bucketId = bucketIdArray[(j + PFD_INFO) & hash_mask];
				_mm_prefetch((const char *)(info_container + f_info_bucketId), _MM_HINT_T0);
				t_bucket_count f_bucketId = bucketIdArray[(j + PFD_HASH) & hash_mask];
				t_bin_count f_binId = f_bucketId >> binShift;
				t_bin_count f_start = f_binId * binSize + info_container[f_bucketId];
				_mm_prefetch((const char *)(hashTable + f_start), _MM_HINT_T0);
#endif
				t_bucket_count bucketId = bucketIdArray[j & hash_mask];
				HashElement *he = find_internal(keyArray[j], bucketId);
				if(he != NULL)
				{
					countResult[j] = 1;
					foundCount++;
				}
				else
					countResult[j] = noValue;
			}
        }
        return foundCount;
    }

    void erase(Key *keyArray, t_key_count numKeys)
    {
        if((coherence != COHERENT) && (coherence != ERASE))
        {
//            printf("ERROR! The hashtable is not coherent at the moment. erase() can not be serviced\n");
            return;
        }
        coherence = ERASE;
#if ENABLE_PREFETCH
        t_pfd_size PFD_INFO = PFD;
        t_pfd_size PFD_HASH = PFD >> 1;
#endif
        t_key_count i;
		t_key_count hash_batch_size = 1024;
        hash_val_type bucketIdArray[2 * hash_batch_size];
		memset(bucketIdArray, 0, 2 * hash_batch_size * sizeof(hash_val_type));
		t_key_count hash_mask = 2 * hash_batch_size - 1;

		t_key_count hash_count = std::min(numKeys, hash_batch_size);
        hash_mod2(keyArray, hash_count, bucketIdArray);

//        for(i = 0; i < PFD_INFO; i++)
//        {
//            bucketIdArray[i] = hash(keyArray[i]) & bucketMask;
//        }

        for(i = 0; i < numKeys; i += hash_batch_size)
        {
            //startTick = __rdtsc();
			t_key_count hash_first = i + hash_batch_size;
			if(hash_first > numKeys) hash_first = numKeys;
			t_key_count hash_last = i + 2 * hash_batch_size;
			if(hash_last > numKeys) hash_last = numKeys;
			t_key_count hash_count = hash_last - hash_first;
			hash_mod2(keyArray + hash_first, hash_count, bucketIdArray + ((hash_first) & hash_mask));
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
			t_key_count last = i + hash_batch_size;
			if(last > numKeys) last = numKeys;
			t_key_count j;


			for(j = i; j < last; j++)
			{
#if ENABLE_PREFETCH
				t_bucket_count f_info_bucketId = bucketIdArray[(j + PFD_INFO) & hash_mask];
				_mm_prefetch((const char *)(info_container + f_info_bucketId), _MM_HINT_T0);
				t_bucket_count f_bucketId = bucketIdArray[(j + PFD_HASH) & hash_mask];
				t_bin_count f_binId = f_bucketId >> binShift;
				t_bin_count f_start = f_binId * binSize + info_container[f_bucketId];
				_mm_prefetch((const char *)(hashTable + f_start), _MM_HINT_T0);
#endif
				t_bucket_count bucketId = bucketIdArray[j & hash_mask];
				HashElement *he = find_internal(keyArray[j], bucketId);
				if(he != NULL)
					he->val = noValue;
			}
        }
    }

    t_key_count finalize_erase()
    {
        if(coherence != ERASE)
        {
//            printf("ERROR! The hashtable coherence is not set to ERASE at the moment. finalize_erase() can not be serviced\n");
            return 0;
        }
        t_bin_count i;

        t_key_count eraseCount = totalKeyCount;
        totalKeyCount = 0;
        for(i = 0; i < numBins; i++)
        {
            t_bin_size count = countArray[i];

            t_bin_size p1, p2;
            t_bin_size y = std::min(count, static_cast<t_bin_size>(binSize - 1));
            p1 = p2 = 0;
            for(p2 = 0; p2 < y; p2++)
            {
                HashElement he = hashTable[i * binSize + p2];
                if(he.val != noValue)
                {
                    hashTable[i * binSize + p1] = he;
                    p1++;
                }
            }
            t_bucket_count overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; p2 < count; p2++)
            {
                HashElement he = overflowBuf[overflowBufId * binSize + p2 - (binSize - 1)];
                if(he.val != noValue)
                {
                    if(p1 < (binSize - 1))
                    {
                        hashTable[i * binSize + p1] = he;
                    }
                    else
                    {
                        overflowBuf[overflowBufId * binSize + p1 - (binSize - 1)] = he;
                    }
                    p1++;
                }
            }
            count = p1;
            t_bucket_count firstBucketId = i << binShift;
            t_bucket_count lastBucketId = firstBucketId + sortBufSize - 1;
            t_bucket_count prevBucketId = firstBucketId - 1;
            t_bucket_count k;
            t_bin_size j;
            y = std::min(count, static_cast<t_bin_size>(binSize - 1));
            for(j = 0; j < y; j++)
            {
                t_bucket_count bucketId = hashTable[i * binSize + j].bucketId;
                if(bucketId != prevBucketId)
                {
                    for(k = prevBucketId; k < bucketId; k++)
                        info_container[k + 1] = j;
                    prevBucketId = bucketId;
                }
            }
            for(; j < count; j++)
            {
                t_bucket_count bucketId = overflowBuf[overflowBufId * binSize + j - (binSize - 1)].bucketId;
                if(bucketId != prevBucketId)
                {
                    for(k = prevBucketId; k < bucketId; k++)
                        info_container[k + 1] = j;
                    prevBucketId = bucketId;
                }
            }
            for(k = prevBucketId; k < lastBucketId; k++)
                info_container[k + 1] = count;
            countArray[i] = count;
            totalKeyCount += count;
        }
        coherence = COHERENT;
        eraseCount -= totalKeyCount;
        return eraseCount;
    }

    void sanity_check()
    {
        t_key_count totalCount = 0;
        t_bin_size maxCount = 0;
        t_bin_count maxBin = -1;
        t_bin_count i;

        size_t kmerCountSum = 0;
        t_bucket_count prevBucketId = 0;
        for(i = 0; i < numBins; i++)
        {
            t_bin_size j;
            t_bin_size count = countArray[i];
            if(count > maxCount)
            {
                maxCount = count;
                maxBin = i;
            }
            totalCount += count;
			t_bin_size y = std::min(count, static_cast<t_bin_size>(binSize - 1));
            for(j = 0; j < y; j++)
            {
                HashElement he = hashTable[i * binSize + j];
                t_bucket_count bucketId = he.bucketId;
                if(bucketId < prevBucketId)
                {
                    printf("ERROR! [%ld,%d] prevBucketId = %u, bucketId = %u\n", i, j, prevBucketId, bucketId);
                    exit(0);
                }
                kmerCountSum += (he.val * he.val);
                prevBucketId = bucketId;
            }
            t_bucket_count overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                HashElement he = overflowBuf[overflowBufId * binSize + j - (binSize - 1)];
                t_bucket_count bucketId = he.bucketId;
                if(bucketId < prevBucketId)
                {
                    printf("ERROR! [%ld,%d] prevBucketId = %u, bucketId = %u\n", i, j, prevBucketId, bucketId);
                    exit(0);
                }
                kmerCountSum += (he.val * he.val);
                prevBucketId = bucketId;
            }
        }
        printf("kmerCountSum = %ld\n", kmerCountSum);
        for(i = 0; i < numBuckets; i++)
        {
            t_bin_size j;
            t_bucket_count binId = i >> binShift;
            t_bin_size start = info_container[i];
            t_bucket_count binId2 = (i + 1) >> binShift;
            t_bin_size end;
            if(binId == binId2)
                end = info_container[i + 1];
            else
                end = countArray[binId];

            if(end < start)
            {
                printf("ERROR! [%ld] start = %d, end = %d, binId = %ld, binId2 = %ld\n", i, start, end, binId, binId2);
                exit(0);
            }
            for(j = start; j < end; j++)
            {
                t_bucket_count bucketId;
                if(j < (binSize - 1))
                {
                    bucketId = hashTable[binId * binSize + j].bucketId;
                }
                else
                {
                    t_bucket_count overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
                    bucketId = overflowBuf[overflowBufId * binSize + j - (binSize - 1)].bucketId;
                }
                if(bucketId != i)
                {
                    printf("ERROR! [%ld,%d] bucketId = %u\n", i, j, bucketId);
                    exit(0);
                }
            }
        }
        printf("Sanity check passed!\n");
        printf("totalCount = %d, maxCount = %d, maxBin = %d, curOverflowBufId = %d\n", totalCount, maxCount, maxBin, curOverflowBufId);

    }

    t_key_count size() const {
		if(coherence != COHERENT)
		{
//			printf("ERROR! The hashtable is not coherent at the moment. const version of size() called and cannot continue. \n");
			throw std::logic_error("size() called on incoherent hash table");
		}

        return totalKeyCount;
    }

    t_key_count size() {
		if(coherence == INSERT)
		{
//			printf("WARNING! The hashtable is in INSERT mode at the moment. finalize_insert will be called \n");
			finalize_insert();
		} else if (coherence == ERASE) {
//		  printf("WARNING! The hashtable is in ERASE mode at the moment. finalize_erase will be called \n");
		  finalize_erase();
		}

        return totalKeyCount;
    }


    t_key_count capacity() const {
      return numBuckets;
    }


	inline hyperloglog64<Key, Hash<Key>, 12>& get_hll() {
		return this->hll;
	}

	::std::pair<Key, V> *getData(t_key_count *resultCount) const
	{
// 		if(coherence == INSERT)
// 		{
// //			printf("WARNING! The hashtable is in INSERT mode at the moment. finalize_insert will be called \n");
// 			finalize_insert();
// 		} else if (coherence == ERASE) {
// //		  printf("WARNING! The hashtable is in ERASE mode at the moment. finalize_erase will be called \n");
// 		  finalize_erase();
// 		}
        if(coherence != COHERENT)
		{
			printf("ERROR! The hashtable is not coherent at the moment. const version of size() called and cannot continue. \n");
			throw std::logic_error("size() called on incoherent hash table");
		}
        ::std::pair<Key, V> *result = (::std::pair<Key, V> *)_mm_malloc(totalKeyCount * sizeof(::std::pair<Key, V>), 64);

	    t_bin_count i;
        t_bin_size j;
		t_key_count elemCount = 0;
		for(i = 0; i < numBins; i++)
		{
			t_bin_size count = countArray[i];
			t_bin_size y = std::min(count, static_cast<t_bin_size>(binSize - 1));
			for(j = 0; j < y; j++)
			{
				HashElement he = hashTable[i * binSize + j];
				result[elemCount] = ::std::make_pair(he.key, he.val);
				elemCount++;
			}
            t_bucket_count overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                HashElement he = overflowBuf[overflowBufId * binSize + j - (binSize - 1)];
				result[elemCount] = ::std::make_pair(he.key, he.val);
				elemCount++;
            }
		}
		(*resultCount) = elemCount;
		return result;
	}

    /// Fill data into an array.  array should be preallocated to be big enough.
	t_key_count getData(::std::pair<Key, V> *result) const
	{
// 		if(coherence == INSERT)
// 		{
// //			printf("WARNING! The hashtable is in INSERT mode at the moment. finalize_insert will be called \n");
// 			finalize_insert();
// 		} else if (coherence == ERASE) {
// //		  printf("WARNING! The hashtable is in ERASE mode at the moment. finalize_erase will be called \n");
// 		  finalize_erase();
// 		}
        if(coherence != COHERENT)
		{
			printf("ERROR! The hashtable is not coherent at the moment. const version of size() called and cannot continue. \n");
			throw std::logic_error("size() called on incoherent hash table");
		}
	    t_bin_count i;
        t_bin_size j;
		t_key_count elemCount = 0;
		for(i = 0; i < numBins; i++)
		{
			t_bin_size count = countArray[i];
			t_bin_size y = std::min(count, static_cast<t_bin_size>(binSize - 1));
			for(j = 0; j < y; j++)
			{
				HashElement he = hashTable[i * binSize + j];
				result[elemCount] = ::std::make_pair(he.key, he.val);
				elemCount++;
			}
            t_bucket_count overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                HashElement he = overflowBuf[overflowBufId * binSize + j - (binSize - 1)];
				result[elemCount] = ::std::make_pair(he.key, he.val);
				elemCount++;
            }
		}
		return elemCount;
	}


	::std::vector<::std::pair<Key, V> > to_vector() const {
// 		if(coherence == INSERT)
// 		{
// //			printf("WARNING! The hashtable is in INSERT mode at the moment. finalize_insert will be called \n");
// 			finalize_insert();
// 		} else if (coherence == ERASE) {
// //		  printf("WARNING! The hashtable is in ERASE mode at the moment. finalize_erase will be called \n");
// 		  finalize_erase();
// 		}
        if(coherence != COHERENT)
		{
			printf("ERROR! The hashtable is not coherent at the moment. const version of size() called and cannot continue. \n");
			throw std::logic_error("size() called on incoherent hash table");
		}

        ::std::vector<::std::pair<Key, V> > result(totalKeyCount);
        getData(result.data());
		return result;
	}

	::std::vector<Key> keys() const {
// 		if(coherence == INSERT)
// 		{
// //			printf("WARNING! The hashtable is in INSERT mode at the moment. finalize_insert will be called \n");
// 			finalize_insert();
// 		} else if (coherence == ERASE) {
// //		  printf("WARNING! The hashtable is in ERASE mode at the moment. finalize_erase will be called \n");
// 		  finalize_erase();
// 		}
        if(coherence != COHERENT)
		{
			printf("ERROR! The hashtable is not coherent at the moment. const version of size() called and cannot continue. \n");
			throw std::logic_error("size() called on incoherent hash table");
		}

        ::std::vector< Key > result(totalKeyCount);

	    t_bin_count i;
        t_bin_size j;
		t_key_count elemCount = 0;
		for(i = 0; i < numBins; i++)
		{
			t_bin_size count = countArray[i];
			t_bin_size y = std::min(count, static_cast<t_bin_size>(binSize - 1));
			for(j = 0; j < y; j++)
			{
				HashElement he = hashTable[i * binSize + j];
				result[elemCount] = he.key;
				elemCount++;
			}
            t_bucket_count overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                HashElement he = overflowBuf[overflowBufId * binSize + j - (binSize - 1)];
				result[elemCount] = he.key;
				elemCount++;
            }
		}
		return result;
	}


    /// serialize to unsigned char array.  return byte count written.
    template <typename SERIALIZER>
	size_t serialize(unsigned char *result, SERIALIZER const & ser) const
	{
// 		if(coherence == INSERT)
// 		{
// //			printf("WARNING! The hashtable is in INSERT mode at the moment. finalize_insert will be called \n");
// 			finalize_insert();
// 		} else if (coherence == ERASE) {
// //		  printf("WARNING! The hashtable is in ERASE mode at the moment. finalize_erase will be called \n");
// 		  finalize_erase();
// 		}
        if(coherence != COHERENT)
		{
			printf("ERROR! The hashtable is not coherent at the moment. const version of size() called and cannot continue. \n");
			throw std::logic_error("size() called on incoherent hash table");
		}

	    t_bin_count i;
        t_bin_size j;
        unsigned char * it = result;
		for(i = 0; i < numBins; i++)
		{
			t_bin_size count = countArray[i];
			t_bin_size y = std::min(count, static_cast<t_bin_size>(binSize - 1));
			for(j = 0; j < y; j++)
			{
				HashElement he = hashTable[i * binSize + j];
				it = ser(he.key, he.val, it); // serialize the key and value, and advance 'it' as far as needed.
			}
            t_bucket_count overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                HashElement he = overflowBuf[overflowBufId * binSize + j - (binSize - 1)];
				it = ser(he.key, he.val, it); // serialize the key and value, and advance 'it' as far as needed.
            }
		}
		return std::distance(result, it);
	}
};
template <class Key, class V, template <typename> class Hash,
          template <typename> class Equal, typename Reduc>
constexpr typename hashmap_radixsort<Key, V, Hash, Equal, Reduc>::t_pfd_size hashmap_radixsort<Key, V, Hash, Equal, Reduc>::PFD;


}  // namespace fsc
#endif /* KMERHASH_HASHMAP_RADIXSORT_HPP_ */
