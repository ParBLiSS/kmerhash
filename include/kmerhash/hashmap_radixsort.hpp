#ifndef KMERHASH_HASHMAP_RADIXSORT_HPP_
#define KMERHASH_HASHMAP_RADIXSORT_HPP_
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <math.h>
#include <functional>
#ifdef VTUNE_ANALYSIS
#include <ittnotify.h>
#endif

//#include "MurmurHash3.h"
#include "math_utils.hpp"
#include "mem_utils.hpp"
#include "hash.hpp"

#include "iterators/transform_iterator.hpp"

//#define min(x,y) (((x) < (y)) ? (x):(y))

namespace fsc {
template <class Key, class V, template <typename> class Hash = ::std::hash,
          template <typename> class Equal = ::std::equal_to
         >
class hashmap_radixsort {


public:


    using key_type              = Key;
    using mapped_type           = V;
    using value_type            = ::std::pair<const Key, V>;
    using hasher                = Hash<Key>;
    using key_equal             = Equal<Key>;
    using size_type             = size_t;
    using difference_type       = ptrdiff_t;

    typedef struct elem
    {
        Key key;
        V val;
        int32_t bucketId;
    } HashElement;

    struct HashElementTransform {
    	::std::pair<const Key, V> operator()(HashElement const & x) const {
    		return std::make_pair(x.key, x.val);
    	}
    };
    struct HashElementConstTransform {
    	const ::std::pair<const Key, V> operator()(HashElement const & x) const {
    		return std::make_pair(x.key, x.val);
    	}
    };
	using iterator              = ::bliss::iterator::transform_iterator<HashElement*, HashElementTransform>;
	using const_iterator        = ::bliss::iterator::transform_iterator<HashElement const *, HashElementConstTransform>;


#define COHERENT 0
#define INSERT 1
#define ERASE 2

    protected:
    int32_t numBuckets;
    int32_t bucketMask;
    int32_t numBins;
    int32_t binMask;
    int32_t binShift;
    int32_t binSize;
    int32_t overflowBufSize;
    int32_t curOverflowBufId;
    int32_t sortBufSize;
    mutable V noValue;
    int8_t  coherence;
    int32_t seed;
    int32_t totalKeyCount;

    uint16_t *countArray;
    HashElement *hashTable;
    HashElement *overflowBuf;
    HashElement *sortBuf;
    uint16_t *countSortBuf;
    int16_t *info_container;

    Equal<Key> eq;
    Hash<Key> hash;
	hyperloglog64<Key, Hash<Key>, 12> hll;  // precision of 12bits  error rate : 1.04/(2^6)

	template <typename S>
	struct modulus2 {
		static constexpr size_t batch_size = (sizeof(S) == 4 ? 8 : 4);
		S mask;
		modulus2(S const & _mask) : mask(_mask) {}

		template <typename IN>
		IN operator()(IN const & x) const { return (x & mask); }

		template <typename IN, typename OUT>
		void operator()(IN const * x, size_t const & _count, OUT * y) const {
			for (size_t i = 0; i < _count; ++i)  y[i] = x[i] & mask;
		}
	};

	// mod 2 okay since hashtable size is always power of 2.
	using InternalHash = ::fsc::hash::TransformedHash<Key, Hash, ::bliss::transform::identity, modulus2>;
	using hash_val_type = typename InternalHash::HASH_VAL_TYPE;
    InternalHash hash_mod2; 


    int32_t radixSort(HashElement *A,
                  int32_t size)
    {
        HashElement *sortBuf = this->sortBuf;
        uint16_t *countBuf = this->countSortBuf;
        //int32_t shift = this->binShift;
        int32_t bufSize = this->sortBufSize;
        //int32_t binSize = this->binSize;
#if 1
        if(size <= 1) return size;
        int32_t mask = bufSize - 1;
        memset(countBuf, 0, bufSize * sizeof(uint16_t));
        int i;
        for(i = 0; i < size; i++)
        {
            HashElement he = A[i];
            int32_t id = he.bucketId & mask;
            //printf("%d] bucketId = %d, id = %d\n", i, he.bucketId, id);
            countBuf[id]++;
        }
        int cumulSum = 0;
        for(i = 0; i < bufSize; i++)
        {
            int32_t c = countBuf[i];
            countBuf[i] = cumulSum;
            //printf("%d] count = %d, cumulSum = %d\n", i, c, cumulSum);
            cumulSum += c;
        }

        for(i = 0; i < size; i++)
        {
            HashElement he = A[i];
            int32_t id = he.bucketId & mask;
            int32_t pos = countBuf[id];
            //printf("%d] bucketId = %d, id = %d, pos = %d\n", i, he.bucketId, id, pos);
            sortBuf[pos] = he;
            countBuf[id]++;
        }

        for(i = 1; i < size; i++)
        {
            if(sortBuf[i].bucketId < sortBuf[i - 1].bucketId)
            {
                printf("ERROR! %d] %d, %d\n", i, sortBuf[i - 1].bucketId, sortBuf[i].bucketId);
                exit(0);
            }
        }

        uint32_t curBid = sortBuf[0].bucketId;
        uint32_t curStart = 0;
        A[0] = sortBuf[0];
        int32_t count = 1;
        for(i = 1; i < size; i++)
        {
            uint32_t bid = sortBuf[i].bucketId;
            Key key = sortBuf[i].key;
            if(bid == curBid)
            {
                int32_t j;
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
            int32_t id = he.bucketId & mask;
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

    int32_t merge(HashElement *A, int32_t sizeA, HashElement *B, int32_t sizeB)
    {
        //printf("sizeA = %d, sizeB = %d\n", sizeA, sizeB);
        int32_t pA, pB;

        pA = pB = 0;

        int32_t count = 0;

        while((pA < sizeA) && (pB < sizeB))
        {
            int32_t bidA = A[pA].bucketId;
            int32_t bidB = B[pB].bucketId;
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
        int32_t i;
        uint32_t curBid = sortBuf[0].bucketId;
        uint32_t curStart = 0;
        int32_t size = count;
        count = 1;
        HashElement *newBuf = (HashElement *)_mm_malloc(size * sizeof(HashElement), 64);
        newBuf[0] = sortBuf[0];
        for(i = 1; i < size; i++)
        {
            uint32_t bid = sortBuf[i].bucketId;
            Key key = sortBuf[i].key;
            if(bid == curBid)
            {
                int32_t j;
                for(j = curStart; j < count; j++)
                {
                    if(eq(key, newBuf[j].key))
                    {
                        newBuf[j].val++;
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
        for(i = 1; i < count; i++)
        {
            if(newBuf[i].bucketId < newBuf[i - 1].bucketId)
            {
                printf("ERROR! %d] %d, %d\n", i, newBuf[i - 1].bucketId, newBuf[i].bucketId);
                exit(0);
            }
        }
        for(i = 0; i < sizeA; i++)
            A[i] = newBuf[i];

        for(; i < count; i++)
            B[i - sizeA] = newBuf[i];

		_mm_free(newBuf);
        return count;
    }

    inline HashElement *find_internal(Key key, int32_t bucketId) const
    {
        int32_t binId = bucketId >> binShift;
        int32_t binId2 = (bucketId + 1) >> binShift;
        int32_t start = info_container[bucketId];
        int32_t end;
        if(binId == binId2)
            end = info_container[bucketId + 1];
        else
            end = countArray[binId];

        int32_t j;
        HashElement *he;
        for(j = start; j < end; j++)
        {
            if(j < (binSize - 1))
            {
                he = hashTable + binId * binSize + j;
            }
            else
            {
                int32_t overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
                he = overflowBuf + overflowBufId * binSize + j - (binSize - 1);
            }
            if(eq(key, he->key))
            {
                return he;
            }
        }
        return NULL;
    }

    public:
    hashmap_radixsort(uint32_t _numBuckets = 1048576,
            uint32_t _binSize = 1024,
            V _noValue = 0) :
            	numBuckets(next_power_of_2(_numBuckets)),
				bucketMask(numBuckets - 1),
				hash_mod2(hash, ::bliss::transform::identity<Key>(), modulus2<hash_val_type>(bucketMask))
    {
        binSize = next_power_of_2(_binSize);
        numBins = numBuckets * 2 / binSize;
        binMask = numBins - 1;
        overflowBufSize = numBins / 8;
        noValue = _noValue;

        countArray = (uint16_t *)_mm_malloc(numBins * sizeof(uint16_t), 64);
        memset(countArray, 0, numBins * sizeof(uint16_t));
        hashTable = (HashElement *)_mm_malloc(numBins * binSize * sizeof(HashElement), 64);
        overflowBuf = (HashElement *)_mm_malloc(overflowBufSize * binSize * sizeof(HashElement), 64);
        sortBufSize = numBuckets / numBins;
        sortBuf = (HashElement *)_mm_malloc(sortBufSize * binSize * sizeof(HashElement), 64);
        countSortBuf = (uint16_t *)_mm_malloc(sortBufSize * sizeof(uint16_t), 64);
        binShift = log2(sortBufSize);
        info_container = (int16_t *)_mm_malloc(numBuckets * sizeof(int16_t), 64);
        curOverflowBufId = 0;
        printf("numBuckets = %d, numBins = %d, binSize = %d, overflowBufSize = %d, sortBufSize = %d, binShift = %d\n",
                numBuckets, numBins, binSize, overflowBufSize, sortBufSize, binShift);
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

	void set_novalue(V _noValue) const { noValue = _noValue; }

	const_iterator cbegin() const {
		std::cerr << "WARNING: NOT FILTERING OUT EMPTY ENTRIES" << std::endl;
		return const_iterator(hashTable, HashElementConstTransform());
	}

	const_iterator cend() const {
		std::cerr << "WARNING: NOT FILTERING OUT EMPTY ENTRIES" << std::endl;
		return const_iterator(hashTable + numBins * binSize, HashElementConstTransform());
	}

	::std::vector<::std::pair<Key, V> > to_vector() const {
		std::cerr << "WARNING: NOT IMPLEMENTED" << std::endl;
		return ::std::vector<::std::pair<Key, V> >();
	}

	::std::vector<Key> keys() const {
		std::cerr << "WARNING: NOT IMPLEMENTED" << std::endl;
		return ::std::vector<Key>();
	}

	void reserve(uint32_t _newElementCount) {
		resize(next_power_of_2(_newElementCount));
	}

	void resize(uint32_t _newNumBuckets)
	{
        int PFD = 16;
		HashElement *oldElemArray = (HashElement *)_mm_malloc((totalKeyCount + PFD) * sizeof(HashElement), 64);

	    int32_t i, j;
		int32_t elemCount = 0;
		for(i = 0; i < numBins; i++)
		{
			int count = countArray[i];
			int y = std::min(count, binSize - 1);
			for(j = 0; j < y; j++)
			{
				oldElemArray[elemCount++] = hashTable[i * binSize + y];
			}
            int32_t overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                oldElemArray[elemCount++] = overflowBuf[overflowBufId * binSize + j - (binSize - 1)];
            }
		}
		printf("elemCount = %d\n", elemCount);

        numBuckets = next_power_of_2(_newNumBuckets);
        bucketMask = numBuckets - 1;
        numBins = std::max(1, (numBuckets << 1) / binSize);   // TCP: at least 1 bin
        binMask = numBins - 1;
        overflowBufSize = std::max(1, numBins / 8);   // TCP: overflow of at least 1.

		_mm_free(countArray);
        countArray = (uint16_t *)_mm_malloc(numBins * sizeof(uint16_t), 64);
        memset(countArray, 0, numBins * sizeof(uint16_t));

		_mm_free(hashTable);
        hashTable = (HashElement *)_mm_malloc(numBins * binSize * sizeof(HashElement), 64);

		_mm_free(overflowBuf);
        overflowBuf = (HashElement *)_mm_malloc(overflowBufSize * binSize * sizeof(HashElement), 64);
        sortBufSize = numBuckets / numBins;

		_mm_free(sortBuf);
        sortBuf = (HashElement *)_mm_malloc(sortBufSize * binSize * sizeof(HashElement), 64);

		_mm_free(countSortBuf);
        countSortBuf = (uint16_t *)_mm_malloc(sortBufSize * sizeof(uint16_t), 64);
        binShift = log2(sortBufSize);

		_mm_free(info_container);
        info_container = (int16_t *)_mm_malloc(numBuckets * sizeof(int16_t), 64);
        curOverflowBufId = 0;
        printf("numBuckets = %d, numBins = %d, binSize = %d, overflowBufSize = %d, sortBufSize = %d, binShift = %d\n",
                numBuckets, numBins, binSize, overflowBufSize, sortBufSize, binShift);

		// insert the elements again
        coherence = INSERT;

        hash_mod2.posttrans.mask = bucketMask;
        hash_val_type bucketIdArray[32];
        Key keyArray[32];
        int32_t max = std::min(PFD << 1, elemCount);
        for(i = 0; i < max; i++)
        {
        	keyArray[i] = oldElemArray[i].key;
        	//            oldElemArray[i].bucketId = hash(oldElemArray[i].key) & bucketMask;
        }
        hash_mod2(keyArray, PFD, bucketIdArray);
        max = std::min(PFD, elemCount);
        for(i = 0; i < max; i++)
        {
        	oldElemArray[i].bucketId = bucketIdArray[i];
        }


        max = elemCount - (elemCount & (PFD - 1)) - PFD;
        for(i = 0; i < max; i++)
        {
        	if ((i & (PFD - 1)) == 0)
        		hash_mod2(keyArray + ((i + PFD) & 31), PFD, bucketIdArray + ((i+PFD) & 31));
        	// prep the key array for the next PFD
        	keyArray[i & 31] = oldElemArray[(i + (PFD << 1)) & 31].key;
        	// copy out the bucketIdArray

            HashElement he = oldElemArray[i];
            int binId = he.bucketId >> binShift;
            int count = countArray[binId];
            int32_t f_bucketId = oldElemArray[(i + PFD)].bucketId =  bucketIdArray[ (i + PFD) & 31 ];
            	//(hash(oldElemArray[i + PFD].key) & bucketMask);
            int f_binId = f_bucketId >> binShift;
            int f_count = countArray[f_binId];
            _mm_prefetch((const char *)(hashTable + f_binId * binSize + f_count), _MM_HINT_T0);
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
                        exit(1);
                    }
                    int32_t overflowBufId = curOverflowBufId;
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
                int32_t overflowBufId;
                overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
                if(count == (2 * binSize - 1))
                {
                    int32_t c = radixSort(overflowBuf + overflowBufId * binSize,
                                          count - (binSize - 1));
                    count = merge(hashTable + binId * binSize, binSize - 1,
                          overflowBuf + overflowBufId * binSize, c);
                    countArray[binId] = count;
                }
                if(count == (2 * binSize - 1))
                {
                    printf("ERROR! binId = %d, count = 2 * binSize - 1. Please use larger binSize or numBins\n", binId);
                    exit(1);
                }
                overflowBuf[overflowBufId * binSize + count - (binSize - 1)] = he;
                countArray[binId]++;
            }
        }

        max = elemCount - PFD;
        for(; i < max; i++)
        {

            HashElement he = oldElemArray[i];
            int binId = he.bucketId >> binShift;
            int count = countArray[binId];
            int32_t f_bucketId = oldElemArray[(i + PFD)].bucketId = (hash(oldElemArray[i + PFD].key) & bucketMask);
            int f_binId = f_bucketId >> binShift;
            int f_count = countArray[f_binId];
            _mm_prefetch((const char *)(hashTable + f_binId * binSize + f_count), _MM_HINT_T0);
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
                        exit(1);
                    }
                    int32_t overflowBufId = curOverflowBufId;
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
                int32_t overflowBufId;
                overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
                if(count == (2 * binSize - 1))
                {
                    int32_t c = radixSort(overflowBuf + overflowBufId * binSize,
                                          count - (binSize - 1));
                    count = merge(hashTable + binId * binSize, binSize - 1,
                          overflowBuf + overflowBufId * binSize, c);
                    countArray[binId] = count;
                }
                if(count == (2 * binSize - 1))
                {
                    printf("ERROR! binId = %d, count = 2 * binSize - 1. Please use larger binSize or numBins\n", binId);
                    exit(1);
                }
                overflowBuf[overflowBufId * binSize + count - (binSize - 1)] = he;
                countArray[binId]++;
            }
        }


        for(; i < (elemCount); i++)
        {
            HashElement he = oldElemArray[i];
            int binId = he.bucketId >> binShift;
            int count = countArray[binId];
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
                        exit(1);
                    }
                    int32_t overflowBufId = curOverflowBufId;
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
                int32_t overflowBufId;
                overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
                if(count == (2 * binSize - 1))
                {
                    int32_t c = radixSort(overflowBuf + overflowBufId * binSize,
                                          count - (binSize - 1));
                    count = merge(hashTable + binId * binSize, binSize - 1,
                          overflowBuf + overflowBufId * binSize, c);
                    countArray[binId] = count;
                }
                if(count == (2 * binSize - 1))
                {
                    printf("ERROR! binId = %d, count = 2 * binSize - 1. Please use larger binSize or numBins\n", binId);
                    exit(1);
                }
                overflowBuf[overflowBufId * binSize + count - (binSize - 1)] = he;
                countArray[binId]++;
            }
        }


        _mm_free(oldElemArray);
		finalize_insert();
	}


    //uint64_t hash(uint64_t x)
    //{
    //    uint64_t y[2];
    //    MurmurHash3_x64_128(&x, sizeof(uint64_t), seed, y);
    //    return y[0];
    //}

    void insert(Key *keyArray, int32_t numKeys)
    {
        if((coherence != COHERENT) && (coherence != INSERT))
        {
            printf("ERROR! The hashtable is not coherent at the moment. insert() can not be serviced\n");
            return;
        }
        int PFD = 16;
        int32_t i;
        coherence = INSERT;
		int32_t hash_batch_size = 1024;
        hash_val_type bucketIdArray[2 * hash_batch_size];
		memset(bucketIdArray, 0, 2 * hash_batch_size * sizeof(hash_val_type));
		int32_t hash_mask = 2 * hash_batch_size - 1;
        //int64_t hashTicks = 0;
        //int64_t startTick, endTick;
        //startTick = __rdtsc();

		int32_t hash_count = std::min(numKeys, hash_batch_size);
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
			int32_t hash_first = i + hash_batch_size;
			if(hash_first > numKeys) hash_first = numKeys;
			int32_t hash_last = i + 2 * hash_batch_size;
			if(hash_last > numKeys) hash_last = numKeys;
			int32_t hash_count = hash_last - hash_first;
			hash_mod2(keyArray + hash_first, hash_count, bucketIdArray + ((hash_first) & hash_mask));
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
			int32_t last = i + hash_batch_size;
			if(last > numKeys) last = numKeys;
			int32_t j;
			for(j = i; j < last; j++)
			{
				HashElement he;
				he.key = keyArray[j];
				he.val = 1;
				he.bucketId = bucketIdArray[j & hash_mask];
				int binId = he.bucketId >> binShift;
				int count = countArray[binId];
				int32_t f_bucketId = bucketIdArray[(j + PFD) & hash_mask]; // = (hash(keyArray[i + PFD]) & bucketMask);
				int f_binId = f_bucketId >> binShift;
				int f_count = countArray[f_binId];
				_mm_prefetch((const char *)(hashTable + f_binId * binSize + f_count), _MM_HINT_T0);
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
							exit(1);
						}
						int32_t overflowBufId = curOverflowBufId;
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
					int32_t overflowBufId;
					overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
					if(count == (2 * binSize - 1))
					{
						int32_t c = radixSort(overflowBuf + overflowBufId * binSize,
								count - (binSize - 1));
						count = merge(hashTable + binId * binSize, binSize - 1,
								overflowBuf + overflowBufId * binSize, c);
						countArray[binId] = count;
					}
					if(count == (2 * binSize - 1))
					{
						printf("ERROR! binId = %d, count = 2 * binSize - 1. Please use larger binSize or numBins\n", binId);
						exit(1);
					}
					overflowBuf[overflowBufId * binSize + count - (binSize - 1)] = he;
					countArray[binId]++;
				}
			}
        }

        //printf("hashTicks = %ld\n", hashTicks);
    }

	template <class HashType>
    void insert(Key *keyArray, HashType *hashArray, int32_t numKeys)
    {
        if((coherence != COHERENT) && (coherence != INSERT))
        {
            printf("ERROR! The hashtable is not coherent at the moment. insert() can not be serviced\n");
            return;
        }
        int PFD = 16;
        int32_t i;
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
            he.key = keyArray[i];
            he.val = 1;
            he.bucketId = bucketIdArray[i & 31];
            int binId = he.bucketId >> binShift;
            int count = countArray[binId];
            //startTick = __rdtsc();
            int32_t f_bucketId = bucketIdArray[(i + PFD) & 31] = (hashArray[i + PFD] & bucketMask);
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
            int f_binId = f_bucketId >> binShift;
            int f_count = countArray[f_binId];
            //_mm_prefetch((const char *)(hashTable + f_bucketId), _MM_HINT_T0);
            //_mm_prefetch((const char *)(countArray + f_binId), _MM_HINT_T0);
            _mm_prefetch((const char *)(hashTable + f_binId * binSize + f_count), _MM_HINT_T0);
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
                        exit(1);
                    }
                    int32_t overflowBufId = curOverflowBufId;
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
                int32_t overflowBufId;
                overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
                if(count == (2 * binSize - 1))
                {
                    int32_t c = radixSort(overflowBuf + overflowBufId * binSize,
                                          count - (binSize - 1));
                    count = merge(hashTable + binId * binSize, binSize - 1,
                          overflowBuf + overflowBufId * binSize, c);
                    countArray[binId] = count;
                }
                if(count == (2 * binSize - 1))
                {
                    printf("ERROR! binId = %d, count = 2 * binSize - 1. Please use larger binSize or numBins\n", binId);
                    exit(1);
                }
                overflowBuf[overflowBufId * binSize + count - (binSize - 1)] = he;
                countArray[binId]++;
            }
        }
        //printf("hashTicks = %ld\n", hashTicks);
    }


    void finalize_insert()
    {
        if(coherence != INSERT)
        {
            printf("ERROR! The hashtable coherence is not set to INSERT at the moment. finalize_insert() can not be serviced\n");
            return;
        }
        int i, j;
        totalKeyCount = 0;
        for(i = 0; i < numBins; i++)
        {
            int32_t count = countArray[i];
            if(count < binSize)
            {
                count = radixSort(hashTable + i * binSize,
                        count);
            }
            else
            {
                int32_t overflowBufId;
                overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
                int32_t c = radixSort(overflowBuf + overflowBufId * binSize,
                        count - (binSize - 1));
                count = merge(hashTable + i * binSize, binSize - 1,
                        overflowBuf + overflowBufId * binSize, c);
            }
            int32_t firstBucketId = i << binShift;
            int32_t lastBucketId = firstBucketId + sortBufSize - 1;
            int32_t prevBucketId = firstBucketId - 1;
            int32_t k;
            int32_t y = std::min(count, binSize - 1);
            for(j = 0; j < y; j++)
            {
                int32_t bucketId = hashTable[i * binSize + j].bucketId;
                if(bucketId != prevBucketId)
                {
                    for(k = prevBucketId; k < bucketId; k++)
                        info_container[k + 1] = j;
                    prevBucketId = bucketId;
                }
            }
            int32_t overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                int32_t bucketId = overflowBuf[overflowBufId * binSize + j - (binSize - 1)].bucketId;
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

    size_t find(Key *keyArray, int32_t numKeys, uint32_t *findResult) const
    {
        if(coherence != COHERENT)
        {
            printf("ERROR! The hashtable is not coherent at the moment. find() can not be serviced\n");
            return 0ULL;
        }
        size_t foundCount = 0;
        int32_t PFD_INFO = 16;
        int32_t PFD_HASH = 8;
        int32_t i;
		int32_t hash_batch_size = 1024;
        hash_val_type bucketIdArray[2 * hash_batch_size];
		memset(bucketIdArray, 0, 2 * hash_batch_size * sizeof(hash_val_type));
		int32_t hash_mask = 2 * hash_batch_size - 1;

		int32_t hash_count = std::min(numKeys, hash_batch_size);
        hash_mod2(keyArray, hash_count, bucketIdArray);

//        for(i = 0; i < PFD_INFO; i++)
//        {
//            bucketIdArray[i] = hash(keyArray[i]) & bucketMask;
//        }
        for(i = 0; i < numKeys; i += hash_batch_size)
        {
            //startTick = __rdtsc();
			int32_t hash_first = i + hash_batch_size;
			if(hash_first > numKeys) hash_first = numKeys;
			int32_t hash_last = i + 2 * hash_batch_size;
			if(hash_last > numKeys) hash_last = numKeys;
			int32_t hash_count = hash_last - hash_first;
			hash_mod2(keyArray + hash_first, hash_count, bucketIdArray + ((hash_first) & hash_mask));
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
			int32_t last = i + hash_batch_size;
			if(last > numKeys) last = numKeys;
			int32_t j;

			for(j = i; j < last; j++)
			{
				int32_t f_info_bucketId = bucketIdArray[(j + PFD_INFO) & hash_mask];
				_mm_prefetch((const char *)(info_container + f_info_bucketId), _MM_HINT_T0);
				int32_t f_bucketId = bucketIdArray[(j + PFD_HASH) & hash_mask];
				int32_t f_binId = f_bucketId >> binShift;
				int32_t f_start = f_binId * binSize + info_container[f_bucketId];
				_mm_prefetch((const char *)(hashTable + f_start), _MM_HINT_T0);
				int32_t bucketId = bucketIdArray[j & hash_mask];
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

    size_t count(Key *keyArray, int32_t numKeys, uint8_t *countResult) const
    {
        if(coherence != COHERENT)
        {
            printf("ERROR! The hashtable is not coherent at the moment. count() can not be serviced\n");
            return 0ULL;
        }
        size_t foundCount = 0;
        int32_t PFD_INFO = 16;
        int32_t PFD_HASH = 8;
        int32_t i;
		int32_t hash_batch_size = 1024;
        hash_val_type bucketIdArray[2 * hash_batch_size];
		memset(bucketIdArray, 0, 2 * hash_batch_size * sizeof(hash_val_type));
		int32_t hash_mask = 2 * hash_batch_size - 1;

		int32_t hash_count = std::min(numKeys, hash_batch_size);
        hash_mod2(keyArray, hash_count, bucketIdArray);

//        for(i = 0; i < PFD_INFO; i++)
//        {
//            bucketIdArray[i] = hash(keyArray[i]) & bucketMask;
//        }

        for(i = 0; i < numKeys; i += hash_batch_size)
        {
            //startTick = __rdtsc();
			int32_t hash_first = i + hash_batch_size;
			if(hash_first > numKeys) hash_first = numKeys;
			int32_t hash_last = i + 2 * hash_batch_size;
			if(hash_last > numKeys) hash_last = numKeys;
			int32_t hash_count = hash_last - hash_first;
			hash_mod2(keyArray + hash_first, hash_count, bucketIdArray + ((hash_first) & hash_mask));
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
			int32_t last = i + hash_batch_size;
			if(last > numKeys) last = numKeys;
			int32_t j;


			for(j = i; j < last; j++)
			{
				int32_t f_info_bucketId = bucketIdArray[(j + PFD_INFO) & hash_mask];
				_mm_prefetch((const char *)(info_container + f_info_bucketId), _MM_HINT_T0);
				int32_t f_bucketId = bucketIdArray[(j + PFD_HASH) & hash_mask];
				int32_t f_binId = f_bucketId >> binShift;
				int32_t f_start = f_binId * binSize + info_container[f_bucketId];
				_mm_prefetch((const char *)(hashTable + f_start), _MM_HINT_T0);
				int32_t bucketId = bucketIdArray[j & hash_mask];
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

    void erase(Key *keyArray, int32_t numKeys)
    {
        if((coherence != COHERENT) && (coherence != ERASE))
        {
            printf("ERROR! The hashtable is not coherent at the moment. erase() can not be serviced\n");
            return;
        }
        coherence = ERASE;
        int32_t PFD_INFO = 16;
        int32_t PFD_HASH = 8;
        int32_t i;
		int32_t hash_batch_size = 1024;
        hash_val_type bucketIdArray[2 * hash_batch_size];
		memset(bucketIdArray, 0, 2 * hash_batch_size * sizeof(hash_val_type));
		int32_t hash_mask = 2 * hash_batch_size - 1;

		int32_t hash_count = std::min(numKeys, hash_batch_size);
        hash_mod2(keyArray, hash_count, bucketIdArray);

//        for(i = 0; i < PFD_INFO; i++)
//        {
//            bucketIdArray[i] = hash(keyArray[i]) & bucketMask;
//        }

        for(i = 0; i < numKeys; i += hash_batch_size)
        {
            //startTick = __rdtsc();
			int32_t hash_first = i + hash_batch_size;
			if(hash_first > numKeys) hash_first = numKeys;
			int32_t hash_last = i + 2 * hash_batch_size;
			if(hash_last > numKeys) hash_last = numKeys;
			int32_t hash_count = hash_last - hash_first;
			hash_mod2(keyArray + hash_first, hash_count, bucketIdArray + ((hash_first) & hash_mask));
            //endTick = __rdtsc();
            //hashTicks += (endTick - startTick);
			int32_t last = i + hash_batch_size;
			if(last > numKeys) last = numKeys;
			int32_t j;


			for(j = i; j < last; j++)
			{
				int32_t f_info_bucketId = bucketIdArray[(j + PFD_INFO) & hash_mask];
				_mm_prefetch((const char *)(info_container + f_info_bucketId), _MM_HINT_T0);
				int32_t f_bucketId = bucketIdArray[(j + PFD_HASH) & hash_mask];
				int32_t f_binId = f_bucketId >> binShift;
				int32_t f_start = f_binId * binSize + info_container[f_bucketId];
				_mm_prefetch((const char *)(hashTable + f_start), _MM_HINT_T0);
				int32_t bucketId = bucketIdArray[j & hash_mask];
				HashElement *he = find_internal(keyArray[j], bucketId);
				if(he != NULL)
					he->bucketId = -1;
			}
        }
    }

    size_t finalize_erase()
    {
        if(coherence != ERASE)
        {
            printf("ERROR! The hashtable coherence is not set to ERASE at the moment. finalize_erase() can not be serviced\n");
            return 0;
        }
        int32_t i;

        size_t eraseCount = totalKeyCount;
        totalKeyCount = 0;
        for(i = 0; i < numBins; i++)
        {
            int32_t count = countArray[i];

            int32_t p1, p2;
            int32_t y = std::min(count, binSize - 1);
            p1 = p2 = 0;
            for(p2 = 0; p2 < y; p2++)
            {
                HashElement he = hashTable[i * binSize + p2];
                if(he.bucketId != -1)
                {
                    hashTable[i * binSize + p1] = he;
                    p1++;
                }
            }
            int32_t overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; p2 < count; p2++)
            {
                HashElement he = overflowBuf[overflowBufId * binSize + p2 - (binSize - 1)];
                if(he.bucketId != -1)
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
            int32_t firstBucketId = i << binShift;
            int32_t lastBucketId = firstBucketId + sortBufSize - 1;
            int32_t prevBucketId = firstBucketId - 1;
            int32_t j, k;
            y = std::min(count, binSize - 1);
            for(j = 0; j < y; j++)
            {
                int32_t bucketId = hashTable[i * binSize + j].bucketId;
                if(bucketId != prevBucketId)
                {
                    for(k = prevBucketId; k < bucketId; k++)
                        info_container[k + 1] = j;
                    prevBucketId = bucketId;
                }
            }
            for(; j < count; j++)
            {
                int32_t bucketId = overflowBuf[overflowBufId * binSize + j - (binSize - 1)].bucketId;
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
        int32_t totalCount = 0;
        int32_t maxCount = 0;
        int32_t maxBin = -1;
        int i;

        int64_t kmerCountSum = 0;
        int32_t prevBucketId = -1;
        for(i = 0; i < numBins; i++)
        {
            int32_t j;
            int32_t count = countArray[i];
            if(count > maxCount)
            {
                maxCount = count;
                maxBin = i;
            }
            totalCount += count;
            int32_t y = std::min(count, binSize - 1);
            for(j = 0; j < y; j++)
            {
                HashElement he = hashTable[i * binSize + j];
                int32_t bucketId = he.bucketId;
                if(bucketId < prevBucketId)
                {
                    printf("ERROR! [%d,%d] prevBucketId = %d, bucketId = %d\n", i, j, prevBucketId, bucketId);
                    exit(0);
                }
                kmerCountSum += (he.val * he.val);
                prevBucketId = bucketId;
            }
            int32_t overflowBufId;
            overflowBufId = hashTable[i * binSize + binSize - 1].bucketId;
            for(; j < count; j++)
            {
                HashElement he = overflowBuf[overflowBufId * binSize + j - (binSize - 1)];
                int32_t bucketId = he.bucketId;
                if(bucketId < prevBucketId)
                {
                    printf("ERROR! [%d,%d] prevBucketId = %d, bucketId = %d\n", i, j, prevBucketId, bucketId);
                    exit(0);
                }
                kmerCountSum += (he.val * he.val);
                prevBucketId = bucketId;
            }
        }
        printf("kmerCountSum = %ld\n", kmerCountSum);
        for(i = 0; i < numBuckets; i++)
        {
            int32_t j;
            int32_t binId = i >> binShift;
            int32_t start = info_container[i];
            int32_t binId2 = (i + 1) >> binShift;
            int32_t end;
            if(binId == binId2)
                end = info_container[i + 1];
            else
                end = countArray[binId];

            if(end < start)
            {
                printf("ERROR! [%d] start = %d, end = %d, binId = %d, binId2 = %d\n", i, start, end, binId, binId2);
                exit(0);
            }
            for(j = start; j < end; j++)
            {
                int32_t bucketId;
                if(j < (binSize - 1))
                {
                    bucketId = hashTable[binId * binSize + j].bucketId;
                }
                else
                {
                    int32_t overflowBufId = hashTable[binId * binSize + binSize - 1].bucketId;
                    bucketId = overflowBuf[overflowBufId * binSize + j - (binSize - 1)].bucketId;
                }
                if(bucketId != i)
                {
                    printf("ERROR! [%d,%d] bucketId = %d\n", i, j, bucketId);
                    exit(0);
                }
            }
        }
        printf("Sanity check passed!\n");
        printf("totalCount = %d, maxCount = %d, maxBin = %d, curOverflowBufId = %d\n", totalCount, maxCount, maxBin, curOverflowBufId);

    }

    size_t size() const {
        return totalKeyCount;
    }

	inline hyperloglog64<Key, Hash<Key>, 12>& get_hll() {
		return this->hll;
	}

};
}  // namespace fsc
#endif /* KMERHASH_HASHMAP_RADIXSORT_HPP_ */
