#ifndef CHEVAN_UTILS_ARRAY_H
#define CHEVAN_UTILS_ARRAY_H
#include "chevan_utils_min.h"

typedef struct ch_array
{
  void *_start, *_end, *_max;
} ch_array;

static inline ch_array ch_arrcreate(size_t size)
{
  ch_array a = {malloc(size), 0, 0};
  a._end = (uchar *)a._start + size;
  a._max = a._end;
  return a;
}
#define ch_arrcreate(type, size) ch_arrcreate(sizeof(type) * size)
static inline ch_array ch_arrstack(size_t maxSize)
{
  ch_array a = {malloc(maxSize), 0, 0};
  a._end = a._start;
  a._max = (uchar *)a._start + maxSize;
  return a;
}
#define ch_arrstack(type, size) ch_arrstack(sizeof(type) * size)
#define ch_arrlength(type, arr) (((uchar *)arr._end - (uchar *)arr._start) / sizeof(type))
static inline ch_array ch_arrcopy(ch_array arr)
{
  uint size = (uchar *)arr._end - (uchar *)arr._start;
  ch_array cp;
  cp._start = malloc(size);
  cp._end = (uchar *)cp._start + size;
  cp._max = cp._end;
  memcpy(cp._start,arr._start,size);
  return cp;
}
static inline void *ch_arrget(size_t _size, ch_array arr, int i)
{
  if (i >= 0)
    return (uchar *)arr._start + i * _size;
  else
    return (uchar *)arr._end + i * _size;
}
#define ch_arrgetp(type, arr, i) (((type *)ch_arrget(sizeof(type), arr, i)))
#define ch_arrget(type, arr, i) (*((type *)ch_arrget(sizeof(type), arr, i)))

#define ch_arrpush(type, arr, e)                 \
  if (arr._end != arr._max)                      \
  {                                              \
    *(type *)arr._end = e;                       \
    arr._end = (uchar *)arr._end + sizeof(type); \
  }
static inline void *ch_arrpop(size_t _size, ch_array arr)
{
  arr._end = (uchar *)arr._end - _size;
  return arr._end;
}
#define ch_arrpop(type, arr) (*(type *)((arr._start != arr._end) ? (ch_arrpop(sizeof(type), arr)) : (0)))
static inline ch_array ch_arrexpand(size_t _size, ch_array arr, size_t nsize)
{
  arr._start = realloc(arr._start, _size * nsize);
  arr._end = (uchar *)arr._start + nsize * _size;
  arr._max = arr._end;
  return arr;
}
#define ch_arrexpand(type, arr, nSize) ch_arrexpand(sizeof(type), arr, nSize)
static inline ch_array ch_arrfree(ch_array arr)
{
  free(arr._start);
  arr._end = 0;
  arr._max = 0;
  return arr;
}

#define ch_hash_STRING_SIZE 256
typedef struct ch_hashPair
{
  uchar key[ch_hash_STRING_SIZE];
  // 1 for taken data
  uchar flag;
  void *data;
} ch_hashPair;
static const void *ch_hash_NOTFOUND = 0;
typedef struct ch_hash
{
  ch_hashPair *arr;
  size_t size;
  size_t (*hash)(void *);
} ch_hash;

static inline ch_hash ch_hashcreatesize(size_t _tsize, size_t size)
{
  ch_hash h;
  h.arr = (ch_hashPair *)calloc(size, sizeof(ch_hashPair) + _tsize - sizeof(void *));
  h.size = size;
  h.hash = 0;
  return h;
}
#define ch_hashcreatesize(_type_, size) ch_hashcreatesize(sizeof(_type_), size)
#define ch_hashcreate(_type_) ch_hashcreatesize(_type_, 1024)
static inline size_t ch_stringHash(const char *h)
{
  size_t s = 1;
  uint64_t tmp = 0;
  uint n = 0;
  for (const char *c = h; *c; c++)
  {
    tmp = (tmp << 8) | *c;
    if (++n >= sizeof(uint64_t))
    {
      n = 0;
      s *= tmp;
      tmp = 0;
    }
  }
  s *= tmp ? tmp : 1;
  return s;
}
static void *ch_hashgetInt(ch_hash h, size_t k, size_t _size)
{
  if (!h.arr || !h.size)
    return &ch_hash_NOTFOUND;
  size_t _k = h.hash ? h.hash(&k) : k % h.size;
  for (int i = 0; i < h.size; i++)
  {
    ch_hashPair *p = (ch_hashPair *)(((uchar *)h.arr) + (i + _k) * (sizeof(ch_hashPair) + _size - sizeof(void *)));
    if (!memcmp(&k, p->key, sizeof(k)))
      return ((uchar *)p) + offsetof(ch_hashPair, data);
    if (!(p->flag & 0x1))
      break;
  }
  return &ch_hash_NOTFOUND;
}
static void *ch_hashgetStr(ch_hash h, const char *k, size_t _size)
{
  if (!h.arr || !h.size)
    return &ch_hash_NOTFOUND;
  size_t _k = h.hash ? h.hash((void *)k) : ch_stringHash(k) % h.size;
  for (int i = 0; i < h.size; i++)
  {
    ch_hashPair *p = (ch_hashPair *)(((uchar *)h.arr) + ((i + _k) % h.size) * (sizeof(ch_hashPair) - sizeof(void *) + _size));
    if (!memcmp(k, p->key, MIN(strlen(k), ch_hash_STRING_SIZE)))
      return ((uchar *)p) + offsetof(ch_hashPair, data);
    if (!(p->flag & 0x1))
      break;
  }
  return &ch_hash_NOTFOUND;
}
#ifndef __cplusplus
#define ch_hashget(type, hash, key) (*((type *)_Generic((key), \
    int: ch_hashgetInt,                                        \
    uint: ch_hashgetInt,                                       \
    long: ch_hashgetInt,                                       \
    uint64_t: ch_hashgetInt,                                   \
    char *: ch_hashgetStr,                                     \
    const char *: ch_hashgetStr)(hash, (key), sizeof(type))))
#else
static inline void *ch_hashget(ch_hash h, size_t key, size_t _size)
{
  return ch_hashgetInt(h, key, _size);
}
static inline void *ch_hashget(ch_hash h, const char *key, size_t _size)
{
  return ch_hashgetStr(h, key, _size);
}
#define ch_hashgetp(type, hash, key) ((type *)ch_hashget(hash, key, sizeof(type)))
#define ch_hashget(type, hash, key) (*(type *)ch_hashget(hash, key, sizeof(type)))
#endif

static void *ch_hashinsInt(ch_hash h, size_t k, size_t _size)
{
  size_t _k = h.hash ? h.hash(&k) : k % h.size;
  for (int i = 0; i < h.size; i++)
  {
    ch_hashPair *p = (ch_hashPair *)(((uchar *)h.arr) + ((i + _k) % h.size) * (sizeof(ch_hashPair) + _size - sizeof(void *)));
    if (p->flag & 0x1)
      continue;
    p->flag = 0x1;

    memset(p->key, 0, sizeof(p->key));
    memcpy(p->key, &k, sizeof(k));
    return ((uchar *)p) + offsetof(ch_hashPair, data);
  }
  return 0;
}
static void *ch_hashinsStr(ch_hash h, const char *k, size_t _size)
{
  size_t _k = h.hash ? h.hash((void *)k) : ch_stringHash(k) % h.size;
  for (int i = 0; i < h.size; i++)
  {
    ch_hashPair *p = (ch_hashPair *)(((uchar *)h.arr) + ((i + _k) % h.size) * (sizeof(ch_hashPair) - sizeof(void *) + _size));
    if (p->flag & 0x1)
      continue;
    p->flag = 0x1;
    memset(p->key, 0, sizeof(p->key));
    memcpy(p->key, k, MIN(strlen(k),ch_hash_STRING_SIZE));
    return ((uchar *)p) + offsetof(ch_hashPair, data);
  }
  return 0;
}

#ifndef __cplusplus
#define ch_hashinsert(type, hash, key, e)                           \
  do                                                                \
  {                                                                 \
    type _e = e;                                                    \
    memcpy(_Generic((key),                                          \
           int: ch_hashinsInt,                                      \
           uint: ch_hashinsInt,                                     \
           long: ch_hashinsInt,                                     \
           uint64_t: ch_hashinsInt,                                 \
           char *: ch_hashinsStr,                                   \
           const char *: ch_hashinsStr)(hash, (key), sizeof(type)), \
           &_e, sizeof(type));                                      \
  } while (0)
#else
static inline void *ch_hashinsert(ch_hash h, size_t key, size_t _size)
{
  return ch_hashinsInt(h, key, _size);
}
static inline void *ch_hashinsert(ch_hash h, const char *key, size_t _size)
{
  return ch_hashinsStr(h, key, _size);
}
#define ch_hashinsert(type, hash, key, e) (*(type *)ch_hashinsert(hash, key, sizeof(type))) = e
#endif

static size_t ch_hashlength(size_t _size, ch_hash h)
{
  size_t s = 0;
  for (int i = 0; i < h.size; i++)
  {
    ch_hashPair *p = (ch_hashPair *)(((uchar *)h.arr) + i * (sizeof(ch_hashPair) - sizeof(void *) + _size));
    s += (p->flag & 0x1) ? 1 : 0;
  }
  return s;
}
#define ch_hashlength(type, hash) ch_hashlength(sizeof(type), hash)

#endif