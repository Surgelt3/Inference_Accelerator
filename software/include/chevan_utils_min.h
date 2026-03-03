#pragma once
#ifndef CHEVAN_UTILS_MIN_H
#define CHEVAN_UTILS_MIN_H
#define CHEVAN_UTILS_VERSION "3.2.0"
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#define CHVAL 0x86AC
#define CHARG_EVAL(F,...) F(__VA_ARGS__)
static inline void membuild(void *dst, const void *src, const size_t size)
{
  if(src)
    memcpy(dst, src, size);
  else
    memset(dst, 0, size);
}
#define membuild(type, name, data) \
  type name;                       \
  membuild(&name, data, sizeof(name));
#define mallocArr(type, size) malloc(sizeof(type) * (size))
#define sizeofArr(arr) (sizeof(arr) / sizeof(arr[0]))
#define fcompare(a, b) ((a) - (b) < 0.00001 && (b) - (a) < 0.00001)
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

#define cherrorCodeMessage(code,...) {int _code=code;if(_code){fprintf(stderr,__VA_ARGS__);exit(_code);}}
#define chprinterr(...) cherrorCodeMessage(1,__VA_ARGS__)
#define chstop(message) cherrorCodeMessage(1, "%s:%d: %s\n",__FILE__,__LINE__,message)
#define cherrorCode(code) cherrorCodeMessage(code, "%s:%d: Process exited with error code %d.\n", __FILE__, __LINE__, _code)
#define chferror(fname, ...) cherrorCodeMessage(fname(__VA_ARGS__),"Function %s exited with code %d.\n", #fname, _code)
#define chiniterr(var) cherrorCodeMessage(var == CHVAL, "%s:%d: Variable " #var " was not changed and still has the value 0x%X.\n", __FILE__, __LINE__, CHVAL)
#define chassert(cond,reason) if(!(cond)){cherrorCodeMessage(1,"%s:%d: Assertion "#cond" has failed.\n%s\n",__FILE__,__LINE__,reason)}
#define chfpass(fname,...) {int _code=fname(__VA_ARGS__);if(_code)return _code;}

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ullong;
typedef long long llong;
typedef char string128[128];


#define BitmapGenType(type, name, size) type name[size / (sizeof(type) * 8) + 1];
#define BitmapGen(name, size) BitmapGenType(unsigned int, name, size)
#define BitmapCheck(a, i) ((a[i / (sizeof(i) * 8)] >> (i % (sizeof(i) * 8))) & 1)

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#define SETMIN(set,a) (set)=MIN(set,a)
#define SETMAX(set,a) (set)=MAX(set,a) 
enum Cardinal8dir
{
  CENTER = 0,
  NORTH,
  NORTH_EAST,
  EAST,
  SOUTH_EAST,
  SOUTH,
  SOUTH_WEST,
  WEST,
  NORTH_WEST
};

typedef struct color3_t
{
  union
  {
    struct
    {
      union
      {
        uint8_t x, r;
      };
      union
      {
        uint8_t y, g;
      };
      union
      {
        uint8_t z, b;
      };
    };
    struct
    {
      uint8_t r, g, b;
    } rgb;
    struct
    {
      uint8_t y, cb, cr;
    } ycc;
  };
} color3_t;
typedef struct color4_t
{
  union
  {
    uint8_t x, r;
  };
  union
  {
    uint8_t y, g;
  };
  union
  {
    uint8_t z, b;
  };
  union
  {
    uint8_t w, a;
  };
} color4_t;
static color3_t initColor3(uint8_t x, uint8_t y, uint8_t z)
{
  color3_t c = {x, y, z};
  return c;
}
#define color3_t(x, y, z) initColor3(x, y, z)
static color4_t initColor4(uint8_t x, uint8_t y, uint8_t z, uint8_t w)
{
  color4_t c = {x, y, z, w};
  return c;
}
#define color4_t(x, y, z, w) initColor4(x, y, z, w)

static void *ch_bufferFile(const char *file, void **targetBuffer, size_t *bufferLength)
{
  FILE *fp = fopen(file, "rb");
  if (fp)
  {
    fseek(fp, 0, SEEK_END);
    size_t _bufferLength = CHVAL;
    if (!bufferLength)
      bufferLength = &_bufferLength;
    *bufferLength = ftell(fp);

    fseek(fp, 0, SEEK_SET);
    *targetBuffer = malloc(*bufferLength);
    fread(*targetBuffer, *bufferLength, 1, fp);
    fclose(fp);
    return *targetBuffer;
  }
  return 0;
}
static int ch_writeFile(const char *file, void *buffer, size_t length)
{
  FILE *fp = fopen(file, "wb");
  if (fp)
  {
    fwrite(buffer, 1, length, fp);
    fclose(fp);
    return 0;
  }
  return 1;
}

static void ch_printMem(const uchar *mem, const size_t length)
{
  for (int i = 0; i < length; i++)
  {
    printf("%02X ", mem[i]);
  }
  printf("\n");
}

#ifdef CHEVAN_UTILS_BYTE_TYPEDEF
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef double f64;
#endif

#endif