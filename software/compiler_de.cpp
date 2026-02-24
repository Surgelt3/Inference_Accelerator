#include "compiler.hpp"

#include <queue>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "../address_map_arm.h"

#include <thread>

int open_physical(int);
void *map_physical(int, unsigned int, unsigned int);
void close_physical(int);
int unmap_physical(void *, unsigned int);

#if 0
int main(void)
{
  volatile int *LEDR_ptr; // virtual address pointer to red LEDs
  int fd = -1;            // used to open /dev/mem
  void *LW_virtual;       // physical addresses for light-weight bridge
                          // Create virtual memory access to the FPGA light-weight bridge
  if ((fd = open_physical(fd)) == -1)
    return (-1);
  if (!(LW_virtual = map_physical(fd, LW_BRIDGE_BASE, LW_BRIDGE_SPAN)))
     return (-1);
  // Set virtual address pointer to I/O port
  LEDR_ptr = (int *)(LW_virtual + LEDR_BASE);
  *LEDR_ptr = *LEDR_ptr + 1; // Add 1 to the I/O register
  unmap_physical(LW_virtual, LW_BRIDGE_SPAN);
  close_physical(fd);
  return 0;
}
#endif

// #define debugprint(...) chprintln(__VA_ARGS__)
#ifndef debugprint
#define debugprint(...)
#endif

int open_physical(int fd)
{
  if (fd == -1) // check if already open
    if ((fd = open("/dev/mem", (O_RDWR | O_SYNC))) == -1)
    {
      printf("ERROR: could not open \"/dev/mem\"...\n");
      return (-1);
    }
  return fd;
}
/* Close /dev/mem to give access to physical addresses */
void close_physical(int fd)
{
  close(fd);
}
/* Establish a virtual address mapping for the physical addresses starting
 * at base and extending by span bytes */
void *map_physical(int fd, unsigned int base, unsigned int span)
{
  void *virtual_base;
  // Get a mapping from physical addresses to virtual addresses
  virtual_base = mmap(NULL, span, (PROT_READ | PROT_WRITE), MAP_SHARED,
                      fd, base);
  if (virtual_base == MAP_FAILED)
  {
    printf("ERROR: mmap() failed...\n");
    close(fd);
    return (NULL);
  }
  return virtual_base;
}
/* Close the previously-opened virtual address mapping */
int unmap_physical(void *virtual_base, unsigned int span)
{
  if (munmap(virtual_base, span) != 0)
  {
    printf("ERROR: munmap() failed...\n");
    return (-1);
  }
  return 0;
}

#define DATA_OFFSET 512
#define TEXT_OFFSET 0
#define DATA_OUT_SIZE (sizeof(float) * DATA_OUT_LENGTH)
// #define NUM_BLOCKS 192
#define NUM_BLOCKS 6
#define DATA_INFO_MAGIC 0x86AC
static uint usedBlocks[NUM_BLOCKS];
// static uint blocksQueueIndex = 0;
// static uchar* blocksQueue[NUM_BLOCKS];
static std::queue<uchar*>blocksQueue;
struct __attribute__((packed)) DataInfo
{
  uint32_t lock : 1;
  uint32_t ready : 1;
  uint32_t length : 14;
  uint16_t magic : 16;
};
struct DataInfoOut
{
  uint32_t ready : 1;
  uint32_t _empty : 15;
  uint16_t magic : 16;
};

static size_t hashMapFunc(void *a)
{
  return (*(size_t*)a)/4;
}
MemManager::MemManager()
{
  open_physical(fd);
  this->shared_addr = map_physical(fd, LW_BRIDGE_BASE, LW_BRIDGE_SPAN);
  this->outPtr = (float *)((uchar *)shared_addr + DATA_OFFSET);
  *((float *)((uchar *)shared_addr + DATA_OFFSET) + 0) = 0.0;
  *((float *)((uchar *)shared_addr + DATA_OFFSET) + 1) = 1.0;

  this->base = (uchar *)LW_BRIDGE_BASE;
  this->outPtr = (float *)((uchar *)shared_addr + DATA_OFFSET);
  memset(usedBlocks, 0, sizeof(uint) * NUM_BLOCKS);
  // memset(blocksQueue, 0, sizeof(uchar *) * NUM_BLOCKS);
  blocksQueue=std::queue<uchar*>();
  mappedAddresses = ch_hashcreate(uchar *);
  // mappedAddresses.hash=hashMapFunc;
  BLOCK_SIZE = (LW_BRIDGE_SPAN - DATA_OFFSET - DATA_OUT_SIZE) / NUM_BLOCKS;
  BLOCK_SIZE = BLOCK_SIZE / sizeof(float) * sizeof(float);
  this->maxSize = BLOCK_SIZE * NUM_BLOCKS;
}
MemManager::MemManager(uchar* virt)
{
  this->base = NULL;
  this->shared_addr = virt;
  this->outPtr = (float *)((uchar *)shared_addr + DATA_OFFSET);
  *((float *)((uchar *)shared_addr + DATA_OFFSET) + 0) = 0.0;
  *((float *)((uchar *)shared_addr + DATA_OFFSET) + 1) = 1.0;
  memset(usedBlocks, 0, sizeof(uint) * NUM_BLOCKS);
  // memset(blocksQueue, 0, sizeof(uchar *) * NUM_BLOCKS);
  blocksQueue = std::queue<uchar *>();
  mappedAddresses = ch_hashcreate(uchar *);
  mappedAddresses.hash = hashMapFunc;
  BLOCK_SIZE = (LW_BRIDGE_SPAN - DATA_OFFSET - DATA_OUT_SIZE) / NUM_BLOCKS;
  BLOCK_SIZE = BLOCK_SIZE / sizeof(float) * sizeof(float);
  this->maxSize = BLOCK_SIZE*NUM_BLOCKS;
}

MemManager::~MemManager()
{
  if (this->base)
  {
    unmap_physical(shared_addr, LW_BRIDGE_SPAN);
    close_physical(fd);
  }
  ch_hashfree(mappedAddresses);
}

#if 1
static uint PC=0;
void MemManager::writeInstr(uint32_t instruction)
{
  ((uint32_t *)shared_addr)[PC] = instruction;
  PC = (PC + 1) % (DATA_OFFSET / sizeof(uint32_t));
}

void MemManager::freeLastAdded()
{
  debugprint("Auto free");
  uchar*addr=0;
  volatile DataInfo *info = 0;
  while (!info)
  {
    assert(!blocksQueue.empty());
    addr = blocksQueue.front();
    blocksQueue.pop();
    info = ch_hashget(DataInfo *, mappedAddresses, (size_t)addr);
    if(info && info->lock)
    {
      info = 0;
      blocksQueue.push(addr);
      continue;
    }
    if (info && info->length != 0)
      assert(info->magic == DATA_INFO_MAGIC);
  }
  freeBuffer(addr);
}

float* MemManager::readOut()
{
  if (!outPtr)
    return NULL;
  volatile DataInfoOut *info = (DataInfoOut *)outPtr;
  if (info->magic != DATA_INFO_MAGIC)
    return NULL;
  while(!info->ready);
  return outPtr + sizeof(DataInfo);
}
void MemManager::readComplete()
{
  if (!outPtr)
    return;
  volatile DataInfoOut *info = (DataInfoOut *)outPtr;
  if (info->magic != DATA_INFO_MAGIC)
    return;
  info->ready = 0;
  return;
}

void MemManager::schedule(void *data, size_t N)
{
  assert(N > 0);
  uchar *previousData = ch_hashget(uchar *, mappedAddresses, (size_t)data);
  uint requiredBlocks = (N + sizeof(DataInfo)) / BLOCK_SIZE + 1;
  assert(requiredBlocks <= NUM_BLOCKS);
  assert(requiredBlocks==1);// faster this way
  if (previousData && previousData != ch_hash_NOTFOUND)
  {
    DataInfo *previousInfo = (DataInfo *)previousData;
    if (previousInfo->magic == DATA_INFO_MAGIC &&
        (previousInfo->lock || previousInfo->ready))
    {
      if (previousInfo->length < requiredBlocks)
        freeBuffer(data);
      else
        return;
    }
  }
  debugprint("schedule transfer for ",data);
  int startIndex = -1;
  do
  {
    int continuousBlocks = 0;
    for (int i = 0; i <= NUM_BLOCKS - requiredBlocks; i++)
    {
      if (!usedBlocks[i])
      {
        if (continuousBlocks == 0)
          startIndex = i;
        ++continuousBlocks;
        if (continuousBlocks >= requiredBlocks)
          break;
      }
      else
      {
        continuousBlocks = 0;
      }
    }
    if (startIndex == -1)
      freeLastAdded();
  } while (startIndex == -1);

  for (int i = 0; i < requiredBlocks; i++)
  {
    usedBlocks[startIndex + i] = 1;
  }
  int offset = DATA_OFFSET + startIndex * BLOCK_SIZE;
  uchar *destAddress = (uchar*)shared_addr + offset;
  ch_hashinsert(const uchar *, mappedAddresses, (size_t)data, destAddress);
  assert((destAddress - (uchar *)shared_addr - DATA_OFFSET) / BLOCK_SIZE == startIndex);
  assert(startIndex + requiredBlocks <= NUM_BLOCKS);
  assert(destAddress==ch_hashget(uchar*,mappedAddresses,(size_t)data));

  // assert(blocksQueue[blocksQueueIndex] == 0);
  // while(blocksQueue[blocksQueueIndex] != 0)
  // {
  //   blocksQueueIndex = (blocksQueueIndex - 1 + NUM_BLOCKS) % NUM_BLOCKS;
  // }
  // blocksQueue[blocksQueueIndex] = (uchar *)data;
  // blocksQueueIndex = (blocksQueueIndex + 1) % NUM_BLOCKS;
  blocksQueue.push((uchar *)data);

  DataInfo info;
  info.ready = 0;
  info.lock = 1;
  info.magic = DATA_INFO_MAGIC;
  info.length = requiredBlocks;
  memcpy(destAddress, &info, sizeof(DataInfo));
  debugprint("scheduled ",data," at address ",(uchar *)shared_addr + offset);
  info.ready = 1;
  info.lock = 0;
  assert(N + sizeof(DataInfo) < BLOCK_SIZE * requiredBlocks);
  if (N + sizeof(DataInfo) > BLOCK_SIZE)
  {
    DataInfo *nextBlock = (DataInfo *)(destAddress + BLOCK_SIZE);
    assert(nextBlock->magic != DATA_INFO_MAGIC);
  }
  memcpy(destAddress + sizeof(DataInfo), data, N);
  memcpy(destAddress, &info, sizeof(DataInfo));
  debugprint("completed write for ", destAddress);
}
void MemManager::lock(void*data)
{
  uchar *requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)data);
  if (!requestedData || requestedData == ch_hash_NOTFOUND)
    return;
  volatile DataInfo *info = (DataInfo *)requestedData;
  if ((info->magic != DATA_INFO_MAGIC) || (!info->length))
    return;
  info->lock=1;
}
void *MemManager::get(void *data){
  uchar *requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)data);
  if (!requestedData || requestedData == ch_hash_NOTFOUND)
    return NULL;
  volatile DataInfo *info = (DataInfo *)requestedData;
  if ((info->magic != DATA_INFO_MAGIC) || (!info->length))
    return NULL;
  return requestedData + sizeof(DataInfo);
}
void *MemManager::request(void *ref, size_t N)
{
  debugprint("request for ", ref);
  uchar *requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)ref);
  if (!requestedData || requestedData == ch_hash_NOTFOUND)
  {
    schedule(ref, N);
    requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)ref);
    return requestedData + sizeof(DataInfo);
  }
  volatile DataInfo *info = (DataInfo *)requestedData;
  if ((info->magic != DATA_INFO_MAGIC) || (!info->length) || (!info->lock && !info->ready))
  {
    if (N)
      schedule(ref, N);
    else
      return NULL;
  }

  while(!info->ready)
  {
    requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)ref);
    info = (DataInfo *)requestedData;
  }
  debugprint("request complete for ", ref);

  return requestedData + sizeof(DataInfo);
}
void *MemManager::use(void*ref,size_t N)
{
  void *ptr = request(ref, N);
  DataInfo *info = ch_hashget(DataInfo *, mappedAddresses, (size_t)ref);
  debugprint("locking ressource for ",ref);
  info->lock = 1;
  return ptr;
}
void MemManager::release(void*data)
{
  uchar *requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)data);
  if (!requestedData || requestedData == ch_hash_NOTFOUND)
    return;
  DataInfo *info = (DataInfo *)requestedData;
  if ((info->magic != DATA_INFO_MAGIC) || (!info->length))
  {
    chprintln("bad address");
    return;
  }
  info->lock = 0;
}
void MemManager::freeBuffer(void*data)
{
  uchar *requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)data);
  if (!requestedData || requestedData == ch_hash_NOTFOUND)
    return;
  DataInfo *info = (DataInfo *)requestedData;
  if ((info->magic != DATA_INFO_MAGIC) || (!info->length))
    return;
  while (info->lock)
  {
    info = (DataInfo *)requestedData;
  }
  uint index = (requestedData - (uchar*)shared_addr - DATA_OFFSET) / BLOCK_SIZE;
  assert(index + info->length <= NUM_BLOCKS);
  for (int i = 0; i < info->length; i++)
    usedBlocks[index + i] = 0;
  info->length = 0;
  info->magic = 0;
  ch_hashget(uchar *, mappedAddresses, (size_t)data) = 0;
  ch_hashrem(uchar *, mappedAddresses, (size_t)data);
  for(int i=0;i<blocksQueue.size();i++)
  {
    if(blocksQueue.front()!=data)
    {
      blocksQueue.push(blocksQueue.front());
    }
    blocksQueue.pop();
  }
}
void MemManager::freeAll()
{
  ch_hashclear(uchar *, mappedAddresses);
  memset(usedBlocks, 0, sizeof(uint) * NUM_BLOCKS);
  blocksQueue=std::queue<uchar*>();
}
#else
void MemManager::schedule(void *data, size_t N)
{
  // do nothing lol
}
static int currentBlock = 0;
void *MemManager::request(void *data, size_t N)
{
  uint requiredBlocks = (N + sizeof(DataInfo)) / BLOCK_SIZE + 1;
  assert(requiredBlocks == 1);
  int offset = DATA_OFFSET + currentBlock * BLOCK_SIZE;
  uchar *destAddress = (uchar *)shared_addr + offset;
  DataInfo info;
  info.ready = 0;
  info.lock = 1;
  info.magic = DATA_INFO_MAGIC;
  info.length = requiredBlocks;
  memcpy(destAddress, &info, sizeof(DataInfo));
  memcpy(destAddress + sizeof(DataInfo), data, N);
  info.ready = 1;
  info.lock = 0;
  memcpy(destAddress, &info, sizeof(DataInfo));
}
void *MemManager::use(void *data, size_t N)
{
  request(data, N);
}

#endif