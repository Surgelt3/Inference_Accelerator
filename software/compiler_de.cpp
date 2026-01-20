#include "compiler.hpp"

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
#define NUM_BLOCKS 12
#define DATA_INFO_MAGIC 0x86AC
static uint usedBlocks[NUM_BLOCKS];
static uint blocksQueueIndex = 0;
static uchar* blocksQueue[NUM_BLOCKS];
struct DataInfo
{
  uint16_t magic : 16;
  uint32_t ready : 1;
  uint32_t lock : 1;
  uint32_t length : 14;
};
struct DataInfoOut
{
  uint16_t magic : 16;
  uint32_t ready : 1;
  uint32_t _empty : 15;
};

MemManager::MemManager()
{
  open_physical(fd);
  this->virt_addr = map_physical(fd, LW_BRIDGE_BASE, LW_BRIDGE_SPAN);
  this->outPtr = (float *)((uchar *)virt_addr + DATA_OFFSET);
  *((float *)((uchar *)virt_addr + DATA_OFFSET) + 0) = 0.0;
  *((float *)((uchar *)virt_addr + DATA_OFFSET) + 1) = 1.0;

  this->base = (uchar *)LW_BRIDGE_BASE;
  this->outPtr = (float *)((uchar *)virt_addr + DATA_OFFSET);
  this->constant0 = ((float *)(base + DATA_OFFSET + DATA_OUT_SIZE)) + 0;
  this->constant1 = ((float *)(base + DATA_OFFSET + DATA_OUT_SIZE)) + 1;
  memset(usedBlocks, 0, sizeof(uint) * NUM_BLOCKS);
  memset(blocksQueue, 0, sizeof(uchar *) * NUM_BLOCKS);
  mappedAddresses = ch_hashcreate(uchar *);
  BLOCK_SIZE = (LW_BRIDGE_SPAN - DATA_OFFSET - sizeof(float) * 2 - DATA_OUT_SIZE) / NUM_BLOCKS;
}
MemManager::MemManager(uchar* base)
{
  this->base = base;
  this->virt_addr = base;
  this->outPtr = (float *)((uchar *)virt_addr + DATA_OFFSET);
  this->constant0 = ((float *)(base + DATA_OFFSET + DATA_OUT_SIZE)) + 0;
  this->constant1 = ((float *)(base + DATA_OFFSET + DATA_OUT_SIZE)) + 1;
  *((float *)((uchar *)virt_addr + DATA_OFFSET) + 0) = 0.0;
  *((float *)((uchar *)virt_addr + DATA_OFFSET) + 1) = 1.0;
  memset(usedBlocks, 0, sizeof(uint) * NUM_BLOCKS);
  memset(blocksQueue, 0, sizeof(uchar *) * NUM_BLOCKS);
  mappedAddresses = ch_hashcreate(uchar *);
  BLOCK_SIZE = (LW_BRIDGE_SPAN - DATA_OFFSET - sizeof(float) * 2 - DATA_OUT_SIZE) / NUM_BLOCKS;
}

MemManager::~MemManager()
{
  if (this->base!=this->virt_addr)
  {
    unmap_physical(virt_addr, LW_BRIDGE_SPAN);
    close_physical(fd);
  }
  ch_hashfree(mappedAddresses);
}

static uint PC=0;
void MemManager::writeInstr(uint32_t instruction)
{
  ((uint32_t*)virt_addr)[PC]=instruction;
  PC = (PC + 1) % (DATA_OFFSET / sizeof(uint32_t));
}

void MemManager::freeLastAdded()
{
  for (int i = 0, index = blocksQueueIndex; i < NUM_BLOCKS; index = (++i) % NUM_BLOCKS)
  {
    if(blocksQueue[index])
    {
      volatile DataInfo *info = ch_hashget(DataInfo *, mappedAddresses, (size_t)blocksQueue[index]);
      if(info->ready && !info->lock)
      {
        freeBuffer(blocksQueue[index]);
        break;
      }
    }
  }
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
  uchar *previousData = ch_hashget(uchar *, mappedAddresses, (size_t)data);
  uint requiredBlocks = (N + sizeof(DataInfo)) / BLOCK_SIZE + 1;
  if (!previousData && previousData != ch_hash_NOTFOUND)
  {
    volatile DataInfo *previousInfo = (DataInfo *)previousData;
    if (previousInfo->magic == DATA_INFO_MAGIC && previousInfo->length == requiredBlocks)
      return;
  }
  int startIndex = -1;
  do
  {
    int continuousBlocks = 0;
    for (int i = 0; i < NUM_BLOCKS; i++)
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
  int offset = DATA_OFFSET + sizeof(float) * 2 + startIndex * BLOCK_SIZE;
  const uchar *destAddress = base + offset;
  ch_hashinsert(const uchar *, mappedAddresses, (size_t)data, destAddress);

  blocksQueue[blocksQueueIndex] = (uchar *)data;
  blocksQueueIndex = (blocksQueueIndex + 1) % NUM_BLOCKS;

  DataInfo info;
  info.ready = 0;
  info.lock = 0;
  info.magic = DATA_INFO_MAGIC;
  info.length = requiredBlocks;
  memcpy((uchar *)virt_addr + offset, &info, sizeof(DataInfo));
  std::thread t = std::thread(
      [this, offset, data, N, requiredBlocks]()
      {
        DataInfo info;
        info.ready = 1;
        info.lock = 0;
        info.magic = DATA_INFO_MAGIC;
        info.length = requiredBlocks;
        memcpy((uchar *)virt_addr + offset + sizeof(DataInfo), data, N);
        memcpy((uchar *)virt_addr + offset, &info, sizeof(DataInfo));
      });
  t.detach();
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
void *MemManager::request(void *ref, size_t N)
{
  uchar *requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)ref);
  if (!requestedData || requestedData == ch_hash_NOTFOUND)
  {
    schedule(ref, N);
    requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)ref);
    return requestedData + sizeof(DataInfo);
  }
  volatile DataInfo *info = (DataInfo *)requestedData;
  if ((info->magic != DATA_INFO_MAGIC) || (!info->length))
  {
    if (N)
      schedule(ref, N);
    else
      return NULL;
  }

  while(!info->ready);

  return requestedData + sizeof(DataInfo);
}
void *MemManager::use(void*ref,size_t N)
{
  void*ptr=request(ref,N);
  volatile DataInfo *info = ch_hashget(DataInfo *, mappedAddresses, (size_t)ref);
  info->lock = 1;
  return ptr;
}
void MemManager::release(void*data)
{
  uchar *requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)data);
  if (!requestedData || requestedData == ch_hash_NOTFOUND)
    return;
  volatile DataInfo *info = (DataInfo *)requestedData;
  if ((info->magic != DATA_INFO_MAGIC) || (!info->length))
    return;
  info->lock = 0;
}
void MemManager::freeBuffer(void*data)
{
  uchar *requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)data);
  if (!requestedData || requestedData == ch_hash_NOTFOUND)
    return;
  volatile DataInfo *info = (DataInfo *)requestedData;
  if ((info->magic != DATA_INFO_MAGIC) || (!info->length))
    return;
  while(info->lock);
  uint index = (requestedData - base - DATA_OFFSET - sizeof(float) * 2) / BLOCK_SIZE;
  for (int i = 0; i < info->length; i++)
    usedBlocks[index + i] = 0;
  info->length = 0;
  info->magic = 0;
  for (int i = 0; i < NUM_BLOCKS; i++)
  {
    if (blocksQueue[i] == data)
      blocksQueue[i] = 0;
  }
  ch_hashget(uchar *, mappedAddresses, (size_t)data) = 0;
}
void MemManager::freeAll()
{
  ch_hashclear(uchar *, mappedAddresses);
  memset(usedBlocks, 0, sizeof(uint) * NUM_BLOCKS);
  memset(blocksQueue, 0, sizeof(uchar*) * NUM_BLOCKS);
  blocksQueueIndex = 0;
}