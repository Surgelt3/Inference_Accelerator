#include "compiler.hpp"

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "../address_map_arm.h"

int open_physical(int);
void *map_physical(int, unsigned int, unsigned int);
void close_physical(int);
int unmap_physical(void *, unsigned int);

// const unsigned int LW_BRIDGE_BASE = 0xFF200000;


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

static uint64_t *usedBlocks;
static uint NUM_BLOCKS;
const static uint BLOCK_SIZE = sizeof(float) * 16;
struct DataInfo
{
  uint32_t ready : 1;
  uint32_t length : 31;
};

MemManager::MemManager()
{
  open_physical(fd);
  this->base = (uchar *)LW_BRIDGE_BASE;
  this->virt_addr = map_physical(fd, LW_BRIDGE_BASE, LW_BRIDGE_SPAN);
  this->constant0 = ((float *)(base + DATA_OFFSET)) + 0;
  this->constant1 = ((float *)(base + DATA_OFFSET)) + 1;
  *((float *)((uchar*)virt_addr + DATA_OFFSET) + 0) = 0.0;
  *((float *)((uchar*)virt_addr + DATA_OFFSET) + 1) = 1.0;
  NUM_BLOCKS = LW_BRIDGE_SPAN / BLOCK_SIZE;
  usedBlocks = (uint64_t *)calloc(sizeof(uint64_t), (NUM_BLOCKS / 64));
  mappedAddresses = ch_hashcreate(uchar *);
}
MemManager::~MemManager()
{
  unmap_physical(virt_addr, LW_BRIDGE_SPAN);
  close_physical(fd);
  free(usedBlocks);
  ch_hashfree(mappedAddresses);
}
void MemManager::replace(void *data, size_t N)
{
  uint requiredBlocks = (N + sizeof(DataInfo)) / BLOCK_SIZE + 1;
  int continuousBlocks=0;
  int startIndex=0;
  for(int i=0;i<NUM_BLOCKS*64;i++)
  {
    if (usedBlocks[i / 64] & (1 << (i % 64)))
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
  int offset = DATA_OFFSET + sizeof(float) * 2 + startIndex * BLOCK_SIZE;
  const uchar *destAddress = base + offset;
  ch_hashinsert(const uchar *, mappedAddresses, (size_t)data, destAddress);

  // start seperate thread?
  DataInfo info = {1, requiredBlocks};
  memcpy((uchar *)virt_addr + offset + sizeof(DataInfo), data, N);
  memcpy((uchar *)virt_addr + offset, &info, sizeof(DataInfo));
}
void MemManager::schedule(void *data, size_t N)
{
  replace(data, N);
}
void *MemManager::request(void *ref)
{
  uchar *requestedData = ch_hashget(uchar *, mappedAddresses, (size_t)ref);
  if (!requestedData)
    return NULL;
  volatile DataInfo info = *(DataInfo *)requestedData;
  while(!info.ready);

  return requestedData + sizeof(requestedData);
}
void MemManager::freeLocal()
{
  ch_hashclear(uchar *, mappedAddresses);
  memset(usedBlocks, 0, sizeof(uint64_t) * (NUM_BLOCKS / 64));
}