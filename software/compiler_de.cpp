#include "compiler.hpp"

#include <queue>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "../address_map_arm.h"
#include <ARM_A9_HPS_arm_a9_0.h>

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

#define OUT_ADDRESS 
static int fd = -1;
static volatile int writeReady = 1;
static std::thread resultsThread;
MemManager::MemManager()
{
  this->base = (uchar *)NPU_TOP_0_AVS_WRITE_BASE;
  open_physical(fd);
  this->shared_addr = map_physical(fd, NPU_TOP_0_AVS_WRITE_BASE, NPU_TOP_0_AVS_WRITE_SPAN);
  this->outPtr = (uint32_t*) map_physical(fd, NPU_TOP_0_AVS_READ_BASE, NPU_TOP_0_AVS_READ_SPAN);
  this->resetPtr = (uint32_t *)map_physical(fd, NPU_TOP_0_AVS_RESET_BASE, NPU_TOP_0_AVS_RESET_SPAN);
  this->instrPtr = (uint32_t *)map_physical(fd, NPU_TOP_0_AVS_WRITE_INSTR_BASE, NPU_TOP_0_AVS_WRITE_INSTR_SPAN);
  this->results = ch_hashcreate(float);
  resultsThread=std::thread([this](){
    uint32_t PC = outPtr[1];
    uint32_t lastPC=PC;
    while(1)
    {
      PC = outPtr[1];
      if(PC!=lastPC)
      {
        float val=*((float*)outPtr[0]);
        chprintln("Got value ",val," for PC ",PC);
        ch_hashinsert(float,results,PC,val);
        lastPC=PC;
      }
    }
  });
  resultsThread.detach();

  this->PC = 0;

  this->instructionsFile.open("instr.txt", std::ifstream::out | std::ifstream::trunc);
  this->instructionsFile.close();
  this->instructionsFile.open("instr.txt", std::ios_base::app | std::ios::binary);
}
MemManager::MemManager(uchar* virt)
{
  this->base = NULL;
  this->shared_addr = virt;
  this->PC = 0;
  this->results = ch_hashcreate(float);

  this->instructionsFile.open("instr.txt", std::ifstream::out | std::ifstream::trunc);
  this->instructionsFile.close();
  this->instructionsFile.open("instr.txt", std::ios_base::app | std::ios::binary);
}

MemManager::~MemManager()
{
  // if (this->base)
  // {
  //   unmap_physical(shared_addr, LW_BRIDGE_SPAN);
  //   close_physical(fd);
  // }
}

void MemManager::writeInstruction(uint32_t i)
{
  ++PC;

  instructionsFile.write(reinterpret_cast<char *>(&i), sizeof(i));
  // waiting on lucas
}
void MemManager::writeData(float *ptr, size_t size)
{
  for (int i = 0; i < size / sizeof(float);i++)
  {
    while (!writeReady)
      ;
    // assume NPU_TOP_0_AVS_WRITE_SPAN==sizeof(float)
    memcpy(shared_addr, ptr + i, sizeof(float));
  }
}
float MemManager::getResult(uint32_t PC)
{
  chprintln("Requesting value for PC ",PC);
  float*v=ch_hashgetp(float,results,PC);
  while(v==NULL)
  {
    v = ch_hashgetp(float, results, PC);
  }
  ch_hashrem(float,results,PC);
  chprintln("Request complete");
  return *v;
  // while (outPtr[1] != PC)
  //   ;
  // return *((float*)outPtr[0]);
}
