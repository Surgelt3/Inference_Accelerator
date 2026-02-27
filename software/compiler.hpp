#pragma once

#include "neural_net.h"
#include <chevan_utils_print.hpp>
#include <thread>
#include <mutex>

#define MAC_OP 0b000
#define RELU_OP 0b000
#define GAP_OP 0b000
#define LOAD_OP 0b001
#define START_OP 0b010
#define END_OP 0b011
static uint32_t MAC_Instruction(float *start_loc, size_t size, float *param_loc)
{
  uint32_t i = 0;
  i |= MAC_OP << 29;
  i |= ((size_t)start_loc & 0x1F) << 20;
  i |= (size & 0x3F) << 10;
  i |= ((size_t)param_loc & 0x1F) << 1;
  return i;
}
static uint32_t RELU_Instruction()
{
  // waiting on lucas
  uint32_t i = 0;
  i |= RELU_OP << 29;
  return i;
}
static uint32_t GAP_Instruction()
{
  // waiting on lucas
  uint32_t i = 0;
  i |= GAP_OP << 29;
  return i;
}
static uint32_t LOAD_Instruction(size_t targetAddress)
{
  uint32_t i = 0;
  i |= LOAD_OP << 29;
  i |= (targetAddress & 0x1F) << 20;
  return i;
}
static uint32_t START_Instruction()
{
  // waiting on lucas
  uint32_t i = 0;
  i |= START_OP << 29;
  return i;
}
static uint32_t END_Instruction()
{
  // waiting on lucas
  uint32_t i = 0;
  i |= END_OP << 29;
  return i;
}
#define DATA_OUT_LENGTH 16

class MemManager
{
public:
  const uchar *base;
  void *shared_addr;
  float *outPtr;
  size_t maxSize;
  uint PC;
  float *temporaryLoadAddress = (float *)0x40;
  MemManager();
  MemManager(uchar *shared);
  ~MemManager();

  void writeInstruction(uint32_t i);
  void writeData(float *ptr, size_t size);
  float getResult(uint32_t PC);
};

#define onDevice 0
class Compiler
{
  MemManager manager;

public:
  Compiler() : manager(MemManager((uchar *)malloc(0x5000)))
  {
  }
  void writeInstructions(const Net &net);
  void compileModel(const Net &net);
};