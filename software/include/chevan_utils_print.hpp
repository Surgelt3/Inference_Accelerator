#ifndef CHEVAN_UTILS_PRINT_H
#define CHEVAN_UTILS_PRINT_H
#include <chevan_utils_min.h>
#include <stdio.h>
#include <string>
#include <iostream>
template <typename P>
static void chprint(P p);
template <typename P, typename... P2>
static void chprint(P p, P2... p2);
template <typename P>
static void chprintln(P p);
template <typename P, typename... P2>
static void chprintln(P p, P2... p2);

template <typename Vector>
static void chprintVec(Vector v)
{
  for (auto t : v)
  {
    std::cout << t;
  }
  std::cout << std::endl;
}
template <typename V>
static void chprintVec2(V v)
{
  chprintln("{", v.x, ", ", v.y, "}");
}
template <typename V>
static void chprintVec3(V v)
{
  chprintln("{", v.x, ", ", v.y, ", ", v.z, "}");
}
template <typename V>
static void chprintVec4(V v)
{
  chprintln("{", v.x, ", ", v.y, ", ", v.z, ", ", v.w, "}");
}
static void chprintMat4(float mat[16])
{
  for (uint i = 0; i < 4; i++)
  {
    chprintln(mat[0 + i * 4], ", ", mat[1 + i * 4], ", ", mat[2 + i * 4], ", ", mat[3 + i * 4]);
  }
}
static void chprintMat4(double mat[16])
{
  for (uint i = 0; i < 4; i++)
  {
    chprintln(mat[0 + i * 4], ", ", mat[1 + i * 4], ", ", mat[2 + i * 4], ", ", mat[3 + i * 4]);
  }
}
template <typename M>
static void chprintMat4(M mat)
{
  chprintMat4((float *)&mat);
}
template <typename M>
static void chprintMat4D(M mat)
{
  chprintMat4((double *)&mat);
}
static void chprintMem(void *p, ulong length = 256)
{
  std::string s = "";
  for (uint i = 0; i < length; i++)
  {
    s += std::to_string(((uchar *)p)[i]);
  }
  std::cout << s << std::endl;
}
template <typename T>
static void chprintBits(T *t, size_t length = 1)
{
  for (uint i = 0; i < length * 8; i++)
  {
    std::cout << ((((uchar *)t)[i / 8] & (1 << i % 8)) > 0 ? 1 : 0);
  }
  std::cout << std::endl;
}
template <typename T>
static void chprintBits(T t)
{
  for (uint i = 0; i < sizeof(T) * 8; i++)
  {
    std::cout << ((t & (1 << i)) > 0 ? 1 : 0);
  }
  std::cout << std::endl;
}
// ----------------------------------------------
static void chprint()
{
  std::cout << std::endl;
}
static void chprintln()
{
  std::cout << std::endl;
}
static void chprintSep()
{
  std::cout << "---------------------------------" << std::endl;
}
static void chprint(uchar *p)
{
  std::cout << (void *)p;
}
static void chprint(uchar p)
{
  chprint((int)p);
}
#ifdef CHEVAN_UTILS_ARR_HPP
template <typename L>
static void chprint(Array<char, L> arr, L length = 256)
{
  L l = arr.length < length ? arr.length : length;
  std::cout << std::string(arr.arr, arr.arr + l);
}
template <typename L>
static void chprint(Array<uchar, L> arr, L length = 256)
{
  L l = arr.length < length ? arr.length : length;
  std::cout << std::string(arr.arr, arr.arr + l);
}
template <typename T, typename L>
static void chprint(Array<T, L> arr, L length = 256)
{
  L l = arr.length < length ? arr.length : length;
  std::string s = "{";
  s += std::to_string(arr[0]);
  for (uint i = 1; i < l; i++)
  {
    s += "," + std::to_string(arr[i]);
  }
  s += "}";
  std::cout << s;
}
#endif
template <typename T>
static void chprint(std::vector<T> vec)
{
  for (auto t : vec)
  {
    std::cout << t;
  }
}
#ifdef CHEVAN_UTILS_VEC2
static void chprint(CHEVAN_UTILS_VEC2 v)
{
  chprint("{", v.x, ", ", v.y, "}");
}
#endif
#ifdef CHEVAN_UTILS_VEC3
static void chprint(CHEVAN_UTILS_VEC3 v)
{
  chprint("{", v.x, ", ", v.y, ", ", v.z, "}");
}
#endif
#ifdef CHEVAN_UTILS_VEC4
static void chprint(CHEVAN_UTILS_VEC4 v)
{
  chprint("{", v.x, ", ", v.y, ", ", v.z, ", ", v.w, "}");
}
#endif
static void chprint(Cardinal8dir dir)
{
  switch (dir)
  {
  case Cardinal8dir::CENTER:
    std::cout << "CENTER" << std::endl;
    break;
  case Cardinal8dir::NORTH:
    std::cout << "NORTH" << std::endl;
    break;
  case Cardinal8dir::SOUTH:
    std::cout << "SOUTH" << std::endl;
    break;
  case Cardinal8dir::EAST:
    std::cout << "EAST" << std::endl;
    break;
  case Cardinal8dir::WEST:
    std::cout << "WEST" << std::endl;
    break;
  case Cardinal8dir::NORTH_EAST:
    std::cout << "NORTH_EAST" << std::endl;
    break;
  case Cardinal8dir::NORTH_WEST:
    std::cout << "NORTH_WEST" << std::endl;
    break;
  case Cardinal8dir::SOUTH_EAST:
    std::cout << "SOUTH_EAST" << std::endl;
    break;
  case Cardinal8dir::SOUTH_WEST:
    std::cout << "SOUTH_WEST" << std::endl;
    break;
  default:
    break;
  }
}
template <typename Printable>
static void chprint(Printable p)
{
  std::cout << p;
}
template <typename Printable>
static void chprintln(Printable p)
{
  chprint(p);
  std::cout << std::endl;
}
template <typename Printable, typename... Printable2>
static void chprint(Printable p, Printable2... p2)
{
  chprint(p);
  chprint(p2...);
}
template <typename Printable, typename... Printable2>
static void chprintln(Printable p, Printable2... p2)
{
  chprint(p);
  chprint(p2...);
  std::cout << std::endl;
}

#endif