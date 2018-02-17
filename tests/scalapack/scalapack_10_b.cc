// ---------------------------------------------------------------------
//
// Copyright (C) 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#include "../tests.h"
#include "../lapack/create_matrix.h"

// test serial saving and loading of distributed ScaLAPACKMatrices with prescribed chunk sizes

#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/scalapack.h>

#include <fstream>
#include <iostream>
#include <cstdio>


template <typename NumberType>
void test(const std::pair<unsigned int,unsigned int> &size, const unsigned int block_size, const std::pair<unsigned int,unsigned int> &chunk_size)
{
  const std::string filename ("scalapck_10_b_test.h5");

  MPI_Comm mpi_communicator(MPI_COMM_WORLD);
  const unsigned int this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator));
  ConditionalOStream pcout (std::cout, (this_mpi_process ==0));

  FullMatrix<NumberType> full(size.first,size.second);
  create_random(full);

  //create 2d process grid
  std::shared_ptr<Utilities::MPI::ProcessGrid> grid = std::make_shared<Utilities::MPI::ProcessGrid>(mpi_communicator,size.first,
                                                      size.second,block_size,block_size);

  ScaLAPACKMatrix<NumberType> scalapack_matrix(size.first,size.second,grid,block_size,block_size);
  ScaLAPACKMatrix<NumberType> scalapack_matrix_copy(size.first,size.second,grid,block_size,block_size);

  scalapack_matrix = full;
  scalapack_matrix.save(filename.c_str(),chunk_size);
  scalapack_matrix_copy.load(filename.c_str());

  FullMatrix<NumberType> copy(size.first,size.second);
  scalapack_matrix_copy.copy_to(copy);
  copy.add(-1,full);

  pcout << size.first << "x" << size.second << " & "
        << block_size << " & "
        << chunk_size.first << "x" << chunk_size.second << std::endl;
  AssertThrow(copy.frobenius_norm()<1e-12,ExcInternalError());
  std::remove(filename.c_str());
}



int main (int argc,char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

  std::vector<std::pair<unsigned int,unsigned int>> sizes;
  sizes.push_back(std::make_pair(100,75));
  sizes.push_back(std::make_pair(200,225));
  sizes.push_back(std::make_pair(300,250));

  const std::vector<unsigned int> block_sizes = {{1,16,32}};

  std::vector<std::pair<unsigned int,unsigned int>> chunk_sizes;
  chunk_sizes.push_back(std::make_pair(1,1));
  chunk_sizes.push_back(std::make_pair(10,10));
  chunk_sizes.push_back(std::make_pair(50,50));
  chunk_sizes.push_back(std::make_pair(100,75));

  for (unsigned int i=0; i<sizes.size(); ++i)
    for (unsigned int j=0; j<block_sizes.size(); ++j)
      for (unsigned int k=0; k<chunk_sizes.size(); ++k)
        test<double>(sizes[i],block_sizes[j],chunk_sizes[k]);

  for (unsigned int i=0; i<sizes.size(); ++i)
    for (unsigned int j=0; j<block_sizes.size(); ++j)
      for (unsigned int k=0; k<chunk_sizes.size(); ++k)
        test<float>(sizes[i],block_sizes[j],chunk_sizes[k]);
}