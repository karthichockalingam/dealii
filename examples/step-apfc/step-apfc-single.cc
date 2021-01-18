


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <iostream>
#include <fstream>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/lac/sparse_direct.h>

namespace StepAPFC
{
  using namespace dealii;

  template <int dim>
  class AmplitudePhaseFieldCrystalProblem
  {
  public:
    AmplitudePhaseFieldCrystalProblem(const unsigned int degree);
    void run();
  private:
    void   make_grid_and_dofs();
    void   assemble_system();
    void   solve();
    void   output_results() const;
    const unsigned int degree;
    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    const unsigned int n_refinement_steps;
    DiscreteTime time;
    BlockVector<double> solution;
    BlockVector<double> old_solution;
    BlockVector<double> system_rhs;
  };


  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
    InitialValues()
      : Function<dim>(dim + 2)
    {}

        
    virtual void vector_value(const Point<dim> & /*p*/,
                              Vector<double> &  values) const override
    {
      Assert(values.size() == dim + 2,
             ExcDimensionMismatch(values.size(), dim + 2));
      values(0) = 1.0; // inital condition for alpha_re

      values(1) = 0.0; // initial condition for alpha_im
        
      values(2) = 0.0; // initial condition for xi_re
        
      values(3) = 0.0; // initial condition for xi_im
    }
  };


  


  template <int dim>
  AmplitudePhaseFieldCrystalProblem<dim>::AmplitudePhaseFieldCrystalProblem(const unsigned int degree)
    : degree(degree)
    , fe(FE_Q<dim>(degree),
         1,
         FE_Q<dim>(degree),
         1,
         FE_Q<dim>(degree),
         1,
         FE_Q<dim>(degree),
         1)
    , dof_handler(triangulation)
    , n_refinement_steps(6)
    , time(/*start time*/ 0., /*end time*/ 1000.)
  {}

  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::make_grid_and_dofs()
  {
    GridGenerator::hyper_cube(triangulation, 0, 50);
    triangulation.refine_global(n_refinement_steps);
    dof_handler.distribute_dofs(fe);
      
    DoFRenumbering::component_wise(dof_handler);
      
    const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
      
    const unsigned int n_alpha_re      = dofs_per_component[0],
                       n_alpha_im      = dofs_per_component[1],
                       n_xi_re         = dofs_per_component[2],
                       n_xi_im         = dofs_per_component[3];
      
    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_alpha_re  << '+' << n_alpha_im << '+' << n_xi_re << '+' << n_xi_im << ')' << std::endl
              << std::endl;
      
    const unsigned int n_couplings = dof_handler.max_couplings_between_dofs();
      
    sparsity_pattern.reinit(4, 4);
    sparsity_pattern.block(0, 0).reinit(n_alpha_re,   n_alpha_re,      n_couplings);
    sparsity_pattern.block(1, 0).reinit(n_alpha_im,   n_alpha_re,      n_couplings);
    sparsity_pattern.block(2, 0).reinit(n_xi_re,      n_alpha_re,      n_couplings);
    sparsity_pattern.block(3, 0).reinit(n_xi_im,      n_alpha_re,      n_couplings);
    sparsity_pattern.block(0, 1).reinit(n_alpha_re,   n_alpha_im,      n_couplings);
    sparsity_pattern.block(1, 1).reinit(n_alpha_im,   n_alpha_im,      n_couplings);
    sparsity_pattern.block(2, 1).reinit(n_xi_re,      n_alpha_im,      n_couplings);
    sparsity_pattern.block(3, 1).reinit(n_xi_im,      n_alpha_im,      n_couplings);
    sparsity_pattern.block(0, 2).reinit(n_alpha_re,   n_xi_re,         n_couplings);
    sparsity_pattern.block(1, 2).reinit(n_alpha_im,   n_xi_re,         n_couplings);
    sparsity_pattern.block(2, 2).reinit(n_xi_re,      n_xi_re,         n_couplings);
    sparsity_pattern.block(3, 2).reinit(n_xi_im,      n_xi_re,         n_couplings);
    sparsity_pattern.block(0, 3).reinit(n_alpha_re,   n_xi_im,         n_couplings);
    sparsity_pattern.block(1, 3).reinit(n_alpha_im,   n_xi_im,         n_couplings);
    sparsity_pattern.block(2, 3).reinit(n_xi_re,      n_xi_im,         n_couplings);
    sparsity_pattern.block(3, 3).reinit(n_xi_im,      n_xi_im,         n_couplings);
      
    sparsity_pattern.collect_sizes();
      
    DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern);
    sparsity_pattern.compress();
    system_matrix.reinit(sparsity_pattern);
      
    solution.reinit(4);
    solution.block(0).reinit(n_alpha_re);
    solution.block(1).reinit(n_alpha_im);
    solution.block(2).reinit(n_xi_re);
    solution.block(3).reinit(n_xi_im);
    solution.collect_sizes();
      
    old_solution.reinit(4);
    old_solution.block(0).reinit(n_alpha_re);
    old_solution.block(1).reinit(n_alpha_im);
    old_solution.block(2).reinit(n_xi_re);
    old_solution.block(3).reinit(n_xi_im);
    old_solution.collect_sizes();
      
    system_rhs.reinit(4);
    system_rhs.block(0).reinit(n_alpha_re);
    system_rhs.block(1).reinit(n_alpha_im);
    system_rhs.block(2).reinit(n_xi_re);
    system_rhs.block(3).reinit(n_xi_im);
    system_rhs.collect_sizes();
  }

  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::assemble_system()
  {
    system_matrix = 0;
    system_rhs    = 0;
    QGauss<dim>     quadrature_formula(degree + 1);

    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      

    std::vector<double>  old_solution_alpha_re_values(n_q_points);
    std::vector<double>  old_solution_alpha_im_values(n_q_points);
      
    const FEValuesExtractors::Scalar alpha_re(0);
    const FEValuesExtractors::Scalar alpha_im(1);
    const FEValuesExtractors::Scalar xi_re(2);
    const FEValuesExtractors::Scalar xi_im(3);
      
    const double db = 1.0;
    const double bx = 1.0;
    const double v = 0.3333;
   // const double t = 0.5;
      
    Tensor<1, dim> q_vector;
    q_vector[0] = 1.0;
    q_vector[1] = 1.0;
      
    const double K = bx;
      
    const double A2 = 1.0;
      
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;
          
        fe_values[alpha_re].get_function_values(old_solution, old_solution_alpha_re_values);
        fe_values[alpha_im].get_function_values(old_solution, old_solution_alpha_im_values);
      
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double old_alpha_re = old_solution_alpha_re_values[q];
            const double old_alpha_im = old_solution_alpha_im_values[q];
            
            const double G1 = (1.0/time.get_next_step_size())+db+3.0*v*(A2+old_alpha_re*old_alpha_re-old_alpha_im*old_alpha_im);
            const double G2 = (1.0/time.get_next_step_size())+db+3.0*v*(A2+old_alpha_im*old_alpha_im-old_alpha_re*old_alpha_re);
            
            const double H1 = ((1.0/time.get_next_step_size())+6.0*v*old_alpha_re*old_alpha_re)*old_alpha_re;
            const double H2 = ((1.0/time.get_next_step_size())+6.0*v*old_alpha_im*old_alpha_im)*old_alpha_im;
            
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, dim> grad_phi_i_alpha_re = fe_values[alpha_re].gradient(i, q);
              const Tensor<1, dim> grad_phi_i_alpha_im = fe_values[alpha_im].gradient(i, q);
              const Tensor<1, dim> grad_phi_i_xi_re = fe_values[xi_re].gradient(i, q);
              const Tensor<1, dim> grad_phi_i_xi_im = fe_values[xi_im].gradient(i, q);
                
              const double phi_i_alpha_re  = fe_values[alpha_re].value(i, q);
              const double phi_i_alpha_im  = fe_values[alpha_im].value(i, q);
              const double phi_i_xi_re  = fe_values[xi_re].value(i, q);
              const double phi_i_xi_im  = fe_values[xi_im].value(i, q);
                
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<1, dim> grad_phi_j_alpha_re = fe_values[alpha_re].gradient(j, q);
                  const Tensor<1, dim> grad_phi_j_alpha_im = fe_values[alpha_im].gradient(j, q);
                  const Tensor<1, dim> grad_phi_j_xi_re = fe_values[xi_re].gradient(j, q);
                  const Tensor<1, dim> grad_phi_j_xi_im = fe_values[xi_im].gradient(j, q);
                    
                  const double phi_j_alpha_re = fe_values[alpha_re].value(j, q);
                  const double phi_j_alpha_im = fe_values[alpha_im].value(j, q);
                  const double phi_j_xi_re = fe_values[xi_re].value(j, q);
                  const double phi_j_xi_im = fe_values[xi_im].value(j, q);
                
                  const double A_j_alpha_re = 2.0 * q_vector * grad_phi_j_alpha_re;
                  const double A_j_alpha_im = 2.0 * q_vector * grad_phi_j_alpha_im;
                  const double A_j_xi_re = 2.0 * q_vector * grad_phi_j_xi_re;
                  const double A_j_xi_im = 2.0 * q_vector * grad_phi_j_xi_im;
                    
                  local_matrix(i, j) +=
                  (grad_phi_i_alpha_re * grad_phi_j_alpha_re + phi_i_alpha_re * A_j_alpha_im + phi_i_alpha_re * phi_j_xi_re
                 - phi_i_alpha_im * A_j_alpha_re + grad_phi_i_alpha_im * grad_phi_j_alpha_im + phi_i_alpha_im * phi_j_xi_im
                 + phi_i_xi_re * G1 * phi_j_alpha_re - K * grad_phi_i_xi_re * grad_phi_j_xi_re - phi_i_xi_re * K * A_j_xi_im
                 + phi_i_xi_im * G2 * phi_j_alpha_im + phi_i_xi_im * K * A_j_xi_re - K * grad_phi_i_xi_im * grad_phi_j_xi_im)
                  * fe_values.JxW(q);
                } // end of j loop
                
              local_rhs(i) += (phi_i_xi_re * H1 + phi_i_xi_im * H2) * fe_values.JxW(q);
            } // end of i loop
      
        cell->get_dof_indices(local_dof_indices);
          
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              local_matrix(i, j));
          
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs(local_dof_indices[i]) += local_rhs(i);
                     
            }// end of q loop
          
      } // end of cell loop
  }


  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::solve()
  {
  std::cout  << "Solving linear system... " << std::endl;
  
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);

  old_solution = solution;
  }

  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::output_results() const
  {
    if (time.get_step_number() % 10 != 0)
      return;
    std::vector<std::string> solution_names;
    switch (dim)
      {
        case 2:
          solution_names = {"alpha_re_one", "alpha_im_one", "xi_re_one", "xi_im_one"};
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, solution_names);
    data_out.build_patches(degree + 1);
    std::ofstream output("solution-" +
                         Utilities::int_to_string(time.get_step_number(), 4) +
                         ".vtk");
    data_out.write_vtk(output);
  }
  
  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::run()
  {
    make_grid_and_dofs();
    {
      AffineConstraints<double> constraints;
      constraints.close();

      
      VectorTools::project(dof_handler,
                           constraints,
                           QGauss<dim>(degree + 1),
                           InitialValues<dim>(),
                           old_solution);
    }
    do
      {
        std::cout << "Timestep " << time.get_step_number() + 1 << std::endl;
        time.set_desired_next_step_size(1.0);
        assemble_system();
        solve();
        output_results();
        time.advance_time();
        std::cout << "   Now at t=" << time.get_current_time()
                  << ", dt=" << time.get_previous_step_size() << '.'
                  << std::endl
                  << std::endl;
      }
    while (time.is_at_end() == false);
  }
} // namespace StepAPFC


int main()
{
  try
    {
      using namespace StepAPFC;
      AmplitudePhaseFieldCrystalProblem<2> amplitude_phase_field_crystal_problem(1);
      amplitude_phase_field_crystal_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
