

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
    AmplitudePhaseFieldCrystalProblem(const unsigned int degree, const unsigned int num_index);
    void run();
  private:
    void   make_grid_and_dofs();
    void   assemble_system(int index);
    void   solve(int index);
    void   output_results() const;
    void   material_system(const typename DoFHandler<dim>::active_cell_iterator cell,
			   std::vector<double> & A2, std::vector<std::vector<double>> & values);
    const unsigned int degree;
    const unsigned int num_index;
    Triangulation<dim> triangulation;
    FESystem<dim>  fe;
    DoFHandler<dim>    dof_handler;
    BlockSparsityPattern      sparsity_pattern;
    std::vector<BlockSparseMatrix<double>> system_matrix{}; // = std::vector<BlockSparseMatrix<double>>(3);
    std::vector<BlockVector<double>> solution{}; // = std::vector<BlockVector<double>>(3);
    std::vector<BlockVector<double>> old_solution{}; // = std::vector<BlockVector<double>>(3);
    std::vector<BlockVector<double>> system_rhs{}; // = std::vector<BlockVector<double>>(3);
    const unsigned int n_refinement_steps;
    DiscreteTime time;
  };


  
  template <int dim>
  AmplitudePhaseFieldCrystalProblem<dim>::AmplitudePhaseFieldCrystalProblem(const unsigned int degree, const unsigned int num_index)
    : degree(degree)
    , num_index(num_index)
    , fe(FE_Q<dim>(degree),
         1,
         FE_Q<dim>(degree),
         1,
         FE_Q<dim>(degree),
         1,
         FE_Q<dim>(degree),
         1)
    , dof_handler(triangulation)  
    , system_matrix(num_index)
    , solution(num_index)
    , old_solution(num_index)
    , system_rhs(num_index) 
    , n_refinement_steps(6)
    , time(/*start time*/ 0., /*end time*/ 1000.)
  {
  }



  template <int dim>
  class RotatedIC : public Function<dim>
  {
  public:
      RotatedIC(unsigned int i)
      : Function<dim>(dim + 2), index{i}
    {}

    unsigned int index;

    virtual void vector_value(const Point<dim> & p,
                              Vector<double> &  values) const override
    {
      Assert(values.size() == dim + 2,
             ExcDimensionMismatch(values.size(), dim + 2));

      const unsigned int ncomp = 3;
      std::vector<Tensor<1, dim>> k_vector(ncomp);
      std::vector<std::vector<double>> k_Rvector(ncomp , std::vector<double> (dim, 0.0));
      std::vector<std::vector<double>> dk(ncomp , std::vector<double> (dim, 0.0));
      
      k_vector[0][0] = -0.866025;
      k_vector[0][1] = -0.5;
      
      k_vector[1][0] = 0.0;
      k_vector[1][1] = 1.0;
      
      k_vector[2][0] = 0.866025;
      k_vector[2][1] = -0.5;
      
      const double db = 0.02;
      // const double bx = 0.98;                                                                                                                                
      const double v = 0.3333;
      const double t = 0.5;
      const double theta = 5;
      const double thetaR = theta * M_PI/180.0;
      const double epsilon = 3;
      const double radius = 10;

      for (unsigned int i = 0; i < ncomp; ++i)
	{
	  k_Rvector[i][0] = k_vector[i][0] * std::cos(thetaR) - k_vector[i][1] * std::sin(thetaR);
	  k_Rvector[i][1] = k_vector[i][0] * std::sin(thetaR) + k_vector[i][1] * std::cos(thetaR);
	  dk[i][0] = k_Rvector[i][0] - k_vector[i][0];
	  dk[i][1] = k_Rvector[i][1] - k_vector[i][1];
	}

      std::vector<double> exp(ncomp, 0.0);
   
      for (unsigned int i = 0; i < ncomp; ++i)
	for (unsigned int j = 0; j < dim; ++j)
	  exp[i] += dk[i][j] * p(j);

      double r = p.norm();

        //sign distance function                                                                                                                                     
      double d = r - radius;
      
      double param = (3.0 * d/epsilon);
      double shi = 0.5 * (1.0 - std::tanh(param));
      
      double paramN = -(3.0 * d/epsilon);
      double shiN = 0.5 * (1.0 - std::tanh(paramN));
      
      double num = t * t - 15.0 * v * db;
      double phi = (t + std::pow(num, 0.5))/(15.0 * v);

      // initial condition for alpha_re       
      values(0) = shiN * phi + shi * phi * std::cos(exp[index]);

      // initial condition for alpha_im
      values(1) = shi * phi * std::sin(exp[index]);
      
      values(2) = 0.0; // initial condition for xi_re                                                                                                         
      values(3) = 0.0; // initial condition for xi_im                                                                                                          
    }
  };


  

   
  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
      InitialValues()
      : Function<dim>(dim + 2)
    {}
    
    unsigned int index;
           
    virtual void vector_value(const Point<dim> & /*p*/,
                              Vector<double> &  values) const override
    {
      Assert(values.size() == dim + 2,
             ExcDimensionMismatch(values.size(), dim + 2));

      const double db = 0.02;
      //const double bx = 0.98;
      const double v = 0.3333;
      const double t = 0.5;

      double num = t * t - 15.0 * v * db;
    
      values(0) = (t + std::pow(num, 0.5))/(15.0 * v); // inital condition for alpha_re

      values(1) = 0.0; // initial condition for alpha_im
        
      values(2) = 0.0; // initial condition for xi_re
        
      values(3) = 0.0; // initial condition for xi_im
    }
  };

  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::make_grid_and_dofs()
  {
    GridGenerator::hyper_cube(triangulation, -25, 25);
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

    for(unsigned int i=0; i < num_index; i++)
      {
          system_matrix[i].reinit(sparsity_pattern);
      
          solution[i].reinit(4);
          solution[i].block(0).reinit(n_alpha_re);
          solution[i].block(1).reinit(n_alpha_im);
          solution[i].block(2).reinit(n_xi_re);
          solution[i].block(3).reinit(n_xi_im);
          solution[i].collect_sizes();
      
          old_solution[i].reinit(4);
          old_solution[i].block(0).reinit(n_alpha_re);
          old_solution[i].block(1).reinit(n_alpha_im);
          old_solution[i].block(2).reinit(n_xi_re);
          old_solution[i].block(3).reinit(n_xi_im);
          old_solution[i].collect_sizes();
      
          system_rhs[i].reinit(4);
          system_rhs[i].block(0).reinit(n_alpha_re);
          system_rhs[i].block(1).reinit(n_alpha_im);
          system_rhs[i].block(2).reinit(n_xi_re);
          system_rhs[i].block(3).reinit(n_xi_im);
          system_rhs[i].collect_sizes();
      }
      
      
  }


  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::material_system
  (const typename DoFHandler<dim>::active_cell_iterator cell, std::vector<double> & A2
   ,std::vector<std::vector<double>> & values)
  {
   QGauss<dim>     quadrature_formula(degree + 1);
   const unsigned int n_q_points    = quadrature_formula.size();
   
   const FEValuesExtractors::Scalar alpha_re(0);
   const FEValuesExtractors::Scalar alpha_im(1);

   std::vector<double>  old_solution_alpha_re_values(n_q_points);
   std::vector<double>  old_solution_alpha_im_values(n_q_points);

   FEValues<dim>     fe_values(fe,
			       quadrature_formula,
			       update_values | update_gradients |
			       update_quadrature_points | update_JxW_values);

   //std::vector<std::vector<double>> values(2*num_index , std::vector<double> (n_q_points, 0.0));
   
   for(unsigned int index = 0; index < num_index; index++)
     {
       fe_values.reinit(cell);
       fe_values[alpha_re].get_function_values(old_solution[index], old_solution_alpha_re_values);
       fe_values[alpha_im].get_function_values(old_solution[index], old_solution_alpha_im_values);

       values[2*index]    = old_solution_alpha_re_values;
       values[2*index+1]  = old_solution_alpha_im_values;
            
       for (unsigned int q = 0; q < n_q_points; ++q)
	 {
	   const double alpha_re = old_solution_alpha_re_values[q];
	   const double alpha_im = old_solution_alpha_im_values[q];

	   A2[q] += alpha_re * alpha_re + alpha_im * alpha_im;
	 }
     }
  }


  unsigned int ID(unsigned int i)
  {
    return (i > 5) ? i-6:i;
  }
  
  double Ref(double a, double b, double c, double d)
  {
    return a*c-b*d;  //Re((a-ib)(c-id))  
  }
  
  double Imf(double a,double b,double c,double d)
  {
    return -(a*d+b*c); //Im((a-ib)(c-id))
  }

  void dfdn(const std::vector<double> & vals, double & val_re, double & val_im, unsigned int i)
  {
    double t = 0.5;
    
    val_re = -2.0 * t * Ref(vals[ID(2*i+2)],vals[ID(2*i+3)],vals[ID(2*i+4)],vals[ID(2*i+5)]);
    
    val_im = -2.0 * t * Imf(vals[ID(2*i+2)],vals[ID(2*i+3)],vals[ID(2*i+4)],vals[ID(2*i+5)]);
  }
  
  
  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::assemble_system(int index)
  {
    system_matrix[index] = 0;
    system_rhs[index]    = 0;
    QGauss<dim>     quadrature_formula(degree + 1);

    FEValues<dim>     fe_values(fe,
				quadrature_formula,
				update_values | update_gradients |
				update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
    
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    /*
    std::vector<double>  old_solution_alpha_re_values(n_q_points);
    std::vector<double>  old_solution_alpha_im_values(n_q_points);
    std::vector<double>  solution_alpha_re_values(n_q_points);
    std::vector<double>  solution_alpha_im_values(n_q_points);*/
    
    const FEValuesExtractors::Scalar alpha_re(0);
    const FEValuesExtractors::Scalar alpha_im(1);
    const FEValuesExtractors::Scalar xi_re(2);
    const FEValuesExtractors::Scalar xi_im(3);
    
    const double db = 0.02;
    const double bx = 0.98;
    const double v = 0.3333;
    // const double t = 0.5;
    
    std::vector<Tensor<1, dim>> q_vector(num_index);

    q_vector[0][0] = -0.866025;
    q_vector[0][1] = -0.5;
    
    q_vector[1][0] = 0.0;
    q_vector[1][1] = 1.0;
    
    q_vector[2][0] = 0.866025;
    q_vector[2][1] = -0.5;
    
    const double K = bx;

    std::vector<std::vector<double>> values(2*num_index , std::vector<double> (n_q_points, 0.0));

    std::vector<double>  A2(n_q_points,0.0);
    
    std::vector<double> qp_values(2*num_index,0.0);

    double Ref, Imf;
    
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;
        
	// fe_values[alpha_re].get_function_values(old_solution[index], old_solution_alpha_re_values);
        // fe_values[alpha_im].get_function_values(old_solution[index], old_solution_alpha_im_values);

        std::fill(A2.begin(), A2.end(), 0.0);
	
	material_system(cell, A2, values);

        for (unsigned int q = 0; q < n_q_points; ++q)
	  {
            const double old_alpha_re = values[2*index][q]; //old_solution_alpha_re_values[q];
            const double old_alpha_im = values[2*index+1][q]; //old_solution_alpha_im_values[q];

	    for(unsigned int count = 0; count < num_index; count++)
	      {
		qp_values[2*count] = values[2*count][q];
		qp_values[2*count+1] = values[2*count+1][q];
	      }

	    dfdn(qp_values, Ref, Imf, index);
	    
	    const double G1 = (1.0/time.get_next_step_size())+db+3.0*v*(2.0*A2[q]+old_alpha_re*old_alpha_re-old_alpha_im*old_alpha_im);
            const double G2 = (1.0/time.get_next_step_size())+db+3.0*v*(2.0*A2[q]+old_alpha_im*old_alpha_im-old_alpha_re*old_alpha_re);
            
            const double H1 = ((1.0/time.get_next_step_size())+6.0*v*old_alpha_re*old_alpha_re)*old_alpha_re-Ref;
            const double H2 = ((1.0/time.get_next_step_size())+6.0*v*old_alpha_im*old_alpha_im)*old_alpha_im-Imf;
            
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
		    
		    const double A_j_alpha_re = 2.0 * q_vector[index] * grad_phi_j_alpha_re;
		    const double A_j_alpha_im = 2.0 * q_vector[index] * grad_phi_j_alpha_im;
		    const double A_j_xi_re = 2.0 * q_vector[index] * grad_phi_j_xi_re;
		    const double A_j_xi_im = 2.0 * q_vector[index] * grad_phi_j_xi_im;
                    
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
		system_matrix[index].add(local_dof_indices[i],
					 local_dof_indices[j],
					 local_matrix(i, j));
	    
	    for (unsigned int i = 0; i < dofs_per_cell; ++i)
	      system_rhs[index](local_dof_indices[i]) += local_rhs(i);
	    
	  }// end of q loop
	
      } // end of cell loop
  }
  
  
  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::solve(int index)
  {
    std::cout  << "Solving linear system... " << std::endl;
    
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix[index]);
    A_direct.vmult(solution[index], system_rhs[index]);
    
    old_solution[index] = solution[index];
  }
  
  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::output_results() const
  {
    if (time.get_step_number() % 10 != 0)
      return;
      
    std::vector<std::string> solution_names_one   = {"alpha_re_one", "alpha_im_one", "xi_re_one", "xi_im_one"};
    std::vector<std::string> solution_names_two   = {"alpha_re_two", "alpha_im_two", "xi_re_two", "xi_im_two"};
    std::vector<std::string> solution_names_three = {"alpha_re_three", "alpha_im_three", "xi_re_three", "xi_im_three"};

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution[0], solution_names_one);
    data_out.add_data_vector(solution[1], solution_names_two);
    data_out.add_data_vector(solution[2], solution_names_three);
      
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

      for(unsigned int i=0; i < num_index; i++)
      {
	
	VectorTools::project(dof_handler,
			     constraints,
			     QGauss<dim>(degree + 1),
			     RotatedIC<dim>(i),
			     old_solution[i]);
	/*
          VectorTools::project(dof_handler,
	  constraints,
	  QGauss<dim>(degree + 1),
	  InitialValues<dim>(),
	  old_solution[i]);*/
      }
      
    }
    do
      {
        std::cout << "Timestep " << time.get_step_number() + 1 << std::endl;

	for(unsigned int i=0; i < num_index; i++)
            {
                time.set_desired_next_step_size(1.0);
                assemble_system(i);
                solve(i);
            }
        
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
      AmplitudePhaseFieldCrystalProblem<2> amplitude_phase_field_crystal_problem(1,3);
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
