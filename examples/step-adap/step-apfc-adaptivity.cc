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
#include <deal.II/grid/grid_refinement.h>
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
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

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
    void   setup_system();
    void   assemble_system(int index);
    void   solve(int index);
    void   solve_direct(int index);
    void   output_results() const;
    void refine_mesh(const unsigned int min_grid_level,
		      const unsigned int max_grid_level);
    void   material_system(const typename DoFHandler<dim>::active_cell_iterator cell,
			   std::vector<double> & A2, std::vector<std::vector<double>> & values);
    const unsigned int degree;
    const unsigned int num_index;
    Triangulation<dim> triangulation;
    FESystem<dim>  fe;
    DoFHandler<dim>    dof_handler;
    SparsityPattern      sparsity_pattern;
    std::vector<SparseMatrix<double>> system_matrix{};
    std::vector<Vector<double>> solution{};
    std::vector<Vector<double>> old_solution{};
    std::vector<Vector<double>> system_rhs{};
    const unsigned int n_refinement_steps;
    DiscreteTime time;// <========== include or delete??????
    AffineConstraints<double> constraints;
    //double       time;
    //double       time_step;
    //unsigned int timestep_number;
  };


  template <int dim>
  AmplitudePhaseFieldCrystalProblem<dim>::AmplitudePhaseFieldCrystalProblem(const unsigned int degree, const unsigned int num_index)
    : degree(degree)
    , num_index(num_index)
    , fe(FE_Q<dim>(degree),4)
    , dof_handler(triangulation)  
    , system_matrix(num_index)
    , solution(num_index)
    , old_solution(num_index)
    , system_rhs(num_index) 
    , n_refinement_steps(6)
    , time(/*start time*/ 0., /*end time*/ 1000.)
  {}

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
  void AmplitudePhaseFieldCrystalProblem<dim>::setup_system()
  {
    /*
    const double min_coord = -25;
    const double max_coord = 25;
    
    GridGenerator::hyper_cube(triangulation, min_coord, max_coord, true);    
    triangulation.refine_global(n_refinement_steps);*/

    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;
    
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);    
    constraints.close();
    
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    
    DoFTools::make_sparsity_pattern(dof_handler,
				    dsp,
				    constraints,
				    /*keep_constrained_dofs = */ false);
    
    sparsity_pattern.copy_from(dsp);
    
    
    for(unsigned int i=0; i < num_index; i++)
      {
	system_matrix[i].reinit(sparsity_pattern);
        solution[i].reinit(dof_handler.n_dofs());
        old_solution[i].reinit(dof_handler.n_dofs());
        system_rhs[i].reinit(dof_handler.n_dofs());
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
  
   std::vector<Vector<double> > old_local_solution_values(n_q_points, Vector<double> (dim+2));

   FEValues<dim>     fe_values(fe,
			       quadrature_formula,
			       update_values | update_gradients |
			       update_quadrature_points | update_JxW_values);

   //std::vector<std::vector<double>> values(2*num_index , std::vector<double> (n_q_points, 0.0));
   
   for(unsigned int index = 0; index < num_index; index++)
     {
       fe_values.reinit(cell);
       fe_values.get_function_values(old_solution[index], old_local_solution_values);
         
      // fe_values[alpha_im].get_function_values(old_solution[index], old_solution_alpha_im_values);

      // values[2*index]    = old_solution_alpha_re_values;
      // values[2*index+1]  = old_solution_alpha_im_values;
            
       for (unsigned int q = 0; q < n_q_points; ++q)
       {
	 values[2*index][q]    = old_local_solution_values[q][0];
	 values[2*index+1][q]  = old_local_solution_values[q][1];
	 
	 const double alpha_re = old_local_solution_values[q][0];
	 const double alpha_im = old_local_solution_values[q][1];
	 
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
		unsigned int _i =  fe.system_to_component_index(i).first;

		for (unsigned int j = 0; j < dofs_per_cell; ++j)
		  {
		    unsigned int _j =  fe.system_to_component_index(j).first;
		    
		    if((_i == 0) && (_j == 0))
		      local_matrix(i, j) += fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
		    
		    if((_i == 0) && (_j == 1))
		      {
			const double A_j = 2.0 * q_vector[index] * fe_values.shape_grad(j, q);
			local_matrix(i, j) += fe_values.shape_value(i, q) * A_j * fe_values.JxW(q);
		      }
		    
		    if((_i == 0) && (_j == 2))
		      local_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
		    
		    
		    if((_i == 1) && (_j == 0))
		      {
			const double A_j = 2.0 * q_vector[index] * fe_values.shape_grad(j, q);
			local_matrix(i, j) -= fe_values.shape_value(i, q) * A_j * fe_values.JxW(q);
		      }
		    
		    if((_i == 1) && (_j == 1))
		      local_matrix(i, j) += fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
		    
		    if((_i == 1) && (_j == 3))
		      local_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
		    
		    
		    
		    if((_i == 2) && (_j == 0))
		      local_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * G1 * fe_values.JxW(q);
		    
		    if((_i == 2) && (_j == 2))
		      local_matrix(i, j) -= fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * K * fe_values.JxW(q);
		    
		    if((_i == 2) && (_j == 3))
		      {
			const double A_j = 2.0 * q_vector[index] * fe_values.shape_grad(j, q);
			local_matrix(i, j) -= fe_values.shape_value(i, q) * A_j * K * fe_values.JxW(q);
		      }
		    
		    if((_i == 3) && (_j == 1))
		      local_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * G2 * fe_values.JxW(q);
		    
		    if((_i == 3) && (_j == 2))
		      {
			const double A_j = 2.0 * q_vector[index] * fe_values.shape_grad(j, q);
			local_matrix(i, j) += fe_values.shape_value(i, q) * A_j * K * fe_values.JxW(q);
		      }
		    
		    if((_i == 3) && (_j == 3))
		      local_matrix(i, j) -= fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * K * fe_values.JxW(q);
		    
		  } // end of j loop
                
		if(_i == 2)
		  local_rhs(i) += fe_values.shape_value(i, q) * H1 * fe_values.JxW(q);
		
		if(_i == 3)
		  local_rhs(i) += fe_values.shape_value(i, q) * H2 * fe_values.JxW(q);
		
	      } // end of i loop

	  }// end of q loop      
	    
	cell->get_dof_indices(local_dof_indices);
	
	constraints.distribute_local_to_global(
					       local_matrix, local_rhs, local_dof_indices, system_matrix[index], system_rhs[index]);
	/*
	  for (unsigned int i = 0; i < dofs_per_cell; ++i)
	  for (unsigned int j = 0; j < dofs_per_cell; ++j)
	  system_matrix[index].add(local_dof_indices[i],
	  local_dof_indices[j],
	  local_matrix(i, j));
	  
	  for (unsigned int i = 0; i < dofs_per_cell; ++i)
	  system_rhs[index](local_dof_indices[i]) += local_rhs(i);*/
	
      } // end of cell loop
    
  }


  
  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::refine_mesh(const unsigned int min_grid_level,
							   const unsigned int max_grid_level)
  {
    std::vector<Vector<float>> estimated_error_per_cell(num_index,Vector<float>(triangulation.n_active_cells()));
    Vector<float> sum_estimated_error_per_cell(triangulation.n_active_cells());
    
    FEValuesExtractors::Scalar alpha_im(1);
    
    for(unsigned int i=0; i < num_index; i++)
      KellyErrorEstimator<dim>::estimate(
					 dof_handler,
					 QGauss<dim - 1>(fe.degree + 1),
					 {},
					 solution[i],
					 estimated_error_per_cell[i],
					 fe.component_mask(alpha_im));
    
    //for(unsigned int i=0; i < num_index; i++)
    //sum_estimated_error_per_cell += estimated_error_per_cell[i];
    
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell[0],
                                                      0.80,
                                                      0.20);
    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
	     triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    
    for (const auto &cell :
	   triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();
    
    SolutionTransfer<dim> solution_trans(dof_handler);
    
    std::vector<Vector<double>> previous_solution(num_index);
    
    for(unsigned int i=0; i < num_index; i++)
      previous_solution[i] = solution[i];
    
    triangulation.prepare_coarsening_and_refinement();
    
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
    
    triangulation.execute_coarsening_and_refinement();
    
    setup_system();
    
    solution_trans.interpolate(previous_solution, solution);
    
    for(unsigned int i=0; i < num_index; i++)
      constraints.distribute(solution[i]);
    
  }

  
  
  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::solve(int index)
  {
    std::cout  << "Solving linear system... " << std::endl;

    /*
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix[index]);
    A_direct.vmult(solution[index], system_rhs[index]);
    */
    
    SolverControl solver_control(std::max<std::size_t>(10000,
						       system_rhs[index].size() / 10), 1e-6 * system_rhs[index].l2_norm());
    SolverGMRES<Vector<double>> solver(solver_control);
    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix[index], 1.0);
    solver.solve(system_matrix[index], solution[index], system_rhs[index], preconditioner);
    
    
    constraints.distribute(solution[index]);
  }





  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::solve_direct(int index)
  {
    std::cout  << "Solving linear system... " << std::endl;
                                                                                                                                                       
    SparseDirectUMFPACK A_direct;                                                                                                                            
    A_direct.initialize(system_matrix[index]);                                                                                                               
    A_direct.vmult(solution[index], system_rhs[index]);                                                                                                

    /*
    SolverControl solver_control(std::max<std::size_t>(1000,
                                                       system_rhs[index].size() / 10), 1e-6 * system_rhs[index].l2_norm());
    SolverGMRES<Vector<double>> solver(solver_control);
    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix[index], 1.0);
    solver.solve(system_matrix[index], solution[index], system_rhs[index], preconditioner);
    */

    constraints.distribute(solution[index]);
  }


  
  
  template <int dim>
  void AmplitudePhaseFieldCrystalProblem<dim>::output_results() const
  {
    if (time.get_step_number() % 10 != 0) // what do we do this ???????????????
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
    const unsigned int initial_global_refinement       = 4;
    const unsigned int n_adaptive_pre_refinement_steps = 2;
    const double min_coord = -25;                                                                                                   
    const double max_coord = 25;
    GridGenerator::hyper_cube(triangulation, min_coord, max_coord, true);                                                                     
    triangulation.refine_global(initial_global_refinement);

    unsigned int pre_refinement_step = 0;

    setup_system();
    /*    
    for(unsigned int i=0; i < num_index; i++)   
      VectorTools::project(dof_handler,
			   constraints,
			   QGauss<dim>(degree + 1),
			   RotatedIC<dim>(i),
			   old_solution[i]);*/
              
    while(pre_refinement_step < n_adaptive_pre_refinement_steps)
      {	
	for(unsigned int i=0; i < num_index; i++)
	  {
	    VectorTools::project(dof_handler,
				 constraints,
				 QGauss<dim>(degree + 1),
				 RotatedIC<dim>(i),
				 old_solution[i]);
	  }

	for(unsigned int i=0; i < num_index; i++)
          solution[i] = old_solution[i];

	std::cout << "Initial interation " <<  pre_refinement_step << std::endl;
	output_results();

	
	for(unsigned int i=0; i < num_index; i++)
	  {
	    time.set_desired_next_step_size(1.0);
	    assemble_system(i);
	    solve_direct(i);
	  }
	
	refine_mesh(initial_global_refinement,
		    initial_global_refinement +
		    n_adaptive_pre_refinement_steps);
	
	++pre_refinement_step;
      }

    //The following lines are very important!!!!!!
      for(unsigned int i=0; i < num_index; i++)
	old_solution[i] = solution[i];
      
      do
	{
	  //output at time zero
	  output_results();
	  
	  std::cout << "Timestep " << time.get_step_number() + 1 << std::endl;
	  
	  for(unsigned int i=0; i < num_index; i++)
	    {
	      time.set_desired_next_step_size(1.0);
	      assemble_system(i);
	      solve_direct(i);
	    }
	  
	  // output_results();
	  
	  refine_mesh(initial_global_refinement,                                                                                                
		      initial_global_refinement +                                                                                              
		      n_adaptive_pre_refinement_steps);    
	  
	  for(unsigned int i=0; i < num_index; i++)
	    old_solution[i] = solution[i];
	  
	  time.advance_time();
	  std::cout << "   Now at t=" << time.get_current_time()
		    << ", dt=" << time.get_previous_step_size() << '.'
		    << std::endl
		    << std::endl;
	}
      while (time.is_at_end() == false);
      
  }// end of run()
  
  
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
