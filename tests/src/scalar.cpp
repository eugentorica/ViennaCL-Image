/* =======================================================================
   Copyright (c) 2010, Institute for Microelectronics, TU Vienna.
   http://www.iue.tuwien.ac.at
                             -----------------
                     ViennaCL - The Vienna Computing Library
                             -----------------
                            
   authors:    Karl Rupp                          rupp@iue.tuwien.ac.at
               Florian Rudolf
               Josef Weinbub                      weinbub@iue.tuwien.ac.at

   license:    MIT (X11), see file LICENSE in the ViennaCL base directory
======================================================================= */

//
// *** System
//
#include <iostream>
#include <algorithm>
#include <cmath>

//
// *** ViennaCL
//
#include "viennacl/scalar.hpp"

//
// -------------------------------------------------------------
//
template <typename ScalarType>
ScalarType diff(ScalarType & s1, viennacl::scalar<ScalarType> & s2) 
{
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;

   NumericT s1 = 3.1415926;
   NumericT s2 = 2.71763;
   int s3 = 42;

   viennacl::scalar<NumericT> vcl_s1;
   viennacl::scalar<NumericT> vcl_s2;
   viennacl::scalar<NumericT> vcl_s3 = 1.0;
      
   vcl_s1 = s1;

   if( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: vcl_s1 = s1;" << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   vcl_s2 = s2;
   if( fabs(diff(s2, vcl_s2)) > epsilon )   
   {
      std::cout << "# Error at operation: vcl_s2 = s2;" << std::endl;
      std::cout << "  diff: " << fabs(diff(s2, vcl_s2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   vcl_s3 = s3;
   if( s3 != vcl_s3 ) 
   {
      std::cout << "# Error at operation: vcl_s3 = s3;" << std::endl;
      std::cout << "  diff: " << s3 - vcl_s3 << std::endl;
      retval = EXIT_FAILURE;
   }


   s1 += s2;
   vcl_s1 += vcl_s2;
   if( fabs(diff(s1, vcl_s1)) > epsilon ) 
   {
      std::cout << "# Error at operation: += " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 *= s3;
   vcl_s1 *= vcl_s3;

   if( fabs(diff(s1, vcl_s1)) > epsilon )   
   {
      std::cout << "# Error at operation: *= " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2;
   vcl_s1 -= vcl_s2;
   if( fabs(diff(s1, vcl_s1)) > epsilon )   
   {
      std::cout << "# Error at operation: -= " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 /= s3;
   vcl_s1 /= vcl_s3;

   if( fabs(diff(s1, vcl_s1)) > epsilon )  
   {
      std::cout << "# Error at operation: /= " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = vcl_s1;

   s1 = s2 + s3;
   vcl_s1 = vcl_s2 + vcl_s3;
   if( fabs(diff(s1, vcl_s1)) > epsilon )  
   {
      std::cout << "# Error at operation: + " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 - s3;
   vcl_s1 = vcl_s2 - vcl_s3;
   if( fabs(diff(s1, vcl_s1)) > epsilon )  
   {
      std::cout << "# Error at operation: - " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 * s3;
   vcl_s1 = vcl_s2 * vcl_s3;
   if( fabs(diff(s1, vcl_s1)) > epsilon )  
   {
      std::cout << "# Error at operation: * " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 / s3;
   vcl_s1 = vcl_s2 / vcl_s3;
   if( fabs(diff(s1, vcl_s1)) > epsilon )  
   {
      std::cout << "# Error at operation: / " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 + s3 * s2 - s3 / s1;
   vcl_s1 = vcl_s2 + vcl_s3 * vcl_s2 - vcl_s3 / vcl_s1;
   if( fabs(diff(s1, vcl_s1)) > epsilon )  
   {
      std::cout << "# Error at operation: + * - / " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   return retval;
}
//
// -------------------------------------------------------------
//
int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Scalar" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = 1.0E-5;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = 1.0E-6;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = 1.0E-7;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   if( viennacl::ocl::current_device().double_support() )
   {
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-10;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
         if( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-15;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
         if( retval == EXIT_SUCCESS )
            std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-20;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
         if( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-25;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
         if( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-30;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
         if( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-35;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
         if( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
   }
   return retval;
}
//
// -------------------------------------------------------------
//

