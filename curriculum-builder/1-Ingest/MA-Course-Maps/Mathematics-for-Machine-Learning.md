# Mathematics for Machine Learning

## Course Description

Our Mathematics for Machine Learning course provides a comprehensive foundation of the essential mathematical tools required to study machine learning.
This course is divided into three main categories: linear algebra, multivariable calculus, and probability & statistics. The linear algebra section covers crucial machine learning fundamentals such as matrices, vector spaces, diagonalization, projections, singular value decomposition, and regression. The multivariable calculus section examines vector-valued functions, partial derivatives, and multiple integrals. Finally, the probability and statistics section covers random variables, point estimation, maximum likelihood, hypothesis testing, and confidence intervals.
On completing this course, students will be well-prepared for a university-level machine learning course that tackles concepts such as gradient descent, neural networks, backpropagation, support vector machines, naive Bayes classifiers, and Gaussian mixture models.

## Course Overview

After briefly looking at some essential set theory, logic, and vector geometry, students explore matrices in-depth. They will study Gaussian elimination, solve systems of equations, learn about determinants and their properties, and compute inverse matrices.
As part of this course, students perform a deep dive into vector spaces, exploring linear independence, subspaces, bases, dimension, rank, and nullity. Students will generalize key concepts to abstract vector spaces and inner product spaces. Various aspects of orthogonality in vector spaces are considered, including orthogonal sets, complements, orthogonal matrices, orthogonal projections, and the Gram-Schmidt process.
Students will learn how to find the eigenvectors of a matrix, compute a matrix diagonalization, and extend this understanding to symmetric matrices.
In addition, this course discusses various linear algebra applications relevant to machine learning, such as singular value decomposition, linear least-squares, regression, and principal component analysis.
A solid grasp of some key multivariable calculus concepts is needed to understand fundamental machine learning algorithms successfully. In this course, students will become well-versed in partial derivatives and gradient vectors (for gradient descent), the multivariable chain rule (essential for backpropagation), vector-valued functions, and generally, the differential calculus of maps between multi-dimensional vector spaces (which show up when machine learning models are represented using matrix notation). Students will also work with standard multivariable surfaces to build intuition for the concept of a loss surface of a machine learning model. The remainder of the multivariable calculus discusses double integrals, a crucial tool for fully grasping continuous probability distributions and related concepts.
On the probability and statistics side, students will unravel discrete and continuous random variables. They will familiarize themselves with probability density functions, random variable transformations, expectation, moments, and variance. Some important discrete and continuous probability distributions will be discussed in detail.
Students then extend their knowledge of random variables to include joint, marginal, and conditional probability distributions, sums and products of random variables, conditional expectations, and variances. Special attention will be given to combinations of normally distributed random variables and the bivariate normal distribution.
The statistics part of the course concludes with an in-depth study of parametric inference, exploring point estimation, maximum likelihood estimation, and hypothesis testing. Students will also learn to construct confidence intervals for various parameters, including means, proportions, variances, and regression coefficients.

## Course Outcomes

Upon successful completion of this course, students will have mastered the following:

- Develop a solid understanding of sets, logical quantifiers, vector geometry, and hyperbolic functions.
- Master determinants, Gaussian elimination, and learn how to compute inverse matrices efficiently.
- Develop a thorough understanding of vector spaces, including bases, dimension, rank, nullity, and the rank-nullity theorem.
- Understand the concept of diagonalization and how to apply it.
- Develop knowledge of orthogonality in vector spaces, including orthogonal sets, complements, and orthogonal projections.
- Learn bilinear and quadratic forms, including positive-definite and negative-definite quadratic forms.
- Understand singular value decomposition and its applications, including the pseudoinverse matrix.
- Understand and apply principal component analysis, including its connection with singular value decomposition.
- Solve linear least-squares problems, both with and without collinearity.
- Perform linear, polynomial, and multiple linear regressions.
- Compute partial derivatives, gradient vectors, Jacobian matrices, Hessian matrices, and understand their geometric interpretations.
- Understand vector-valued functions and their properties.
- Extend differential calculus to maps between multi-dimensional vector spaces.
- Identify and graph a variety of standard multivariable surfaces in 3D space.
- Use Riemann sums to approximate volumes.
- Calculate double integrals over rectangular and non-rectangular domains.
- Understand and apply fundamental probability concepts such as the law of total probability and Bayes' theorem.
- Work with discrete and continuous random variables.
- Apply transformations to discrete and continuous random variables, and compute the associated CDFs and PDFs of these distributions.
- Understand and compute expectation and variance for discrete and continuous random variables.
- Understand and work with joint distributions for discrete and continuous random variables.
- Calculate expectation for joint distributions.
- Understand and compute with covariance of random variables.
- Work with normally distributed random variables and understand their combinations.
- Apply point estimation and maximum likelihood.
- Understand and apply hypothesis testing and confidence intervals to situations in context.

## Course Content

### 1. Preliminaries 28 topics

**1.1. Introduction to Set Theory**

- 1.1.1. Special Sets
- 1.1.2. Statements and Predicates
- 1.1.3. Equivalent Sets
- 1.1.4. The Constructive Definition of a Set
- 1.1.5. The Conditional Definition of a Set
- 1.1.6. Describing Sets Using Set-Builder Notation
- 1.1.7. Describing Planar Regions Using Set-Builder Notation
- 1.1.8. Subsets

**1.2. Set Operations**

- 1.2.1. The Difference of Sets
- 1.2.2. Set Complements
- 1.2.3. The Cartesian Product
- 1.2.4. Visualizing Cartesian Products
- 1.2.5. Indexed Sets
- 1.2.6. Sets and Functions

**1.3. Properties of Sets**

- 1.3.1. Cardinality of Finite Sets
- 1.3.2. Infinite Sets
- 1.3.3. Interior and Boundary Points
- 1.3.4. Interiors and Boundaries of Sets
- 1.3.5. Open and Closed Sets

**1.4. Vector Geometry**

- 1.4.1. The Vector Equation of a Line
- 1.4.2. The Parametric Equations of a Line
- 1.4.3. The Cartesian Equation of a Line
- 1.4.4. The Vector Equation of a Plane
- 1.4.5. The Cartesian Equation of a Plane
- 1.4.6. The Parametric Equations of a Plane
- 1.4.7. The Intersection of Two Planes

**1.5. The Hyperbolic Functions**

- 1.5.1. The Hyperbolic Functions
- 1.5.2. Graphs of the Hyperbolic Functions

### 2. Matrices 26 topics

**2.1. Determinants**

- 2.1.1. The Determinant of an NxN Matrix
- 2.1.2. Finding Determinants Using Laplace Expansions
- 2.1.3. Basic Properties of Determinants
- 2.1.4. Further Properties of Determinants
- 2.1.5. Row and Column Operations on Determinants
- 2.1.6. Conditions When a Determinant Equals Zero

**2.2. Gaussian Elimination**

- 2.2.1. Systems of Equations as Augmented Matrices
- 2.2.2. Row Echelon Form
- 2.2.3. Solving Systems of Equations Using Back Substitution
- 2.2.4. Elementary Row Operations
- 2.2.5. Creating Rows or Columns Containing Zeros Using Gaussian Elimination
- 2.2.6. Solving 2x2 Systems of Equations Using Gaussian Elimination
- 2.2.7. Solving 2x2 Singular Systems of Equations Using Gaussian Elimination
- 2.2.8. Solving 3x3 Systems of Equations Using Gaussian Elimination
- 2.2.9. Identifying the Pivot Columns of a Matrix
- 2.2.10. Solving 3x3 Singular Systems of Equations Using Gaussian Elimination
- 2.2.11. Reduced Row Echelon Form
- 2.2.12. Gaussian Elimination For NxM Systems of Equations

**2.3. The Inverse of a Matrix**

- 2.3.1. Finding the Inverse of a 2x2 Matrix Using Row Operations
- 2.3.2. Finding the Inverse of a 3x3 Matrix Using Row Operations
- 2.3.3. Matrices With Easy-to-Find Inverses
- 2.3.4. The Invertible Matrix Theorem in Terms of 2x2 Systems of Equations
- 2.3.5. Triangular Matrices

**2.4. Affine Transformations**

- 2.4.1. Affine Transformations
- 2.4.2. The Image of an Affine Transformation
- 2.4.3. The Inverse of an Affine Transformation

### 3. Vector Spaces 20 topics

**3.1. Vectors in N-Dimensional Space**

- 3.1.1. Vectors in N-Dimensional Euclidean Space
- 3.1.2. Linear Combinations of Vectors in N-Dimensional Euclidean Space
- 3.1.3. Linear Span of Vectors in N-Dimensional Euclidean Space
- 3.1.4. Linear Dependence and Independence

**3.2. Subspaces of N-Dimensional Space**

- 3.2.1. Subspaces of N-Dimensional Space
- 3.2.2. Subspaces of N-Dimensional Space: Geometric Interpretation
- 3.2.3. The Column Space of a Matrix
- 3.2.4. The Null Space of a Matrix

**3.3. Bases of N-Dimensional Space**

- 3.3.1. Finding a Basis of a Span
- 3.3.2. Finding a Basis of the Column Space of a Matrix
- 3.3.3. Finding a Basis of the Null Space of a Matrix
- 3.3.4. Expressing the Coordinates of a Vector in a Given Basis
- 3.3.5. Writing Vectors in Different Bases
- 3.3.6. The Change-of-Coordinates Matrix
- 3.3.7. Changing a Basis Using the Change-of-Coordinates Matrix

**3.4. Dimension and Rank in N-Dimensional Space**

- 3.4.1. The Dimension of a Span
- 3.4.2. The Rank of a Matrix
- 3.4.3. The Dimension of the Null Space of a Matrix
- 3.4.4. The Invertible Matrix Theorem in Terms of Dimension, Rank and Nullity
- 3.4.5. The Rank-Nullity Theorem

### 4. Diagonalization of Matrices 12 topics

**4.1. Eigenvectors and Eigenvalues**

- 4.1.1. The Eigenvalues and Eigenvectors of a 2x2 Matrix
- 4.1.2. Calculating the Eigenvalues of a 2x2 Matrix
- 4.1.3. Calculating the Eigenvectors of a 2x2 Matrix
- 4.1.4. The Characteristic Equation of a Matrix
- 4.1.5. Calculating the Eigenvectors of a 3x3 Matrix With Distinct Eigenvalues
- 4.1.6. Calculating the Eigenvectors of a 3x3 Matrix in the General Case

**4.2. Diagonalization**

- 4.2.1. Diagonalizing a 2x2 Matrix
- 4.2.2. Diagonalizing a 3x3 Matrix With Distinct Eigenvalues
- 4.2.3. Diagonalizing a 3x3 Matrix in the General Case
- 4.2.4. Symmetric Matrices
- 4.2.5. Diagonalization of 2x2 Symmetric Matrices
- 4.2.6. Diagonalization of 3x3 Symmetric Matrices

### 5. Orthogonality & Projections 17 topics

**5.1. Inner Products**

- 5.1.1. The Dot Product in N-Dimensional Euclidean Space
- 5.1.2. The Norm of a Vector in N-Dimensional Euclidean Space
- 5.1.3. Introduction to Abstract Vector Spaces
- 5.1.4. Defining Abstract Vector Spaces
- 5.1.5. Inner Product Spaces

**5.2. Orthogonality**

- 5.2.1. Orthogonal Vectors in Euclidean Spaces
- 5.2.2. The Cauchy-Schwarz Inequality and the Angle Between Two Vectors
- 5.2.3. Orthogonal Complements
- 5.2.4. Orthogonal Sets in Euclidean Spaces
- 5.2.5. Orthogonal Matrices
- 5.2.6. Orthogonal Linear Transformations

**5.3. Orthogonal Projections**

- 5.3.1. Projecting Vectors Onto One-Dimensional Subspaces
- 5.3.2. The Components of a Vector with Respect to an Orthogonal or Orthonormal Basis
- 5.3.3. Projecting Vectors Onto Subspaces in Euclidean Spaces (Orthogonal Bases)
- 5.3.4. Projecting Vectors Onto Subspaces in Euclidean Spaces (Arbitrary Bases)
- 5.3.5. Projecting Vectors Onto Subspaces in Euclidean Spaces (Arbitrary Bases): Applications
- 5.3.6. The Gram-Schmidt Process for Two Vectors

### 6. Singular Value Decomposition 12 topics

**6.1. Quadratic Forms**

- 6.1.1. Bilinear Forms
- 6.1.2. Quadratic Forms
- 6.1.3. Change of Variables in Quadratic Forms
- 6.1.4. Positive-Definite and Negative-Definite Quadratic Forms
- 6.1.5. Constrained Optimization of Quadratic Forms
- 6.1.6. Constrained Optimization of Quadratic Forms: Determining Where Extrema are Attained

**6.2. Singular Value Decomposition**

- 6.2.1. The Singular Values of a Matrix
- 6.2.2. Computing the Singular Values of a Matrix
- 6.2.3. Singular Value Decomposition of 2x2 Matrices
- 6.2.4. Singular Value Decomposition of 2x2 Matrices With Zero or Repeated Eigenvalues
- 6.2.5. Singular Value Decomposition of Larger Matrices
- 6.2.6. Singular Value Decomposition and the Pseudoinverse Matrix

### 7. Applications of Linear Algebra 8 topics

**7.1. Principal Component Analysis**

- 7.1.1. Introduction to Principal Component Analysis
- 7.1.2. Computing Principal Components
- 7.1.3. The Connection Between PCA and SVD

**7.2. Linear Least-Squares Problems**

- 7.2.1. The Least-Squares Solution of a Linear System (Without Collinearity)
- 7.2.2. The Least-Squares Solution of a Linear System (With Collinearity)

**7.3. Linear Regression**

- 7.3.1. Linear Regression With Matrices
- 7.3.2. Polynomial Regression With Matrices
- 7.3.3. Multiple Linear Regression With Matrices

### 8. Multivariable Calculus 42 topics

**8.1. Quadric Surfaces and Cylinders**

- 8.1.1. Ellipsoids
- 8.1.2. Hyperboloids
- 8.1.3. Paraboloids
- 8.1.4. Elliptic Cones
- 8.1.5. Cylinders
- 8.1.6. Identifying Quadric Surfaces

**8.2. Partial Derivatives**

- 8.2.1. The Domain of a Multivariable Function
- 8.2.2. Level Curves
- 8.2.3. Limits and Continuity of Multivariable Functions
- 8.2.4. Introduction to Partial Derivatives
- 8.2.5. Computing Partial Derivatives Using the Rules of Differentiation
- 8.2.6. Geometric Interpretations of Partial Derivatives
- 8.2.7. Partial Differentiability of Multivariable Functions
- 8.2.8. Higher-Order Partial Derivatives
- 8.2.9. Equality of Mixed Partial Derivatives
- 8.2.10. Tangent Planes to Surfaces
- 8.2.11. Linearization of Multivariable Functions
- 8.2.12. The Multivariable Chain Rule

**8.3. Vector-Valued Functions**

- 8.3.1. The Domain of a Vector-Valued Function
- 8.3.2. Tangent Vectors and Tangent Lines to Curves
- 8.3.3. The Gradient Vector
- 8.3.4. Directional Derivatives
- 8.3.5. The Multivariable Chain Rule in Vector Form

**8.4. Differentiation**

- 8.4.1. The Jacobian
- 8.4.2. The Inverse Function Theorem
- 8.4.3. The Jacobian of a Three-Dimensional Transformation
- 8.4.4. The Derivative of a Multivariable Function
- 8.4.5. The Second Derivative of a Multivariable Function
- 8.4.6. Second-Degree Taylor Polynomials of Multivariable Functions

**8.5. Approximating Volumes With Riemann Sums**

- 8.5.1. Partitions of Intervals
- 8.5.2. Calculating Double Summations Over Partitions
- 8.5.3. Approximating Volumes Using Lower Riemann Sums
- 8.5.4. Approximating Volumes Using Upper Riemann Sums
- 8.5.5. Lower Riemann Sums Over General Rectangular Partitions
- 8.5.6. Upper Riemann Sums Over General Rectangular Partitions
- 8.5.7. Defining Double Integrals Using Lower and Upper Riemann Sums

**8.6. Double Integrals**

- 8.6.1. Double Integrals Over Rectangular Domains
- 8.6.2. Double Integrals Over Non-Rectangular Domains
- 8.6.3. Properties of Double Integrals
- 8.6.4. Type I and II Regions in Two-Dimensional Space
- 8.6.5. Double Integrals Over Type I Regions
- 8.6.6. Double Integrals Over Type II Regions

### 9. Probability & Random Variables 40 topics

**9.1. Probability**

- 9.1.1. Extending the Law of Total Probability
- 9.1.2. Bayes' Theorem
- 9.1.3. Extending Bayes' Theorem

**9.2. Random Variables**

- 9.2.1. Probability Density Functions of Continuous Random Variables
- 9.2.2. Calculating Probabilities With Continuous Random Variables
- 9.2.3. Continuous Random Variables Over Infinite Domains
- 9.2.4. Cumulative Distribution Functions for Continuous Random Variables
- 9.2.5. Approximating Discrete Random Variables as Continuous
- 9.2.6. Simulating Random Observations

**9.3. Transformations of Random Variables**

- 9.3.1. One-to-One Transformations of Discrete Random Variables
- 9.3.2. Many-to-One Transformations of Discrete Random Variables
- 9.3.3. The Distribution Function Method
- 9.3.4. The Change-of-Variables Method for Continuous Random Variables
- 9.3.5. The Distribution Function Method With Many-to-One Transformations

**9.4. Expectation**

- 9.4.1. Expected Values of Discrete Random Variables
- 9.4.2. Properties of Expectation for Discrete Random Variables
- 9.4.3. Moments of Discrete Random Variables
- 9.4.4. Variance of Discrete Random Variables
- 9.4.5. Properties of Variance for Discrete Random Variables
- 9.4.6. Expected Values of Continuous Random Variables
- 9.4.7. Moments of Continuous Random Variables
- 9.4.8. Variance of Continuous Random Variables
- 9.4.9. The Rule of the Lazy Statistician

**9.5. Discrete Probability Distributions**

- 9.5.1. The Bernoulli Distribution
- 9.5.2. Modeling With the Binomial Distribution
- 9.5.3. The CDF of the Binomial Distribution
- 9.5.4. Mean and Variance of the Binomial Distribution
- 9.5.5. The Discrete Uniform Distribution
- 9.5.6. Modeling With Discrete Uniform Distributions
- 9.5.7. Mean and Variance of the Discrete Uniform Distribution
- 9.5.8. The Poisson Distribution
- 9.5.9. Modeling With the Poisson Distribution
- 9.5.10. The CDF of the Poisson Distribution

**9.6. Continuous Probability Distributions**

- 9.6.1. The Continuous Uniform Distribution
- 9.6.2. Mean and Variance of the Continuous Uniform Distribution
- 9.6.3. Modeling With Continuous Uniform Distributions
- 9.6.4. The Gamma Function
- 9.6.5. The Chi-Square Distribution
- 9.6.6. The Student's T-Distribution
- 9.6.7. The Exponential Distribution

### 10. Combining Random Variables 29 topics

**10.1. Distributions of Two Discrete Random Variables**

- 10.1.1. Double Summations
- 10.1.2. Joint Distributions for Discrete Random Variables
- 10.1.3. Marginal Distributions for Discrete Random Variables
- 10.1.4. Independence of Discrete Random Variables
- 10.1.5. Conditional Distributions for Discrete Random Variables
- 10.1.6. The Joint CDF of Two Discrete Random Variables

**10.2. Distributions of Two Continuous Random Variables**

- 10.2.1. Joint Distributions for Continuous Random Variables
- 10.2.2. Marginal Distributions for Continuous Random Variables
- 10.2.3. Independence of Continuous Random Variables
- 10.2.4. Conditional Distributions for Continuous Random Variables
- 10.2.5. The Joint CDF of Two Continuous Random Variables
- 10.2.6. Properties of the Joint CDF of Two Continuous Random Variables

**10.3. Expectation for Joint Distributions**

- 10.3.1. Expected Values of Sums and Products of Random Variables
- 10.3.2. Variance of Sums of Independent Random Variables
- 10.3.3. Computing Expected Values From Joint Distributions
- 10.3.4. Conditional Expectation for Discrete Random Variables
- 10.3.5. Conditional Variance for Discrete Random Variables
- 10.3.6. Conditional Expectation for Continuous Random Variables
- 10.3.7. Conditional Variance for Continuous Random Variables
- 10.3.8. The Rule of the Lazy Statistician for Two Random Variables

**10.4. Covariance of Random Variables**

- 10.4.1. The Covariance of Two Random Variables
- 10.4.2. Variance of Sums of Random Variables
- 10.4.3. The Correlation Coefficient for Two Random Variables
- 10.4.4. The Covariance Matrix

**10.5. Normally Distributed Random Variables**

- 10.5.1. Normal Approximations of Binomial Distributions
- 10.5.2. Combining Two Normally Distributed Random Variables
- 10.5.3. Combining Multiple Normally Distributed Random Variables
- 10.5.4. I.I.D Normal Random Variables
- 10.5.5. The Bivariate Normal Distribution

### 11. Parametric Inference 32 topics

**11.1. Point Estimation**

- 11.1.1. The Sample Mean
- 11.1.2. Sampling Distributions
- 11.1.3. Variance of Sample Means
- 11.1.4. The Sample Variance
- 11.1.5. Sample Means From Normal Populations
- 11.1.6. The Central Limit Theorem
- 11.1.7. Sampling Proportions From Finite Populations
- 11.1.8. Point Estimates of Population Proportions
- 11.1.9. The Sample Covariance Matrix

**11.2. Maximum Likelihood**

- 11.2.1. Product Notation
- 11.2.2. Logarithmic Differentiation
- 11.2.3. Likelihood Functions for Discrete Probability Distributions
- 11.2.4. Log-Likelihood Functions for Discrete Probability Distributions
- 11.2.5. Likelihood Functions for Continuous Probability Distributions
- 11.2.6. Log-Likelihood Functions for Continuous Probability Distributions
- 11.2.7. Maximum Likelihood Estimation

**11.3. Hypothesis Testing**

- 11.3.1. Introduction to Hypothesis Testing
- 11.3.2. Hypothesis Tests for the Rate of a Poisson Distribution
- 11.3.3. Critical Regions for Left-Tailed Hypothesis Tests
- 11.3.4. Critical Regions for Right-Tailed Hypothesis Tests
- 11.3.5. Two-Tailed Hypothesis Tests
- 11.3.6. Type I and Type II Errors
- 11.3.7. Hypothesis Tests for One Mean: Known Population Variance
- 11.3.8. Hypothesis Tests for One Mean: Unknown Population Variance
- 11.3.9. Hypothesis Tests for Two Means: Known Population Variances

**11.4. Confidence Intervals**

- 11.4.1. Confidence Intervals for One Mean: Known Population Variance
- 11.4.2. Confidence Intervals for One Mean: Unknown Population Variance
- 11.4.3. Confidence Intervals for One Proportion
- 11.4.4. Confidence Intervals for Two Means: Known and Unequal Population Variances
- 11.4.5. Confidence Intervals for One Variance
- 11.4.6. Confidence Intervals for Linear Regression Slope Parameters
- 11.4.7. Confidence Intervals for Linear Regression Intercept Parameters
