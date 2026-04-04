# Contents

| 1 | $\mathbf{W}\mathbf{h}$ | nat is signal processing, and why study it?               | 3 |
|---|------------------------|-----------------------------------------------------------|---|
|   | 1.1                    | Signals                                                   | 3 |
|   |                        | 1.1.1 Sketching signals                                   | 4 |
|   | 1.2                    | Systems                                                   | 4 |
|   |                        | 1.2.1 Abstract block diagram representation of systems    | 6 |
|   |                        | 1.2.2 An example of a discrete-time system - Mortgages    | 6 |
|   | 1.3                    | What we will learn in this course                         | 7 |
|   | 1.4                    | Why is learning this material important?                  | 8 |
|   | 1.5                    | What prior knowledge is needed to do well in this course? | 8 |
|   | 1.6                    | Problems                                                  | 8 |

2 CONTENTS

## Chapter 1

# What is signal processing, and why study it?

#### 1.1 Signals

A *signal* is any function that contains information. Examples include my speech, the Dow-Jones average that shows the closing value of blue chip stocks each day, fuel usage in a car as a function of the distance traveled, pictures, movies, and your GPA at the end of each semester. Let us notice a few things here.

- Functions depend on one or more independent variables. For example, speech is a function of time. So you might mathematically represent speech as s(t), where t is the independent variable. The signal may depend on more than one independent variable. A photograph is a function of two spatial variables that we will call the x and y directions. So, we may use a function p(x,y) to denote a photograph. A movie is a function of three variables, x and y directions, and time. So the movie may be represented as m(x,y,t).
- In this class we will consider predominantly functions of one variable. Even though this independent variable can be anything, we will call it time.
- In the examples above, we found two types of signals that were functions of time. In the case of speech, the signal exists over a continuous range of values of time. Such signals are called continuous-time signals. In the case of Dow-Jones averages, the signals were defined once per day at 4 pm eastern time. That is, the signals were defined only at discrete values of time. We will call such signals discrete-time signals. In the case of discrete-time signals, we will only study signals for which the time between adjacent samples are always the same, say T. In such cases, the nth sample in a discrete-time signal occurs at the time nT. In many cases we will omit the dependence on T and just call the signal x[n] rather than x[nT]. Note also that we have used square brackets instead of parenthesis for discrete-time signals. One other thing to notice: x[n] is defined only for integer values (can be positive, negative or zero) of n. The signal is not defined when n is a fractional number.
- The signal can take values from a continuous range or these values can take only discrete values. For example, the price of something we buy from a store usually cannot take values that are fractions of a penny. In this case the function can take only discrete values that are defined by integer multiples of cents. On the other hand, the fuel efficiency of a car (say

in miles/gallon) can ideally take any positive value. A continuous time signal that can take values in a continuous range is called an *analog* signal. A discrete-time signal that takes only values belonging to a discrete set is called a *digital* signal. We do not have special names for the other two cases. If we want to record a signal onto a digital computer, we can only do it if we have a digital signal. That is, to record an analog signal in a digital medium, we need to first sample the signal in the time domain, and also represent the values of the samples using a finite number of bits. Note that if we have a finite number of bits, we can only represent a number to its closest approximation possible with the bits available. That is, we have to quantize the signals to a discrete set of values possible using the available number of bits.

- One common way to create discrete-time signals is to use an analog to digital (A/D) converter to sample and record continuous-time signals. If you record your speech with your mobile phone, you are storing the information as a discrete-time signal. Music stored on CD-ROMs and DVDs are discrete-time signals. One of the reasons why discrete-time signals are so common place is because digital computers (and digital systems) can only handle discrete-time signals. Digital computers can also only handle discrete-valued signals. If a computer uses 10 bits (binary digits) to represent numbers, it can only process or store 2<sup>10</sup> = 1024 different values, and any function this computer calculates, records or process in any way must only take one of 1024 different values.
- So far we only talked about the how signals are classified according to the nature of the independent variable(s). Let us consider a continuous time signal x(t). The dependent variable x vary based on the situation. For example, if we record speech in a computer, we have a microphone converting the vibrations in the air to voltages and then the recorder stores the voltages as a function of time. In this case the unit of the function may be Volts (or perhaps millivolts). In the example of the fuel usage of a car, the units of the dependent variable may be gallons of gasoline used as a function of miles traveled. In these examples, there is only one dependent variable. A color picture is typically expressed using three colors at each location on the picture. This is a case of a signal with two independent variables and three dependent variables. In this class we will only consider signals that are functions of one independent variable and has one dependent variable.

#### 1.1.1 Sketching signals

We can sketch the signals on a graph to better visualize the information contained in them. It is typical to use the independent variable (for example, the time variable t) as the x-axis, and the dependent variable (say, x(t)) as the y-axis. it is important to clearly show the variables along the corresponding axes, and also provide sufficient information about the function on the graph. Discrete-time signals are typically plotted as a bar graph. Examples of plotting a continuous-time signal and a discrete-time signal are shown in Figures 1.1 and 1.2, respectively.

### 1.2 Systems

We refer to anything that processes signals as *systems*. Examples include RLC circuits used to filter continuous time signals, a DVD player that plays digital videos or music. We will refer to systems that process continuous-time signals as *continuous-time systems* ands systems that processes discrete-time signals as *discrete-time systems*. We can also have hybrid systems that

1.2. SYSTEMS 5

![](_page_4_Figure_1.jpeg)

Figure 1.1: Plot of a continuous-time signal.

![](_page_4_Figure_3.jpeg)

Figure 1.2: Plot of a discrete-time signal.

![](_page_5_Picture_2.jpeg)

Figure 1.3: Generic block diagram of a system.

accepts continuous time signals as inputs and outputs discrete-time systems or vice versa. Examples include analog-to-digital (A/D) converters and digital-to-analog (D/A) converters.

#### 1.2.1 Abstract block diagram representation of systems

When we study systems from a general perspective, it is not always necessary to know the application as long as we have a precise mathematical description of what the system does. For our studies, we will describe the system using equations that relate the input signal to the output signal. A block diagram that describes a generic system is shown in Figure 1.3. It simply states that the system transforms the input signal in some fashion and produces the output signal. To completely understand the system, we will need to know mathematically what this transformation does. We will learn about the mathematical tools that will allow us to characterize, design and implement continuous-time and discrete-time systems in this class.

#### 1.2.2 An example of a discrete-time system - Mortgages

Let us consider a mortgage loan for P dollars. Let us assume that the monthly payment is m[n] dollars each and the interest rate is R% anually resulting in a monthly fractional rate of r = R/1200. Let us assume that the mortgage payments are made at the beginning of each month, and that the initial month is denoted as the 0th month. Let the amount owed at the beginning of the nth month be p[n]. Clearly, p[0] = P is the amount owed when we take out the mortgage. Clearly, we have a discrete-time system here, and the amount owed at the beginning of the nth month can be calculated as the amount owed at the beginning of the (n-1)th month plus the interest accrued during the (n-1)th month minus the payment made at the beginning of the nth month. That is,

$$p[n] = p[n-1] + p[n-1]r - m[n]$$
  
=  $(1+r)p[n-1] - m[n]$  (1.1)

This input-output relationship (The input is the payments m[n] made each month, and the output is the amount owed p[n] during each month.) completely describes what a mortgage system does. Knowing this description, we can make decisions about whether we can afford a mortgage loan or not before signing the contract for the loan. Let us analyze the system a bit more to understand how we can use the information. Let us assume that we plan to make a constant payment of M dollars every month, and we wish to pay off the mortgage in N months. What is the monthly payment M in this case? To find out, we go back to (1.1) and write the following:

$$p[n] = (1+r)p[n-1] - M$$
  
=  $(1+r)\{(1+r)p[n-2] - M\} - M$ 

$$= (1+r)^2 p[n-2] - M(1+r) - M \tag{1.2}$$

We can substitute for p[n-2] in a similar manner to get

$$p[n] = (1+r)^{2}p[n-2] - (1+r)M - M$$

$$= (1+r)^{2}\{(1+r)p[n-3] - M\} - (1+r)M - M$$

$$= (1+r)^{3}p[n-3] - (1+r)^{2}M - (1+r)M - M$$
(1.3)

We can repeat this as many times as necessary to get

$$p[n] = (1+r)^n p[0] - (1+r)^{n-1} M - (1+r)^{n-2} M - \dots - M$$
$$= (1+r)^n p[0] - M \left\{ \sum_{k=0}^{n-1} (1+r)^k \right\}$$
(1.4)

Note that the second term in the above equation is a geometric series, and we have a closed form expression for the sum. Using the closed form expression, and recognizing that in the Nth month, the amount we owe is zero, and that p[0] = P, the amount of the loan, we can write (1.3) as

$$0 = P(1+r)^{N} - M \frac{1 - (1+r)^{N}}{1 - (1+r)}$$
(1.5)

Finally, we can solve for the monthly payment M as

$$M = \frac{Pr(1+r)^N}{(1+r)^N - 1} \tag{1.6}$$

If P = \$100,000, R = 6%, and N = 180 months, we see that r = 0.005, and substituting in (1.6), we get M = \$843.86. On the other hand, if we decide to pay off the amount only in 30 years, theh monthly payment is M = \$599.55. Even though this number looks much better than the amount we need to pay each month for a 15 year loan, the total payment in the two cases are \$215,828 for the 30 year loan and \$151,894 for the 15 year loan.

We can use the system description in (1.1) to find any other information about the mortgage system we seek. For example, we can find out the number of months it takes to pay of a loan for a fixed choice of the payment amount. (Note that if the monthly payment is less than the interest accrued in any month, the amount owed will grow. This example shows the usefulness of system analysis in a non-electrical engineering application. As soon as we translate the problem to a mathematical description, we can deal with the problem regardless of what the application is. It is because of this generality of signals and system analysis that we work in the mathematical domain.

#### 1.3 What we will learn in this course

In this class, we will learn the fundamentals of discrete-time signals and systems. The specific issues we will learn about will include:

- We will study what is known as linear and time-invariant systems in discrete-time. We will learn about how to analyze such systems (i.e., given some description of the system, figure out what it does to the input signals), how to implement such systems, and some intuitive ways to design and build such systems to meet specifications.
- We will learn to analyze discrete-time signals. Specifically, we will learn about Fourier transform techniques that allows us to decompose (split) signals into sinusoidal components.

#### 1.4 Why is learning this material important?

As we found earlier, regardless of what discipline you plan to engage in after you graduate, you will deal with information in some form. Understanding how to extract the relevant information from the data you have is critical, and knowing how to analyze and process signals will help you do that effectively.

#### 1.5 What prior knowledge is needed to do well in this course?

Some of the more important concepts you should know and I will assume throughout include:

- Complex numbers: Magnitude and phase of complex numbers; Polar form of complex numbers; Euler identity  $(e^{j\theta} = \cos(\theta) + j\sin(\theta))$ ; Expressing  $\cos(\theta)$  and  $\sin(\theta)$  using  $e^{j\theta}$ ; Operations on complex numbers addition, conjugation, multiplication, division, and exponentiation; Finding the Nth power and Nth roots of a complex number.
- Sinusoids: Complex sinusoidal functions  $x(t) = Ae^{j\omega t + \phi}$  and real sinusoidal functions  $x(t) = A\cos(\omega t + \phi)$ . Adding two sinusoids of the same frequency and finding the amplitude and phase of the resulting sinusoid. Basic trigonometric identities.
- Partial fraction expansions: Factoring polynomials; Equation for finding the (possibly complex) roots of quadratic polynomials; Finding the factors of higher order polynomials using Matlab; Partial fraction expansion for the case of non-repeating roots.
- Sequences and series: Taylor series expansion of polynomials; Power series expansion for  $e^{j\theta}$ ,  $\cos(x)$ ,  $\sin(x)$ ,  $\frac{1}{1+x}$  (|x| < 1),  $\ln(1+x)$  (|x| < 1); Sum of an infinite geometric series of the form

$$\sum_{n=0}^{\infty} ar^n = a + ar + ar^2 + ar^3 + \dots = \frac{a}{1-r} \quad ; \quad |r| < 1$$

and sum of a finite geometric series of the form

$$\sum_{n=0}^{N} ar^{n} = a + ar + ar^{2} + \dots + ar^{N} = a \frac{1 - r^{N+1}}{1 - r}$$

(You must know how to get these results. Also, how to find the sum if the range is some arbitrary  $N_1$  to  $N_2$  instead of 0 to N.)

#### 1.6 Problems

1. Find the partial fraction expansion for

$$H(s) = \frac{s+2}{s^3 + 3s^2 - s - 3}$$

2. Express the partial fraction expansion for

$$H(z) = \frac{1 - z^{-1}}{(1 - 0.4z^{-1})(1 - 0.3z^{-1})}$$

1.6. PROBLEMS 9

in the form

$$H(z) = \frac{A}{1 - 0.4z^{-1}} + \frac{B}{1 - 0.3z^{-1}}$$

where A and B are constants you need to determine.

3. Simplify

$$x(t) = 3\cos(2\pi 200t + \frac{\pi}{3}) + \sin(2\pi 200t + \frac{\pi}{4})$$

to the form

$$x(t) = \alpha \cos(2\pi 200t + \theta)$$

where  $\alpha$  and  $\theta$  are parameters you need to determine.

Hint: This problem may be easier to solve if you recognize that x(t) is the real part of

$$\tilde{x}(t) = 3e^{j(2\pi 200t + \frac{\pi}{3})} + e^{j(2\pi 200t - \frac{\pi}{2} + \frac{\pi}{4})}$$

(You must show that this is correct.) You may need a calculator to simplify this problem.

4. Find and sketch, on the complex plane, the 7 seventh roots of 128. Repeat the operation for 128j.

Hint: Express 7 as  $7e^{j2\pi k}$  for  $k=0,\ 1,\ 2,\ \cdots,\ 6$  (Why is this correct?) and find the roots for each case. The roots must all fall on a circle on the complex plane. How would you do similarly for 128j?

- 5. Write a simple Matlab program that implements the mortgage calculations described in this chapter (including equation (1.1)). The program should take as inputs the initial amount of the loan, annual percentage interest rate, number of the months for which the loan is taken, and provide as output the amount owed at the beginning of each month money is owed to the lender after the monthly payment is made, assuming that monthly payments are made in equal amounts. Use this program to determine the total savings possible by paying off a 30-year, \$300,000 loan at the annual interest rate of 4.5% in 15 years. (Assume that instead of the monthly payment for the 30-year loan, you will pay a fixed amount more each month to make the 15-year payoff possible.)
- 6. By differentiating the series expansion for  $e^x$ , show that  $\frac{d}{dx}e^x = e^x$ .

Hint: 
$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}$$
.

7. By directly multiplying the series expansions for  $e^x$  and  $e^y$ , show that  $e^x e^y = e^{x+y}$ .