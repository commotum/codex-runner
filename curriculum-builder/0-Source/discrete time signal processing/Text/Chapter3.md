# Contents

| 3 | Line | r and time-invariant systems | ; |
|---|------|------------------------------|---|
|   | 3.1  | formal definitions           |   |
|   |      | .1.1 Linear systems          |   |
|   |      | .1.2 Time-invariant systems  |   |
|   | 3.2  | llustrative examples         |   |
|   | 3.3  | Problems                     |   |

2 CONTENTS

### Chapter 3

## Linear and time-invariant systems

In this chapter, we formally define linear and time-invariant systems, and study many properties of such systems. In very simple terms, linear systems obey the superposition principle. Time-invariant systems do not change their properties with time. Almost all our study of systems in this course involves linear, time-invariant (lti) systems. The main reason for this emphasis is their usefulness in a large variety of applications. Furthermore, all lti systems can be characterized through the concepts of transfer function, frequency response, impulse response function and convolution. Consequently, we can study the principles of analyzing, designing and implementing lti systems in a unified manner.

#### 3.1 Formal definitions

We formally define linear and time-invariant systems before considering several examples.

#### 3.1.1 Linear systems

A linear system satisfies two fundamental properties of homogeneity and additivity. Let y[n] be the output of a system when its input is x[n]. The system is homogeneous if and only if its output is cy[n] whenever its input is cx[n] for any constant c. Let  $y_1[n]$  and  $y_2[n]$  denote the outputs of the system when its inputs are  $x_1[n]$  and  $x_2[n]$ , respectively. The system is additive if and only if its output signal is  $y_1[n] + y_2[n]$  whenever its input signal is  $x_1[n] + x_2[n]$ . The two requirements of homogeneity and additivity may be combined to create one condition that defines linearity as follows: Let  $y_1[n]$  and  $y_2[n]$  denote the outputs of the system when its inputs are  $x_1[n]$  and  $x_2[n]$ , respectively, and let  $\alpha$  and  $\beta$  be two arbitrary constants. The system is linear if and only if its output is  $\alpha y_1[n] + \beta y_2[n]$  whenever its input is  $\alpha x_1[n] + \beta x_2[n]$ . In other words, linear systems obey the superposition principle.

Something to think about: It might appear that all additive systems must be homogeneous and vice versa. However this is not true. An example of a system that is additive, but not homogeneous is one whose output is the real part of a complex-valued input signal. You should verify that  $\text{Real}\{cx[n]\} \neq c\text{Real}\{x[n]\}$ , where c is a complex-valued constant, even though  $\text{Real}\{x_1[n]+x_2[n]\} = \text{Real}\{x_1[n]\} + \text{Real}\{x_2[n]\}$ . A system with input-output relationship given by

$$y(t) = \frac{x[n-1]x[n+1]}{x[n]}$$

is homogeneous, but not additive.

![](_page_3_Figure_2.jpeg)

Figure 3.1: Illustration of time-invariance. To be changed to a discrete-time example.

#### 3.1.2 Time-invariant systems

Let y[n] be the output of a system when its input is x[n]. This system is time-invariant if and only if its output is y[n-k] whenever its input is x[n-k]. That is, a system is time-invariant if and only if its output is delayed by k samples whenever its input is delayed by k samples. Time-invariant systems do not change its properties with time. The timing issues relating the input and output of a time-invariant system is shown in Figure 3.1.

A linear, time-invariant system is one that is both linear and time-invariant.

Something to think about: It may be possible to find examples of signals for which the additivity, homogeneity and time-invariance properties hold. However, this is not enough to prove that the system satisfies these properties. To do so, we must show that for every input signal possible, the properties are satisfied. That is, the proof should involve arbitrary (and general) signals. On the other hand, it is enough to find one signal for which the system does not satisfy the property in question to prove that it is not linear or time-invariant.

### 3.2 Illustrative examples

#### Exercise 3.1

Determine if the systems with the following input-output relationships are linear and/or time-invariant:

- 1. y[n] = x[2n]
- 2.  $y[n] = \log(x[n])$

3. 
$$y[n] = \frac{1}{M} \sum_{m=n-M+1}^{n} x[m]$$

4. 
$$y[n] = x[-n]$$

5. 
$$y[n] = -\sum_{m=1}^{N} a_m y[n-m] + \sum_{k=0}^{M} b_k x[n-k]$$

You can assume that the system is initially at rest, i.e., all initial conditions are zero.

6. 
$$y[n] = \begin{cases} x[n] & ; |x[n]| < L \\ L & ; x[n] \ge L \\ -L & ; x[n] \le -L \end{cases}$$

In this example, assume that the system accepts only real-valued inputs and produces realvalued outputs.

7. 
$$y[n] = \sum_{m=-\infty}^{n} x[m] \lambda^{(n-m)}$$
. Here,  $0 < \lambda < 1$ .

8. 
$$y[n] = \sum_{k=-\infty}^{\infty} x[k]h[n-k]$$
 where  $h[n]$  is an arbitrary but known signal.

#### Answers:

1. Recall that this system simply subsamples the input signal x[n] by discarding every other sample.

Check for linearity: Let the output of the system for two different inputs  $x_1[n]$  and  $x_2[n]$ be  $y_1[n]$  and  $y_2[n]$ , respectively. Let a new input  $x_3[n] = \alpha x_1[n] + \beta x_2[n]$ , where  $\alpha$  and  $\beta$  are arbitrary constants, be processed by the system. Then the output of this system is  $y_3[n] = \alpha x_1[2n] + \beta x_2[2n] = \alpha y_1[n] + \beta y_2[n]$  implying that the system is linear.

Check for time-invariance: Consider an input signal x[n] that produces the output y[n] = x[n]. Now let us consider a shifted input given by  $x_1[n] = x[n-M]$ , where M is an arbitrary time shift. The output of the system for this input is  $y_1[n] = x_1[2n] = x[2n-M]$ . When you delay the output y[n] by M samples, we get  $\tilde{y}[n] = y[n-M] = x[2(n-M)] = x[2n-2M]$ . For the system to be time-invariant,  $y_1[n]$  and  $\tilde{y}[n]$  must be the same. That is not the case here, and therefore the system is not time-invariant.

2. This time we will check additivity and homogeneity separately to test for linearity.

Check for additivity: Using  $\mathcal{T}\{x[n]\}$  to denote the output of the system when the input is x[n], we see that  $\mathcal{T}\{x_1[n] + x_2[n]\} = \log(x_1[n] + x_2[n]) \neq \log(x_1[n]) + \log(x_2[n]) = \mathcal{T}\{x_1[n]\} + \log(x_2[n]) = \mathcal{T}\{x_1[n]\} + \log(x_2[n]) = \mathcal{T}\{x_1[n]\} + \log(x_2[n]) = \mathcal{T}\{x_1[n]\} + \log(x_2[n]) = \mathcal{T}\{x_1[n]\} = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]) = \log(x_1[n]$  $\mathcal{T}\{x_2[n]\}$ , implying that the system is not additive. Since one of additivity and homogeneity is not satisfied, the system is not linear (or nonlinear).

Check for homogeneity: Even though we do not need to check for homogeneity to check for linearity since we already know from the additivity test that the system is not linear, we will go through the steps to become familiar with the process.

We can easily see that  $\mathcal{T}\{\alpha x[t]\} = \log(\alpha x[n]) = \log(\alpha) + \log(x[n]) = \log(\alpha) + \mathcal{T}\{x[n]\} \neq 0$  $\alpha \mathcal{T}\{x[n]\}$ , which implies that the system is not homogeneous either.

Check for time-invariance: Let  $y[n] = \mathcal{T}\{x[n]\}, x_1[n] = x[n-m], \text{ and } y_1[n] = \mathcal{T}\{x_1[n]\}.$  The system is time-invariant if and only if  $y_1[n] = y[n-m]$ .

Now,  $y[n] = \log(x[n])$  and  $y[n-m] = \log(x[n-m])$ . Looking at  $y_1[n] = \log(x[n-m]) =$ y[n-m], we see that the condition for time-invariance is satisfied. Thus we have a timeinvariant, but nonlinear system.

3. This system produces, at time sample n, the sliding average of the input signal over the most recent M samples at the output. It is important to recognize the role of the variables n and m in the system description. The variable n represents the time at which the output y[n] is evaluated. Consequently, we are unable to use n as the variable of summation in the description of the transformation performed by the system. In order to perform the summation operation without introducing difficulties, it is normal to define the input signal and any system function using a dummy variable, for example, m. Thus, we will use x[m] to represent the input signal in the integration.

Check for linearity: The system is linear since

$$\frac{1}{M} \sum_{m=n-M+1}^{n} \left\{ \alpha x_1[n] + \beta x_2[m] \right\} = \alpha \frac{1}{M} \sum_{m=n-M+1}^{n} x_1[m] + \beta \frac{1}{M} \sum_{m=n-M+1}^{n} x_2[m]$$

Check for time-invariance: The output of the system when the input is x[n-p] is

$$\tilde{y}[n] = \frac{1}{M} \sum_{m=n-M+1}^{n} x[m-p]$$

Let us apply a change of variable q = m - p to the above expression. The output signal  $\tilde{y}[n]$  is then given by

$$\tilde{y}[n] = \frac{1}{M} \sum_{q=n-M+1-p}^{(n-p)} x[m]$$

Now.

$$y[n-p] = \frac{1}{M} \sum_{n-p-M+1}^{(n-p)} x[m]$$

Comparing the expressions for  $\tilde{y}[n]$  and y[n-p], we conclude that they are identical, and therefore the system is time-invariant.

- 4. It is straightforward to show that the system is linear. To check for time-invariance, we note that the output of the system when the input is x[n-m] is  $\tilde{y}[n] = x[-n-m]$ . Now,  $y[n-m] = x[-(n-m)] = x[-n+m] \neq \tilde{y}[n]$ , and therefore, the system is time-varying.
- 5. Additivity: Let  $y_1[n]$  and  $y_2[n]$  represent the output of the system when  $x_1[n]$  and  $x_2[n]$ , are respectively the input signals. That is,  $y_1[n] = -\sum_{m=1}^{N} a_m y_1[n-m] + \sum_{k=0}^{M} b_k x_1[n-k]$  and

$$y_2[n] = -\sum_{m=1}^N a_m y_2[n-m] + \sum_{k=0}^M b_k x_2[n-k]$$
. Adding both sides of the two equations, we get

$$y_1[n] + y_2[n] = -\sum_{m=1}^{N} a_m(y_1[n-m] + y_2[n-m]) + \sum_{k=0}^{M} b_k(x_1[n-k] + x_2[n-k])$$

Let  $\tilde{y}[n] = y_1[n] + y_2[n]$  and  $\tilde{x}[n] = x_1[n] + x_2[n]$ . Substituting these in the above equation, we get

$$\tilde{y}[n] = -\sum_{m=1}^{N} a_m \tilde{y}[n-m] + \sum_{k=0}^{M} b_k \tilde{x}[n-k]$$

Indicating that the system that outputs  $\tilde{y}[n]$  when  $\tilde{x}[n]$  is the input has the same input-output relationship as the original equation for the problem. This means that when  $x_1[n] + x_2[n]$  is the input signal, the output is  $y_1[n] + y_2[n]$  and the system is additive.

Homogeneity: Let us multiply both sides of the input-output relationship with an arbitrary constant c to get

$$cy[n] = -\sum_{m=1}^{N} a_m cy[n-m] + \sum_{k=0}^{M} b_k cx[n-k]$$

As before, let us define  $\tilde{x}[n] = cx[n]$  and  $\tilde{y}[n] = cy[n]$ , and substitute the new variables in the above equation. This gives

$$\tilde{y}[n] = -\sum_{m=1}^{N} a_m \tilde{y}[n-m] + \sum_{k=0}^{M} b_k \tilde{x}[n-k]$$

indicating that the system is identical to the one in our problem. This means that the output of the system is cy[n] when its input is cx[n] indicating that the system is homogeneous. Combining the two results, it follows that the system in our problem is linear.

Time invariance: The output y[n] and the input x[n] of the system in our problem are related through the equation

$$y[n] = -\sum_{m=1}^{N} a_m y[n-m] + \sum_{k=0}^{M} b_k x[n-k]$$

We can delay the output signal by p samples by changing n to n-p on both sides of this equation, which gives

$$y[n-p] = -\sum_{m=1}^{N} a_m y[n-p-m] + \sum_{k=0}^{M} b_k x[n-p-k]$$

Defining  $\tilde{y}[n] = y[n-p]$  and  $\tilde{x}[n] = x[n-p]$ , it follows that  $\tilde{y}[n-m] = y[n-m-p]$  and  $\tilde{x}[n-k] = x[n-k-p]$ . Substituting these into the above equation gives

$$\tilde{y}[n] = -\sum_{m=1}^{N} a_m \tilde{y}[n-m] + \sum_{k=0}^{M} b_k \tilde{x}[n-k]$$

which represents the same input-output relationship as our problem. This means that when  $\tilde{x}[n] = x[n-p]$  is the input to the system, the output is  $\tilde{y}[n] = y[n-p]$ , thus satisfying the time-invariance property. Thus, the system in this problem is time-invariant.

*Discussion*: This example shows that any system whose input and output signals are related through a ordinary difference equation with constant coefficients is linear and time-invariant. A large fraction of discrete-time linear, time-invariant systems we will study are of this form.

6. This is a hard limiter, whose output is forced to lie in the range  $-L \leq |y[n]| \leq L$ . Intuitively, we would think that a limiter cannot be linear, but its properties do not change with time. We will analyze the system more formally now.

Check for linearity: Consider an input signal |x[n]| < L and a constant c such that |cx[n]| > L. Clearly, the output of the system when its input is cx[n] is not the same as c times the output of the system when the input is x[n]. This shows that the system in nonlinear.

Check for time-invariance: Let  $\mathcal{T}$  represent the system so that  $y[n] = \mathcal{T}\{x[n]\}$  is the output of the system. Then,

$$\mathcal{T}\{x[n-m]\} = \begin{cases} x[n-m] & ; & |x[n-m]| < L \\ L & ; & x[n-m] \ge L \\ -L & ; & x[n-m] \le -L \end{cases}$$

Comparing the above to y[n], we can easily see that  $\mathcal{T}\{x[n-m]\}=y[n-m]$  indicating that the system is indeed time-invariant.

7. Check for linearity: Let  $y_1[n]$  and  $y_2[n]$  represent the outputs of the system when the inputs are  $x_1[n]$  and  $x_2[n]$ , respectively. The output  $y_3[n]$  of the system when its input is  $\alpha x_1[n] + \beta x_2[n]$  is

$$y_{3}[n] = y[n] = \sum_{k=-\infty}^{\infty} \{\alpha x_{1}[k] + \beta x_{2}[k]\} h[n-k])$$

$$= \alpha \sum_{k=-\infty}^{\infty} x_{1}[k]h[n-k] + \beta \sum_{k=-\infty}^{\infty} x_{2}[k]h[n-k]$$

$$= \alpha y_{1}[n] + \beta y_{2}[n]$$

demonstrating linearity of the system.

Check for time-invariance: As before, let y[n] and  $\tilde{y}[n]$  be the outputs of the system when its inputs are x[n] and x[n-m], respectively. We need to show that  $\tilde{y}[n] = y[n-m]$ .

$$\tilde{y}[n] = \sum_{k=-\infty}^{\infty} x[k-m]h[n-k]$$

Let us use a change of variable p = k - m, implying that n - k = n - p - m. Substituting these in the above equation, we get

$$\tilde{y}[n] = \sum_{n=-\infty}^{\infty} x[p]h[(n-p) - m] = y[n-p]$$

indicating that the system is indeed time-invariant.

Something for the future:

- a. The input-output relationship in this example is the *convolution sum* of the input signal x[n] with an arbitrary (but known) function h[n]. Later on, we will find that all linear, time-invariant systems perform convolution. Here, we showed that convolution is indeed a linear and time-invariant operation.
- b. Let us find the output of the system when the input is the unit impulse function  $\delta[n]$ . We will denote this response by  $y_{\delta}[n]$ . We substitute  $\delta[n]$  for x[n] in the input-output relationship to get

$$y_{\delta}[n] = \sum_{k=-\infty}^{\infty} \delta[k]h[n-k] = h[n]$$

the second equality came about because  $\delta[k]h[n-k]$  is non-zero only when k=0, and for this value of k, the product term is

$$\delta[0]h[n] = h[n]$$

Something to think about: Show, by direct calculation of the output of Problem 7 when its input is  $\delta[n]$  that the unit impulse response signal is  $h[n] = \lambda^n u[n]$ .

3.3. PROBLEMS 9

#### 3.3 Problems

1. Consider the block diagram in Figure 3.2.

![](_page_8_Picture_3.jpeg)

Figure 3.2: Block diagram of the system in Problem 1.

- (a) Express the input output relationship of the system mathematically.
- (b) Is the system linear and/or time-invariant?
- 2. Determine if the systems with the following input-output relationships are linear and/or time invariant. For linearity, check homogeneity and additivity separately.
  - (a)  $y[n] = \cos(2x[n])$
  - (b) y(n) = x[n]x[n-1]
  - (c)  $y(t) = x(t)\cos(10t)$
- 3. The unit impulse response of a discrete-time, linear, time-invariant system is given by

$$h[n] = \begin{cases} 1 & ; & n = 0 \\ 2 & ; & n = 1 \\ -1 & ; & n = 2 \\ 0 & ; & \text{Otherwise.} \end{cases}$$

Show from basic principles (definitions of linearity and time-invariance and the decomposition of input signals using (discrete-time) impulse functions) that the output of this system to any input signal x[n] is given by

$$y[n] = x[n] + 2x[n-1] - x[n-2]$$

- 4. The response of a linear time-invariant system to inputs  $f_1[n]$  and  $f_2[n]$  are, respectively, (n+1)u[n] and  $n^2u[n-1]$ . Find the mathematical expressions for system responses to the following input signals?
  - (a)  $f[n] = 2f_1[n] 3f_2[n]$ .
  - (b)  $g[n] = 3f_1[n-1]$ .
  - (c)  $h[n] = 5f_1[n+1] 4f_2[n-5].$
- 5. Consider a discrete-time, linear, time-invariant system whose response to a unit step signal u[n] is given by

$$p[n] = 0.6^n u[n]$$

Find the output of the system when its input is

$$x[n] = \begin{cases} 1 & ; & n = 0 \\ 2 & ; & n = 1 \\ 1 & ; & n = 2 \\ 0 & ; & \text{otherwise} \end{cases}$$

Hint: Write x[n] as a linear combination of delayed and shifted step functions. Using the fact that  $\delta[n] = u[n] - u[n-1]$  may be useful.

6. Determine, from basic principles, if the system with the following input-output relationship is linear and/or time-invariant:

$$y[n] = 3x[n] + 2x[n-1] + x[n-2] + 2$$

7. Determine, from basic principles, if the discrete-time system with the following input-output relationship is linear and/or time-invariant.

$$y[n] = 2^{-n}x[n-3]$$