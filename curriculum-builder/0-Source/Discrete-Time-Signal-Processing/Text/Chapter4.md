# Contents

| 4 | Con | volution                                                         | 3  |
|---|-----|------------------------------------------------------------------|----|
|   | 4.1 | Signal decomposition using impulse functions and its implication | 3  |
|   | 4.2 | Another form for the convolution sum                             | 4  |
|   | 4.3 | How to compute the convolution sum                               | 4  |
|   |     | 4.3.1 Examples in convolution                                    |    |
|   | 4.4 | Causal systems                                                   | 11 |
|   |     | 4.4.1 Convolution sum for causal systems                         | 11 |
|   | 4.5 | Realization of FIR filters                                       | 11 |
|   | 4.6 | Bounded-Input, Bounded-Output Stable Systems                     | 14 |
|   | 4.7 | Problems                                                         | 16 |

2 CONTENTS

# Chapter 4

## Convolution

We start with a property of the unit impulse function that we studied in Chapter 2.

## 4.1 Signal decomposition using impulse functions and its implication

Recall that

$$x[n] = \sum_{m = -\infty}^{\infty} x[m]\delta[n - m]$$
(4.1)

This result states that any signal can be broken up into a sum involving only impulse functions. Each term of the summation of the form  $x[m]\delta[n-m]$  is a time domain signal (a function of n) that extends from  $n=\infty$  to  $\infty$ , but has only one non-zero sample, and that sample is at time n=m. Let us consider just  $x_m[n]=x[m]\delta[n-m]$ , the mth signal in the summation above. We wish to find the output of a linear and time-invariant system  $\mathcal H$  with unit impulse response signal given by h[n] when its input signal is  $x_m[n]$ . For this signal, x[m] simply represents a constant value that is the value (amplitude) of the impulse at time n=m. Specifically, x[m] is a constant, and not a function that changes with time n. To find the output of the system, we can employ the time-invariance of the lti system to show that

$$\mathcal{H}\{\delta[n-m]\} = h[n-m] \tag{4.2}$$

This is simply a restatement of the fact that the system response to a delayed unit impulse signal is simply the same as the output of the system for a unit impulse signal (occurring at time n = 0), but delayed by the same number of samples as the impulse is delayed at the input.

Now we can use the homogeneity property of linear systems to show that when  $\delta[n-m]$  is amplified (or attenuated) by a constant value x[m], the output of the system is the original output multiplied by the same constant. That is,

$$\mathcal{H}\left\{x[m]\delta[n-m]\right\} = x[m]\mathcal{H}\left\{\delta[n-m]\right\} = x[m]h[n-m] \tag{4.3}$$

Our input signal comprises of several (possibly an infinite number) signals with the same form as  $x_m[n]$ . Additivity property tells us that we can find the output of the linear system to each one of them and then add the results together to find the output of the system for x[n]. That is,

$$\mathcal{H}\{x[n]\} = \mathcal{H}\left\{\sum_{m=-\infty}^{\infty} x[m]\delta[n-m]\right\}$$

![](_page_3_Picture_2.jpeg)

Figure 4.1: Convolution of x[n] with h[n] is identical to the convolution of h[n] with x[n]. The signal inside the box representing the lti system is its unit impulse response.

$$= \sum_{m=-\infty}^{\infty} \mathcal{H} \left\{ x[m]\delta[n-m] \right\}$$

$$= \sum_{m=-\infty}^{\infty} x[m]h[n-m]$$
(4.4)

The expression on the right-hand-side is known as the convolution sum of the signals x[n] and h[n]. The above result tells us that we can find the output of a discrete-time linear, time-invariant system for every input signal if we know the unit impulse response signal of the system. That is, the *unit impulse response signal* of a discrete-time, lti system completely describes the characteristics of the system. This is a property that holds also in the continuous-time case. However, this is true only for linear and time-invariant systems. If the system is nonlinear and/or time-variant, you cannot find its input through a convolution of its input signal and its unit impulse response signal.

#### 4.2 Another form for the convolution sum

let the output of the lti system be y[n]. Then, from (4.4) we have that

$$y[n] = \sum_{m = -\infty}^{\infty} x[m]h[n - m]$$

$$(4.5)$$

In the above equation, apply a change of variable k = n - m so that m = n - k. The range of k, as m varies from  $-\infty$  to  $\infty$  is  $\infty$  to  $-\infty$ . Since the order of summation does not change the total sum, we can use the order of summation from  $-\infty$  to  $\infty$  to get another form for the convolution sum shown below:

$$y[n] = \sum_{k=-\infty}^{\infty} x[n-k]h[k]$$
(4.6)

This means that if we interchange the role of the input signal and the unit impulse response signal in the convolution operation, the results do not change. This concept is shown pictorially in Figure 4.1. In the evaluation of convolution sum, we can use either of the expressions for the operation depending on convenience (perhaps from the perspective of ease of computation), and you should get the same results.

## 4.3 How to compute the convolution sum

We will demonstrate the steps of discrete-time convolution through an example.

#### Exercise 4.1

et the unit impulse response signal of a liner, time-invariant system be

$$h[n] = \begin{cases} 3 & ; & n = 0 \\ 2 & ; & n = 1 \\ 1 & ; & n = 2 \\ 0 & ; & \text{otherwise} \end{cases}$$

We will work with (4.6), the second representation of convolution sum. Substituting h[n] into (4.6) gives

$$y[n] = 3x[n] + 2x[n-1] + x[n-2]$$

This means that at any time n, the system is using the most recent three samples to compute the output at time n. Furthermore, the output of the system at time n is a weighted average (sum) of these three samples. The weighting function is found by reflecting the impulse response signal about the y axis. At each time, the output is found by sliding the "window function" so that the coefficient or weight at time 0 is aligned with the input sample at time n. The samples of the weight and the input signal that overlap in this window are multiplied together, and the products added to get the output sample at time n. Figure 4.2 illustrates these steps.

Some terminology: In the above example, the impulse response signal had only a finite number of non-zero samples. Such systems are known as *finite impulse response* (FIR) filters. There are many linear, time-invariant systems for which the unit impulse response signals last for ever. In such situations, we day that the impulse response signals have infinite duration, and such systems are called *infinite impulse response* (IIR) systems. We will study both FIR and IIR systems.

#### 4.3.1 Examples in convolution

We will use \* to denote the convolution operation. That is

$$x[n] * h[n] = \sum_{m = -\infty}^{\infty} x[m]h[n - m] = \sum_{m = -\infty}^{\infty} h[m]x[n - m]$$
(4.7)

#### Exercise 4.2

Evaluate x[n] \* h[n] when

$$x[n] = u[n]$$

and

$$h[n] = u[n]$$

Answer:

This problem is easiest to visualize graphically. We consider the cases of computing the convolution sum for n < 0 and  $n \ge 0$  separately.

x[m] = u[m] and h[n-m] = u[n-m] are plotted together in Figure 4.3a for a case when n is negative. We can see that x[m] and the reflected and shifted signal h[n-m] do not overlap in this case. Therefore the convolution sum is zero for n < 0.

$$n \ge 0$$

![](_page_5_Figure_2.jpeg)

Figure 4.2: The output of a discrete-time linear, time-variant function is always a weighted sum of the input samples. This figure illustrates how the weighting function is found to find the output at time n = 5. The weighting function is obtained by reflecting the unit impulse response signal about the y axis, and shifting it such that h[0] aligns with the input signal sample x[5]. For an arbitrary time n at which we wish to find the output, the reflected impulse response signal should be shifted so that h[0] aligns with the x[n]. (This is the same process as shifting h[-m] to the right by n samples. If n is negative, shifting to the right is actually shifting to the left.) Then we multiply all overlapping samples of the two signals and add them together to find the output signal at time n = 5 in this example. This operation must be repeated for all times at which we need to find the output.

![](_page_6_Figure_2.jpeg)

Figure 4.3: Graphical representation of the convolution in Exercise 4.3.

Figure 4.3b shows a case when n is greater than or equal to zero. In this case, the overlap of the two signals extends from m=0 to m=n and involve exactly n+1 samples. Since in this range of m, the product is 1, the sum of the product of the two terms over m is equal to n+1 for  $n \ge 0$ . That is,

$$y[n] = \begin{cases} 0 & ; & n < 0 \\ n+1 & ; & n \ge 0 \end{cases}$$

The ramp function that is the output of the convolution in this exercise is shown in Figure 4.3c.

#### Exercise 4.3

Evaluate the convolution of

$$x[n] = \left\{ \begin{array}{ll} n+1 & ; & 0 \leq n \leq 5 \\ 0 & ; & \text{otherwise} \end{array} \right.$$

and

$$h[n] = \begin{cases} 1 & ; & 0 \le n \le 9 \\ 0 & ; & \text{otherwise} \end{cases}$$

Answer:

As was the case in Exercise 4.3, the convolution output y[n] = 0 for n < 0. For  $n \ge 0$ , we have four different additional cases to consider as shown in Figure 4.4. We will take each case separately.

#### Partial overlap of n in the range $0 \le n < 5$

For this case, the overlap of x[m] and h[n-m] is in the range  $0 \le m < 5$ . Recognizing that x[m] = m+1 when x[m] is non-zero and that h[m] = 1 when it is non-zero results in

$$y[n] = \sum_{m=0}^{n} x[m]h[n-m] = \sum_{m=0}^{n} (m+1) = \frac{(n+1)(n+2)}{2}$$

![](_page_7_Figure_2.jpeg)

Figure 4.4: Graphical demonstration of how x[m] and h[n-m] overlaps in different ranges of n.

Complete overlap for n in the range  $5 \le n < 10$ For this case, the overlap of x[m] and h[n-m] is in the range  $0 \le m \le 5$ . The convolution sum is

$$y[n] = \sum_{m=0}^{5} x[m]h[n-m] = \sum_{m=0}^{5} (m+1) = 21$$

Partial overlap for n in the range  $10 \le n < 15$ For this case, the overlap of x[m] and h[n-m] is in the range  $n-9 \le m \le 5$ . The convolution sum is

$$y[n] = \sum_{m=n-9}^{5} x[m]h[n-m] = \sum_{m=n-9}^{5} (m+1) = \frac{(15-n)(n-2)}{2}$$

For  $n \geq 15$ , the two signals do not overlap and therefore the convolution sum is zero. Combining all the above results provides the complete solution:

$$y[n] = \begin{cases} 0 & ; & n < 0 \text{ or } n \ge 15\\ \frac{(n+1)(n+2)}{2} & ; & 0 \le n < 5\\ 21 & ; & 5 \le n < 10\\ \frac{(15-n)(n-2)}{2} & ; & 10 \le n < 14 \end{cases}$$

Figure 4.5 displays the output for this problem.

Something to think about: Suppose that we wish to convolve a discrete-time signal x[n] with nonzero samples in the interval  $[L_x, M_x]$  with h[n] whose non-zero samples fall in the interval  $[L_h, M_h]$ .

![](_page_8_Figure_2.jpeg)

Figure 4.5: The convolution sum of Exercise 4.4.

You should show (the easiest method would be to use a graphical approach) that the following is indeed true:

The non-zero values of the convolution sum of x[n] and h[n] extends from  $n = L_x + L_h$  to  $n = M_x + M_h$ .

#### Exercise 4.4

Evaluate the convolution of

$$x[n] = 2^{-n}u[n]$$

and

$$h[n] = 3^{-n}u[n]$$

Answer:

Since both x[n] and h[n] are zero for negative values of n, we know from previous discussion that the convolution output is zero for n < 0. For  $\geq 0$ , we proceed as follows:

$$y[n] = \sum_{m=-\infty}^{\infty} x[m]h[n-m]$$
$$= \sum_{m=-\infty}^{\infty} 2^{-m}u[m]3^{-(n-m)}u[n-m]$$

Since u[m] is zero for negative values of m, terms in the above summation are zero for m < 0. Therefore, the lower limit of the summation can be replaced with zero. In a similar manner, u[n-m]=0 whenever n-m<0 or equivalently, n < m. This implies that the upper limit in the summation can be replaced with n. Figure 4.6 demonstrates the range of the summation graphically. Applying these observations to the last expression for the convolution sum, we get (for  $n \ge 0$ )

$$y[n] = \sum_{m=0}^{n} 2^{-m} 3^{-(n-m)}$$

Since the summation accounts for the effects of the two step functions, we do not need to explicitly

![](_page_9_Figure_2.jpeg)

Figure 4.6: Graphical demonstration of the limits of the summation in the definition of convolution sum in Exercise 4.5. x[m] and h[n-m] overlaps only in the range  $0 \le m \le n$ .

![](_page_9_Figure_4.jpeg)

Figure 4.7: The convolution sum of Exercise 4.5.

include them in the expressions. Manipulation the above further gives

$$y[n] = 3^{-n} \sum_{m=0}^{n} 2^{-m} 3^{m}$$
$$= 3^{-n} \sum_{m=0}^{n} \left(\frac{3}{2}\right)^{m}$$

Applying the closed form expression for the sum of a geometric series, we get the final expression for the convolution of the two signals to be

$$y[n] = \begin{cases} 0 & ; n < 0 \\ 3^{-n} \left\{ \frac{1 - (3/2)^{n+1}}{1 - (3/2)} \right\} = \left\{ 3(2^{-n}) - 2(3^{-n}) \right\} & ; n \ge 0 \end{cases}$$

This is a signal of infinite duration, but decays to very small values as n becomes large. The dominant part of this signal is depicted in Figure 4.7.

Something to think about: Will we get the same result if you reflected x[m] instead of h[m] in the above exercise? Evaluate the convolution sum using this approach to show that the two results are identical.

### 4.4 Causal systems

A system is said to be *causal* if its output depends only on current and past values of the input signal. That is, it does not have to anticipate future values of the input signal to find the output at the present time. The following result is an important one:

A linear, time-invariant system is causal if and only if its unit impulse response signal is zero for times less than zero.

Showing that this statement is correct is easy. Suppose that the unit impulse function is not zero at some negative value of time, say -r where r is a positive integer. When h[n] is reflected about the y axis and shifted by n samples so that h[0] overlaps x[n], the impulse response sample at -r overlaps the input signal at time n+r. The case for r=1 is illustrated in Figure 4.8. This implies that anytime h[n] is non-zero at negative values of time, the output of the system at time n requires knowing of the input signal at times later than n, meaning that the system is non-causal.

In a similar manner we can also argue that if h[n] = 0 for n < 0, the output y[n] of a linear, time-invariant system at time n depends only on input samples at time n and before.

#### 4.4.1 Convolution sum for causal systems

By simply recognizing the range of non-zero values of the impulse response signal and adjusting the range of summations accordingly, we can show that

$$x[n] * h[n] = \sum_{m=-\infty}^{n} x[m]h[n-m] = \sum_{m=0}^{\infty} h[m]x[n-m]$$
 (4.8)

#### 4.5 Realization of FIR filters

Consider the FIR filter in Exercise 4.2. Since in this example, the impulse response signal has only three non-zero samples, the computation of the output at any time requires only three multiplications and two additions. It is also necessary to remember the most recent three samples to compute the output since these three samples are multiplied by the samples of the impulse response signal and then the products added together to compute the output sample. We may use a series of shift registers as shown in Figure 4.9 to store the most recent three samples in memory, and to update the stored values as a new sample arrives at each sampling instant. The series interconnection of shift registers are commonly referred to as a delay line. Because the shift register simply delays its input by a single sample, in our block diagrams we will represent this building block as a delay element and denote the delay using the notation  $z^{-1}$ . We will find out why this notation is appropriate when we learn about z transform.

To implement an FIR filter with N coefficients of a discrete-time, causal, FIR filter whose impulse response signal is non-zero in the range  $0 \le n \le N-1$ , we simply create a delay line with N-1 delay elements, and then tap each of the N samples (one sample at the input of the first delay element and the other N-1 samples at the outputs of the N-1 delay elements), and input the signal at each tap to a multiplier containing the coefficient values, and then add the products together as shown in Figure 4.10 for a three-coefficient filter.

![](_page_11_Figure_2.jpeg)

Figure 4.8: If the impulse response is non-zero for even one negative value of time, the system is non-causal. Note that the output at time n = 5 depends on the input signal at time n = 6.

![](_page_12_Picture_2.jpeg)

Figure 4.9: Using shift registers to store and update input samples needed to compute the output at each time. As a new sample arrives at time n, the system pushes the previous sample at the input of each shift register to its output. In block diagrams of FIR filter realizations, we will not explicitly refer to shift registers, but use a delay element block shown in (b) to represent it.

![](_page_12_Picture_4.jpeg)

Figure 4.10: Tap delay line realization of a three-coefficient FIR filter.

### 4.6 Bounded-Input, Bounded-Output Stable Systems

A bounded-output, bounded-input (BIBO) stable system is one whose output signals are bounded for all possible bounded input signals. A signal x[n] is said to be bounded if there exists a finite and positive number  $M_x$  such that  $|x[m]| \le M_x < \infty$  for  $-\infty < t < \infty$ . If a system is BIBO stable, for every input signal bounded by some  $0 < M_x < \infty$ , there will be an  $M_y$ ,  $0 < M_y < \infty$  such that the output signal of the system is bounded by  $M_y$ . This definition applies to all systems; not just linear, time-invariant systems.

For linear, time-invariant systems, there is a simple test based on unit impulse response functions that we can use to determine if the system is BIBO stable or not. The test is based on the following result:

A discrete-time, linear, time-invariant system is BIBO stable if and only if its unit impulse response signal is absolutely summable, i. e.,

$$\sum_{n=-\infty}^{\infty} |h[n]| < \infty \tag{4.9}$$

We need to prove both the if condition and the only if condition.

Proof - Part 1: A linear, time-invariant system with unit impulse response function h[n] is BIBO stable if h[n] is absolutely summable.

Because we know that h[n] is absolutely integrable, we know that there exists a positive and finite  $M_h$  such that

$$\sum_{m=-\infty}^{\infty} |h[m]| = M_h < \infty \tag{4.10}$$

Here we used the dummy variable m for clarity of presentation of the rest of the proof. Now, assume that the input signal x[n] is bounded by  $M_x$ . That is,

$$|x[n]| \le M_x < \infty \quad ; \quad -\infty < m < \infty \tag{4.11}$$

The output of the system is given by

$$y[n] = \sum_{m=-\infty}^{\infty} h[m]x[n-m]$$

$$(4.12)$$

Now, take the absolute value of both sides:

$$|y[n]| = \left| \sum_{m=-\infty}^{\infty} h[m]x[n-m] \right| \tag{4.13}$$

$$\leq \sum_{m=-\infty}^{\infty} |h[m]x[n-m]| \tag{4.14}$$

$$= \sum_{m=-\infty}^{\infty} |h[m]| |x[n-m]| \le \sum_{-\infty}^{\infty} |h[m]| M_x$$
 (4.15)

$$= M_x \sum_{m=-\infty}^{\infty} |h[m]| = M_x M_h < \infty \tag{4.16}$$

the transition from the first line of the above equation to the second line is possible because the absolute value of the sum of two or more values is always smaller than or equal to the sum of their absolute values (i. e.,  $|a+b| \le |a| + |b|$ ). The inequality in line three comes because we replaced |x[n-m]| with  $M_x$  which is always larger than or equal to |x[n-m]|. Finally, in line four, we replaced the sum of |h[m]| with its value  $M_h$ . Since both  $N_x$  and  $M_h$  are finite numbers, their product is also finite, indicating that the output signal y[n] is bounded, provided that the input signal x(t) is bounded, and the unit impulse response signal is absolutely summable. This proved the first part of the stability criterion. We now know that a linear, time-invariant system with an absolutely summable unit impulse response signal is BIBO stable.

The next step is to prove that the absolute summability of the unit impulse response is a necessary condition, i. e., if h[n] is not absolutely summable, the system is not BIBO stable.

Proof - Part 2: A linear, time-invariant system with unit impulse response function h[n] is BIBO stable only if h[n] is absolutely summable.

We will prove by assuming that h[n] is not absolutely summable, i. e.,

$$\sum_{n=-\infty}^{\infty} |h[n]| = \infty \tag{4.17}$$

there exists at least one bounded signal that will drive the output to infinity at some time. We proceed in the following manner:

We start by writing an expression for the output y[n] at time n = 0.

$$y[0] = \sum_{m = -\infty}^{\infty} h[m]x[0 - m] = \sum_{m = -\infty}^{\infty} h[m]x[-m]$$
(4.18)

We will now find an x[n] that makes the output at time zero infinity. Using the variable m because the above equation uses this variable, we define

$$x[-m] = \begin{cases} 1 & ; & h[m] > 0 \\ 0 & ; & h[m] = 0 \\ -1 & ; & h[m] < 0 \end{cases}$$
 (4.19)

We note first that x[m] is a bounded signal because its magnitude is never more than 1. If the system output at time zero is infinity for this input signal, the system is definitely not BIBO stable. To proceed to show this, we note also that x[-m] defined above is nothing but the sign of h[m]. This means that

$$h[m]x[-m] = h[m]\operatorname{sign}\{h[m]\} = |h[m]| \tag{4.20}$$

Substituting this result in the equation for y[0], we get,

$$y[0] = \sum_{m = -\infty}^{\infty} h[m]x[-m] = \sum_{m = -\infty}^{\infty} |h[m]| = \infty$$
 (4.21)

This proves the only if part of our stability result. The absolute summability of the unit impulse response signal is not just a sufficient condition; it is also a necessary condition.

#### Exercise 4.5

Determine if the discrete-time, linear time-invariant systems with the following unit impulse response signals stable in the BIBO sense:

- 1.  $h[n] = 0.8^n u[n]$
- 2.  $h[n] = \sin(0.2\pi n)u[n]$
- 3. h[n] = u[n]
- 4.  $h[n] = \frac{1}{n}u[n-1]$

Answer:

- 1. The absolute sum of the unit impulse response signal  $\sum_{n=-\infty}^{\infty} |h[n]| = \sum_{n=0}^{\infty} 0.8^n = \frac{1}{1-0.8}$ , which implies that the unit impulse response signal is absolutely summable, and therefore the system is BIBO stable.
- 2. Since there are an infinite number of periods for this sinusoidal signal, and the sum of the absolute value of the signal in each period is a non-zero, positive number, the total sum will be infinity, implying that the system is not BIBO stable.
- 3. The absolute values of the sample values of the unit step signal u[n] add up to  $\infty$ . Therefore, the system is not BIBI stable.
- 4. We can show that the sum of the absolute value of the samples of this unit impulse response signal is  $\infty$ , indicating that this system is also not BIBO stable. To show this, we divide the time instances from 1 till  $\infty$  into smaller intervals as follows:  $\{n=1\}$ ,  $\{n=2,3\}$ ,  $\{n=4,5,6,7\}$ ,  $\{n=8,9,\cdots,15\}$  and so on. We can now show that the sum of the absolute values of the impulse response samples in each set is more than 0.5. For example, consider the third set. each of h[4] h[7] is greater than 1/8 implying their absolute sum is more that 4(1/8) = 1/2. Similarly, all the samples of the fourth set is larger than 1/16, indicating that they also add up to a value larger than 1/2. There are an infinite number of such subsets of samples that add up to more than 0.5, indicating that the absolute sum of all the samples is  $\infty$ .

#### 4.7 Problems

1. The unit impulse response of a discrete-time, linear, time-invariant system is given by

$$h[n] = \begin{cases} 1 & ; & n = 0 \\ 2 & ; & n = 1 \\ -1 & ; & n = 2 \\ 0 & ; & \text{Otherwise.} \end{cases}$$

Show from basic principles (definitions of linearity and time-invariance and the decomposition of input signals using (discrete-time) impulse functions) that the output of this system to any input signal x[n] is given by

$$y[n] = x[n] + 2x[n-1] - x[n-2]$$

4.7. PROBLEMS

2. The convolution sum of signals x[n] and h[n] is defined as

$$x[n] * h[n] = \sum_{m = -\infty}^{\infty} x[m]h[n - m] = \sum_{m = -\infty}^{\infty} h[m]x[n - m] = h[n] * x[n]$$

Note that the symbol \* represents convolution and not multiplication.

(a) Suppose that h[n] = 0 for n < 0. Show, for this case that

$$x[n] * h[n] = \sum_{m=-\infty}^{n} x[m]h[n-m] = \sum_{m=0}^{\infty} h[m]x[n-m]$$

Suppose in addition that x[n] = 0 for n < 0. Show that

$$x[n] * h[n] = \sum_{m=0}^{n} x[m]h[n-m]$$

(b) Suppose now that you wish to convolve x[n] with h[n] and h[n] is non-zero only for n = 0, 1, 2, 3, 4. Show that one way to implement the convolution is as shown in the block diagram of Figure 4.11. In the figure, the notation  $z^{-1}$  represents a delay element.

![](_page_16_Picture_9.jpeg)

Figure 4.11: Block diagram to implement convolution of x[n] with a finite-length signal h[n].

The figure suggests an implementation based on delay elements, multipliers and adders. How will the block diagram change if h[n] has N non-zero samples for n in the range  $0 \le n \le N-1$ ?

- 3. Compute the following convolutions:
  - (a) u[n] \* u[n]
  - (b)  $\cos(0.25\pi n + 0.1\pi) * \delta(n 10)$
  - (c) x[n] \* h[n] where  $x[n] = 0.5^n u[n]$  and  $h[n] = 0.2^n u[n]$ .

(d) 
$$x[n] * h[n]$$
 where

$$x[n] = \begin{cases} 1 & ; \quad 5 \le n \le 10 \\ 0 & ; \quad \text{otherwise} \end{cases}$$

and

$$h[n] = \begin{cases} 1 & ; & 2 \le n \le 12 \\ 0 & ; & \text{otherwise} \end{cases}$$

4. Consider a discrete-time, linear, time-invariant system whose response to a unit step signal u[n] is given by

$$p[n] = 0.6^n u[n]$$

Find the output of the system when its input is

$$x[n] = \begin{cases} 1 & ; & n = 0 \\ 2 & ; & n = 1 \\ 1 & ; & n = 2 \\ 0 & ; & \text{otherwise} \end{cases}$$

*Hint*: First find the unit impulse response of the signal. Using the fact that  $\delta[n] = u[n] - u[n-1]$  may be useful.

5. Find the output of an lti system with unit impulse response signal

$$h[n] = (-2)^n u[n-1]$$

when its input is

$$x[n] = e^{-n}u[n+1]$$

6. The input-output relationship of a discrete-time lti system is given by

$$y[n] = x[n] - 2x[n-1]$$

We wish to find the output of this system to a unit step function. Solve this problem using the following methods:

- (a) Direct substitution of x[n] = u[n] in the input-output relationship.
- (b) Finding the unit impulse response signal of the system and then convolving the result with u[n].
- (c) Show that

$$u[n] = \sum_{m = -\infty}^{n} \delta[m]$$

and then argue that

$$y[n] = \sum_{m=-\infty}^{n} h[m]$$

when the input signal is the unit step function.

7. This problem investigates an interesting application of discrete-time convolution: The expansion of certain polynomial expressions.

4.7. PROBLEMS

(a) By hand, expand the polynomial  $(z^3 + z^2 + z + 1)^2$ . Compare the coefficients to the convolution of the signal with values (starting from time 0) [1 1 1 1] with itself.

- (b) Formulate a relationship between the coefficients of the product of two polynomial expressions (with constant coefficients) and the convolution of its coefficient sequences.
- (c) Write a Matlab script to find the coefficients of the product of two constant coefficient polynomials and use the script to find the expansions for the following polynomials:

i. 
$$(z^5 + 3z^4 + 3z^2 + 5)(2z^3 + 1)$$
  
ii.  $(z^{-4} - 2z^{-3} + 3z^{-2} + 1)^4$ 

- 8. Let y[n] be the discrete time signal obtained by convolving x[n] with h[n]. Starting with the basic definition of convolution, show that the convolution of x[n] with h[n-k] is given by y[n-k], where k is a constant integer.
- 9. A discrete-time, linear time-invariant system has inpulse response given by  $h[n] = 0.5^{|n|}$ .
  - (a) Is this system causal? Justify your answer.
  - (b) Compute  $\sum_{n=-\infty}^{\infty} |h[n]|$ . Is this system stable in the bounded-input, bounded-output sense?
  - (c) Find the output of this system when its input is u[n-3].
- 10. Two discrete-time, linear, time-invariant systems connected in series (cascade) have unit impulse responses given by  $h_1[n] = 0.9^n u[n] 0.5(0.9)^{n-1} u[n-1]$  and  $h_2[n] = 0.5^n u[n] 0.9(0.5)^{n-1} u[n-1]$ , respectively. Show that the series (cascade) connection of the system produces an identity system, *i.e.*, the unit impulse response function of the cascade is a unit impulse function, or equivalently, the output y[n] of the cascade is the same as the input signal x[n].
- 11. The crosscorrelation of two signals x[n] and y[n] is defined as

$$r_{xy}[m] = \sum_{n = -\infty}^{\infty} x[n]y[n - m]$$

Notice that  $r_{xy}[m]$  is defined in a similar manner, but not quite the same as the convolution sum of the two signals. The independent variable m corresponds to the relative shift between the two signals.

- (a) Show that  $r_{xy}[m]$  is the convolution output at time m of the signals x[n] and y[-n]. Is  $r_{xy}[m] = r_{yx}[m]$ ?
- (b) Crosscorrelation is said to indicate the similarity between two signals. Do you agree? Why or why not?
- (c) Study how the conv command in Matlab may be used to find the convolution output of two finite length signals. Explain how you would use this command to compute the crosscorrelation of two finite length signals.
- 12. In this problem, we learn about linear interpolation of discrete-time signals. Suppose we have a signal x[n] and we wish to increase the sampling rate by a factor of N. We start by creating

a new signal y[n] that is N times larger in length by inserting N-1 samples between adjacent samples. Mathematically, we can define y[n] as

$$y[n] = \begin{cases} x[n/N] & ; & n \text{ is an integer multiple of } N \\ 0 & ; & \text{otherwise} \end{cases}$$

(a) Show that the linear, time-invariant filter with unit impulse response signal

$$h[n] = \sum_{k=-(N-1)}^{N-1} \left(1 - \left|\frac{k}{N}\right|\right) \delta[n-k]$$

operating on y[n] will replace the inserted zeros in y[n] straight line approximation for the "missing" signals. Sketch h[n] for N=4.

- (b) The interpolating filter is non-causal. What is the smallest delay (shift) we need to introduce in h[n] for the system to become causal? What is the effect of this shift on the behavior of the filter?
- 13. Let the output of a discrete-time, linear time-invariant system to a delayed step input u[n-4] be

$$w[n] = \begin{cases} 1 & ; \quad n = 5, 6, 7, 8 \\ 0 & ; \quad \text{otherwise} \end{cases}$$

- (a) Find the unit impulse response of the system.
- (b) Find the output of the system when its input is

$$x[n] = 0.2^n u[n]$$

Hint: How is  $\delta[n]$  related to u[n-4]? Find this relationship and then use the linearity and time invariance of the system to find the unit impulse response signal.