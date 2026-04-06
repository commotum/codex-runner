# Contents

| <b>2</b> | Bas | ic signals and signal operations                             |
|----------|-----|--------------------------------------------------------------|
|          | 2.1 | Amplification/attenuation of signals                         |
|          | 2.2 | Delay/advance of signals                                     |
|          | 2.3 | Reflection about the $y$ axis                                |
|          | 2.4 | Subsampling of signals                                       |
|          | 2.5 | Combinations of transformations                              |
|          | 2.6 | Unit impulse signal                                          |
|          |     | 2.6.1 The unit step signal                                   |
|          |     | 2.6.2 Importance of impulse functions in signals and systems |
|          | 2.7 | Everlasting Complex Sine Waves                               |
|          |     | 2.7.1 A quick refresher on complex numbers                   |
|          |     | 2.7.2 Everlasting complex exponential                        |
|          | 2.8 | Problems                                                     |

2 CONTENTS

## Chapter 2

# Basic signals and signal operations

Keep in mind that we look at signals as nothing but mathematical functions, and the concepts we discuss here are simply transformations of mathematical functions. Of course, each of these transformations defines a system. We will first look at transformations of continuous-time signals. Let us consider an arbitrary signal x[n].

## 2.1 Amplification/attenuation of signals

Amplification or attenuation of a signal x[n] can be accomplished by multiplication of the signal by a constant. When the multiplier is larger than one, amplification results. The signal is attenuated when the multiplier is less than one. The operation is mathematically written as

$$y[n] = \alpha x[n] \tag{2.1}$$

## 2.2 Delay/advance of signals

Consider the signal transformation of the form

$$y[n] = x[n-m] (2.2)$$

for an integer value of m. What this means is that the value of y at time n is the same as the value of x at time n-m. If m is positive, y will come m samples after x. In other words, a positive value of m delays the input signal (shifts the signal x to the right). If m is negative, say -2, then y[n] = x[n+2], which implies that y at time n is the same as x at time n+2. That is y occurs 2 samples earlier than x, implying that this system advances the input signal (shifts the signal x to the left). The delay and advance operations are illustrated in Figures 2.1 and 2.2, respectively.

Something to remember: There is no concept of fractional time shift for discrete-time signals. This is because, if we try to delay by a fraction of an integer, the signal is not defined at the resulting points.

Something to think about: Suppose that  $x[n] = \cos(\frac{\pi}{6}n + \frac{\pi}{4})$ . We wish to delay this signal by m samples. (Here m is a fixed number representing the number of samples the signal is shifted by.)

![](_page_3_Figure_2.jpeg)

Figure 2.1: The transformation y[n] = x[n-2] delays the input signal by 2 samples at the output. Note that each sample in x[n] appears two samples later in y[n]. Left: x[n]; Right: x[n-2].

![](_page_3_Figure_4.jpeg)

Figure 2.2: The transformation y[t] = x[n+2] advances the input signal by 2 samples at the output. Note that each sample in x[n] appears two samples earlier in y[n]. Left: x[n]; Right: x[n+2].

![](_page_4_Figure_2.jpeg)

![](_page_4_Figure_3.jpeg)

Figure 2.3: The transformation y[n] = x[-n] reflects the input about the y axis. (a) A signal x[n]. (b) The signal x[-n].

Which of the following represent the correct description of the resulting signal? (a)  $\cos(\frac{\pi}{6}n + \frac{\pi}{4} - m)$ . (b)  $\cos(\frac{\pi}{6}(n-m) + \frac{\pi}{4})$ .

Hint: After the transformation, the left hand side of the original equation becomes x[n-2], meaning we are changing the variable n to n-2 to implement the transformation. You may also want to sketch the two signals above using Matlab to help you validate your answer.

## 2.3 Reflection about the y axis

Consider the transformation (system) whose input-output relationship is given by

$$y[n] = x[-n] \tag{2.3}$$

For this transformation, the output at the nth sample is the input at the -nth sample, indicating that the output is a mirror image of the input if we place the mirror along the y axis. An example illustrating this concept is shown in Figure 2.3.

Something to think about: Suppose you reflect x[n] about the y axis and then delay the signal by 2 samples. The resulting signal is given by x[-n+2]. Do you agree or disagree with this statement?

## 2.4 Subsampling of signals

Suppose you have a transformation of the form y[n] = x[2n]. This means that the output of this transformation at time n is the input at time 2n, implying that the output is compressed along the x axis by a factor of 2 and also that every other sample of x[n] is discarded. This process is called subsampling. In general, y[n] = x[pn], and p is an integer represents subsampling by a factor of p and only one sample out of p consecutive samples of p appear in p. If p is negative, the signal is subject to reflection as well as subsampling. An example of subsampling of a signal is shown in Figure 2.4

![](_page_5_Figure_2.jpeg)

![](_page_5_Figure_3.jpeg)

Figure 2.4: The transformation y[n] = x[2n] retains every other input sample and discards the rest. The output signal may be thought of as a compressed version of x by a factor of 2 along the time axis. Left: x[n]; Right: x[2n].

Something to think about: When a signal is subsampled using the transformation y[n] = x(pn), y[0] = x[0] regardless of the value of p or the nature of the input signal x[n].

### 2.5 Combinations of transformations

Suppose you want to build a system that delays the input signal by m seconds, subsample the result by a factor p and reflect the result of the subsampling operation about the y axis, what would be the mathematical representation of the combination of these three transformations? This is not difficult to do if we do the operations sequentially in the order they happen, recognizing the fact that each transformation is done to the independent variable n. In the above problem, the first operation is that of delaying the signal by m second. This results in the intermediate output

$$y_1[n] = x[n-m] \tag{2.4}$$

We then subsample  $y_1[n]$  by a factor of p along the time axis, giving us the second intermediate result

$$y_2[n] = y_1[pn] = x[pn - m] (2.5)$$

Finally, we reflect  $y_2[n]$  about the y axis to get the output of the combination of the three transformations. The result is

$$y[n] = y_2[=n] = x[-pn - m]$$
(2.6)

Similar discussions may be made for any combination of the three basic operations described above.

**Example:** Let  $x[n] = 0.9^{-n^2+3}$ . Suppose we wish to reflect it about the y axis, advance the results by 5 samples and then subsample the resulting signal along the time axis by a factor of 2. What is the end result of these transformations?

Solution: The sequence of operations are:

(1) 
$$y_1[n] = x[-n]$$

(2) 
$$y_2[n] = y_1[n+5] = x[-(n+5)] = x[-n-5]$$

(3) 
$$y[n] = y_2[2n] = x(-(2n) - 5)$$

This means that we can get an expression for the output signal y[n] substituting n with -2n-5 in the expression for x[n], which gives

$$y(t) = x(-2n - 5) = 0.9^{-(-2n - 5)^2 + 3}$$

Something to think about: Consider the transformation y[n] = x[-3n+6] that was obtained through the following sequence of transformations: (a) a reflection about the y axis, (b) a s subsampling along the time axis, and (c) a delay or advance. What are the three operations that would result in the final result? Now assume that the sequence of operations were (a) a delay or advance, (b) a scaling along the time axis, and (c) a reflection about the y axis. What would the three operations have been in this case? You should see clearly that it is possible to get the same final result in many different ways.

#### Exercise 2.1

Here, we will use the variable m as the independent variable and consider the following operations on two signals h[m] and x[m] to obtain the output of a system at time n.

- 1. Reflect h[m] about the y axis.
- 2. Shift (delay) the reflected version of h[m] by n samples. If n is positive, this operation is a true delay. The signal is advanced if n is negative.
- 3. Multiply x[m] with the result of the previous step sample by sample.
- 4. Find the output y[n] of the system at time n by adding the sample values of the product signal in the previous step over all values of m from  $-\infty$  to  $\infty$ .

Derive an expression for the output y[n] of the system at time n in terms of h[m] and x[m]. Note that these set of operations have to be done separately for each value of n.

#### Solution:

- 1. Reflection of h[m] gives h[-m].
- 2. Shifting h[-m] by n samples corresponds to changing m to m-n. This gives

$$h[-(m-n)] = h[n-m]$$

- 3. Sample-by-sample multiplication gives the product value at time m as x[m]h[n-m]
- 4. Adding every sample of this product signal gives the value of the output at time n. That is

$$y[n] = \sum_{m = -\infty}^{\infty} h[n - m]x[m]$$

This, by the way, is the famous *convolution sum* that you will learn about in more detail later.

![](_page_7_Figure_2.jpeg)

Figure 2.5: A discrete-time unit impulse signal.

## 2.6 Unit impulse signal

The discrete-time unit impulse function is simply defined as

$$\delta[n] = \begin{cases} 1 & ; & n = 0 \\ 0 & ; & \text{otherwise} \end{cases}$$
 (2.7)

Figure 2.5 displays a unit impulse signal. Although most simple, this signal is of fundamental importance in discrete-time signal processing. For example, we will see soon that any arbitrary discrete time signal x[n] can be described as a sum of scaled (amplified or attenuated) and shifted versions of the unit impulse function. Before verifying this statement, let us discuss some properties of unit impulse functions.

1) The sum of the sample values of a unit impulse function is 1. This is also true for delayed unit impulse signals. That is,

$$\sum_{-\infty}^{\infty} \delta[n-m] = 1 \tag{2.8}$$

2) The sum of the sampled values of the sample-by-sample product of a delayed impulse signal  $\delta[n-m]$  is the value of x at time n=m. That is,

$$\sum_{-\infty}^{\infty} x[n]\delta[n-m] = x[m]$$
(2.9)

This result is known as the *sifting property* of the impulse signal.

3) The product of a delayed impulse (say,  $\delta[n-m]$ ) with any signal x[n] is an impulse at time n=m. The value of the sample at time n=m is x[m], i. e.,

$$x[n]\delta[n-m] = x[m]\delta[n-m] \tag{2.10}$$

![](_page_8_Figure_2.jpeg)

Figure 2.6: Signal decomposition using impulse functions. The signal in the top-left panel is the sum of the signals in the other panels.

Something to think about: The product of x[n] with a delayed impulse  $\delta[n-m]$  depends only on the value of x[n] at time n=m. If x[n] and y[n] are two different signals with identical values at time n=m, ie x[m]=y[m], the product of x[n] and y[n] with  $\delta[n-m]$  will be the same, i. e.,

$$x[n]\delta[n-m] = x[m]\delta[n-m] = y[m]\delta[n-m] = y[n]\delta[n-m]$$

4) Any signal x[n] can be written as a sum of scaled and delayed impulses as

$$x[n] = \cdots, x[-2]\delta[n+2] + x[-1]\delta[n+1] + x[0]\delta[n] +x[1]\delta[n-1] + x[2]\delta[n-2] + \cdots$$

$$= \sum_{m=-\infty}^{\infty} x[m]\delta[n-m]$$
(2.11)

This decomposition is illustrated in Figure 2.6.

The proofs of the above four properties of the discrete-time impulse function are left as exercises for the student.

#### 2.6.1 The unit step signal

The unit step signal u[n] is defined as

$$u[n] = \begin{cases} 1 & ; & n \ge 0 \\ 0 & ; & n < 0 \end{cases}$$
 (2.12)

#### Relationships between unit impulse signal and the unit step signal

1) Consider a differencing operation defined as

$$y[n] = x[n] - x[n-1]$$

![](_page_9_Figure_2.jpeg)

![](_page_9_Figure_3.jpeg)

![](_page_9_Figure_4.jpeg)

Figure 2.7: Differencing operation applied to the unit step signal will result in the unit impulse signal. Left: u[n]; Middle: u[n-1]; and Right: u[n] - u[n-1].

If we apply the differencing operation to the unit step signal, we will get the unit impulse signal, i. e.,

$$\delta[n] = u[n] - u[n-1] \tag{2.13}$$

To see this, observe the plots of u[n] and u[n-1] in Fig. 2.7. We can see that at all values of  $n \neq 0$ , u[n] = u[n-1]. This means that u[n] - u[n-1] when  $n \neq 0$ . When n = 0, u[n] = u[0] = 1 and u[n-1] = u[-1] = 0/ It immediately follows that

$$u[n] - u[n-1] = \begin{cases} 0 & ; n \neq 0 \\ 1 & ; n = 0 \end{cases}$$
$$= \delta[n]$$

2) Consider an accumulator operator defined as

$$y[n] = \sum_{m = -\infty}^{n} x[m]$$

The output of the accumulator at any time n is the sum of all samples that occurred all the way from the beginning  $(-\infty)$  till time instant n.

If we input a unit impulse signal to the accumulator system, we will get the unit step signal at the output. That is,

$$u[n] = \sum_{m = -\infty}^{n} \delta[m]$$

The proof is left as an exercise for the students.

#### Exercise 2.2

Evaluate the following sums:

- 1.  $\sum_{m=-\infty}^{n} \delta[m]$ . Note that your answer must be a function of n. You should derive the results for all values of  $-\infty < n < \infty$ .
- $2. \sum_{n=-\infty}^{\infty} \sin(3n^2 4)\delta[n-1]$

3. 
$$\sum_{n=-3}^{3} (\delta[n] + \delta[n-5])$$

$$4. \sum_{n=-\infty}^{\infty} \delta(2n)$$

$$5. \sum_{n=-\infty}^{\infty} \delta(2n-1)$$

Answers:

1. If n < 0, the impulse does not fall in the range  $[-\infty, t]$  and therefore the sum of the samples in the range  $-\infty < n < 0$  is zero because all samples of  $\delta[n]$  in this range have zero value. If  $n \ge 0$ , the impulse falls inside the range  $-\infty < m \le n$ , and therefore there is one sample corresponding to m = 0 when its value is 1. Further, all other samples have zero value. Consequently,

$$\sum_{m=-\infty}^{n} \delta(m) = \left\{ \begin{array}{ll} 0 & ; & n<0 \\ 1 & ; & n\geq 0 \end{array} \right. = u[n]$$

2. We can evaluate this sum using the sifting property of the unit impulse signal. Recall that the sifting property says that the sum of all the sample values in the product of a signal with a unit impulse function is the value of the signal at the location of the impulse. In this example, the impulse occurs at time n = 1, indicating that our solution must be the value of  $\sin(3n^2 - 4)$  at n = 1, which is equal to  $\sin(-1) = -\sin(1)$ . We will redo the problem from basic principles below.

$$\sum_{n=-\infty}^{\infty} \sin(3n^2 - 4)\delta[n - 1] = \sum_{n=-\infty}^{\infty} \sin(3(1)^2 - 4)\delta[n - 1]$$

$$= \sin(-1)\sum_{n=-\infty}^{\infty} \delta[n - 1]$$

$$= -\sin(1)$$

Note that the unit of angle is radians here. The first equality came about because the only value of the function  $\sin(3n^2-4)$  that matters in the evaluation of the integral is its value at n=1, the location of the impulse function. Once the value n=1 is used in this function, it just becomes a constant value and we can take it outside the integral.

3. We first separate the two terms in the sum as

$$\sum_{n=-3}^{3} (\delta[n] + \delta[n-5]) = \sum_{n=-3}^{3} \delta[n] + \sum_{n=3}^{3} \delta[n-5]$$

We note that the non-zero value of  $\delta[n-5]$  does not fall in the range of the integral [-3,3], and therefore the second term in the above equation is zero. In the first term on the right-hand-side, the non-zero value of the impulse function falls in the range of the summation, and therefore the sum is one. Combining these two facts, we get the result

$$\sum_{n=-3}^{3} (\delta[n] + \delta[n-5]) = 1$$

- 4. We start by trying to figure out what  $\delta[2n]$  looks like. At time sample n = 0,  $\delta[2n] = \delta[0] = 1$ . At all other times,  $\delta[2n] = 0$ , implying that  $\delta[2n] = \delta[n]$ . The sum of all sample values in this signal is one.
- 5. As before, we start by trying to figure out what  $\delta[2n-1]$  looks like. We can get  $\delta[2n-1]$  by first delaying  $\delta[n]$  by one sample (to get  $\delta[n-1]$ ) and then subsampling the resulting signal by a factor of two. This subsampling process retains the even samples of  $\delta[n-1]$  and discards all odd samples, including the sample at n=1 which has a value of 1. This means that  $\delta[2n-1]=0$  for all values of n, indicating that the sum of the sample values in this signal is zero.

#### 2.6.2 Importance of impulse functions in signals and systems

Most of our course will be devoted to the study of linear and time-invariant systems. We have not defined these concepts yet, but will do so shortly. It turns out that if you know the output of a linear and time-invariant system when its input is a unit impulse function, we can find the output of the system for all other inputs too. Thus, for linear and time-invariant systems, their unit impulse response signal (*i.e.*, the output of the system when its input is the unit impulse signal) completely characterizes the systems. Therefore, becoming comfortable with the impulse functions is important in our studies.

## 2.7 Everlasting Complex Sine Waves

#### 2.7.1 A quick refresher on complex numbers

A complex number x is generally written in the form

$$x = x_R + jx_I$$

where  $x_R$  is the real part of x,  $x_I$  its imaginary part, and j is the square root of -1. Looking at the plot in Fig. 2.8, we can see that the length of the vector representing the complex number is the length of the hypotenuse of a right-angled triangle with the real part of the complex number as the base and the imaginary part as the perpendicular. It immediately follows that the magnitude r of x is given by

$$r=\|x\|=\sqrt{x_R^2+x_I^2}$$

Here, ||x|| is simply a notation for the magnitude of x. Its phase  $\theta$  is given by

$$\theta = \angle x = \tan^{-1} \frac{x_I}{x_R}$$

where  $\angle x$  is a notation for the phase of the complex number x. Given r and  $\theta$ , x can be written in polar form as

$$x = re^{j\theta}$$

where

$$e^{j\theta} = \cos(\theta) + i\sin(\theta)$$

Also,  $x_R = r \cos(\theta)$  and  $x_I = r \sin(\theta)$ .

![](_page_12_Figure_2.jpeg)

Figure 2.8: Plotting a complex number on the complex plane. The coordinates are  $\text{Real}\{x\}$  and  $\text{Imag}\{x\}$ .

#### Addition and multiplication of two complex numbers

Let 
$$x_1 = x_{R1} + jx_{I1} = r_1e^{j\theta_1}$$
 and  $x_2 = x_{R2} + jx_{I2} = r_1e^{j\theta_1}$ . Then

$$x_1 + x_2 = (x_{R1} + x_{R2}) + j(x_{I1} + x_{I2})$$

Using the polar representation, this sum is given by

$$x_1 + x_2 = (r_1 \cos(\theta_1) + r_2 \cos(\theta_2)) + j(r_1 \sin(\theta_1) + r_2 \sin(\theta_2))$$

The product of  $x_1$  and  $x_2$  is given, using polar coordinates as

$$x_1 x_2 = r_1 e^{j\theta_1} r_2 e^{j\theta_2} = (r_1 r_2) e^{j(\theta_1 + \theta_2)}$$

Using the real parts and the imaginary parts, the product becomes, by direct multiplication of the terns involved,

$$x_1x_2 = (x_{R1} + jx_{I1})(x_{R2} + jx_{I2}) = (x_{R1}x_{R2} - x_{I1}x_{I2}) + j(x_{R1}x_{I2} + x_{I1}x_{R2})$$

### 2.7.2 Everlasting complex exponential

The everlasting complex exponential is defined as

$$e^{j\omega_0 n} = \cos(\omega_0 n) + j\sin(\omega_0 n) \tag{2.14}$$

where  $\omega_0$  is the frequency of the signal measured in radians/sample. More generally, an everlasting complex exponential with *amplitude* A and *initial phase*  $\varphi$  (which is really the phase value at time 0 although the exponential existed before that time too!) is

$$x[n] = Ae^{j(\omega_0 n + \varphi)} \tag{2.15}$$

At time n=0, this signal takes the value  $Ae^{j\varphi}$ . Every time the time sample increments by one, the total phase given by  $\theta(n)=\omega_0 n+\varphi$  increases by  $\omega_0$ , i. e.,

$$\theta(n) = \theta(n-1) + \omega_0$$

The magnitude of the complex exponential  $e^{j\omega_0 n}$  is always 1 since

$$\left\|e^{j\omega_0 n}\right\| = \sqrt{\cos^2(\omega_0 n) + \sin^2(\omega_0 n)} = 1$$

Then, we can visualize the everlasting complex exponential as a phase vector or phasor given by  $Ae^{j\varphi}$  rotating around the origin of the complex plane at the rate of  $\omega_0$  radians per sample. This rotation is counterclockwise if  $\omega_0$  is positive, and clockwise if  $\omega_0$  is negative.

For each value of n, the complex exponential will lie on a circle centered at the origin of the complex plane and with radius equal to ||A||. Assuming that A is a real valued number, we can also write the real valued functions  $A\cos(\omega_0 n + \varphi)$  and  $A\sin(\omega_0 n + \varphi)$  as

$$A\cos(\omega_0 n + \varphi) = \text{Real}\left\{Ae^{j\omega_0 n + \varphi}\right\}$$
(2.16)

and

$$A\sin(\omega_0 n + \varphi) = \operatorname{Imag}\left\{Ae^{j\omega_0 n + \varphi}\right\}$$
(2.17)

respectively. We also note that the real part and the imaginary part are the same, except for a phase shift of  $\pi/2$  radians.

#### Addition of two everlasting complex exponentials

We can show that the sum of two real-valued sinusoidal signals with the same frequency, but different amplitudes and initial phases, is still an everlasting complex exponential with appropriate amplitude and initial phase. That is,

$$A_1 \cos(\omega_0 n + \varphi_1) + A_2 \cos(\omega_0 n + \varphi_2) = A \cos(\omega_0 n + \varphi)$$
(2.18)

where A and  $\varphi$  need to be determined.

To see this, we write  $A_1 \cos(\omega_0 n + \varphi_1) + A_2 \cos(\omega_0 n + \varphi_2)$  as Real  $\left\{ A_1 e^{j(\omega_0 n + \varphi_1)} + A_2 e^{j(\omega_0 n + \varphi_2)} \right\}$ . Now,

$$\operatorname{Real}\left\{ A_{1}e^{j\omega_{0}n+\varphi_{1})} + A_{2}e^{j\omega_{0}n+\varphi_{2})} \right\} = e^{j\omega_{0}n} \left\{ A_{1}e^{j\varphi_{1}} + A_{2}e^{j\varphi_{2}} \right\}$$

The terms within the curly brackets may be thought off as the complex amplitude of the everlasting complex exponential, and can be written as

$$\left\{ A_1 e^{j\varphi_1} + A_2 e^{j\varphi_2} \right\} = A_1 \cos(\varphi_1) + A_2 \cos(\varphi_2) + j \left\{ A_1 \sin(\varphi_1) + A_2 \sin(\varphi_2) \right\}$$

We can now find the amplitude and initial phase in (2.18) as

$$A = \sqrt{(A_1 \cos(\varphi_1) + A_2 \cos(\varphi_2))^2 + (A_1 \sin(\varphi_1) + A_2 \sin(\varphi_2))^2}$$
 (2.19)

and

$$\varphi = \arctan\left\{\frac{A_1 \sin(\varphi_1) + A_2 \sin(\varphi_2)}{A_1 \cos(\varphi_1) + A_2 \cos(\varphi_2)}\right\}$$
(2.20)

Figure 2.9 shows the addition of the sinusoidal signals as summing up the corresponding phasor vectors.

![](_page_14_Figure_2.jpeg)

Figure 2.9: Summing two sinusoidal signals with the same frequency but different amplitudes and different initial phases from a phasors perspective. Adding  $A_1e^{j\varphi_1}$  with  $A_2e^{j\varphi_2}$  results in  $Ae^{j\varphi}$ . The real part of the sum phasor is the sun of the real parts of the two phasors added together. A similar relationship exists for the imaginary parts of the phasors also.

#### Frequency of discrete-time signals is periodic

Consider an everlasting complex sine wave of the form

$$x[n] = Ae^{j(\omega_0 n + \varphi)}$$

It is not difficult to see that if we change  $\omega_0$  to  $\omega_0 + 2\pi$  in the above expression, we get the same signal back. To see this,

$$Ae^{j((\omega_0+2\pi)n+\varphi)} = Ae^{j(\omega_0n+\varphi)}e^{j2\pi n} = Ae^{j(\omega_0n+\varphi)} = x[n]$$

The second equality arises because  $e^{j2\pi} = 1$ . This means that we only need to consider frequencies in one period of frequency. By convention, we will choose the period from  $-\pi$  to  $2\pi$ , *i. e.*, the fundamental period we consider is  $-\pi \le \omega \pi$ . Thus, the magnitude of the highest frequency in a discrete-time signal is  $\pi$  radians/sample.

#### Relationship between digital frequency and analog frequency

Note that the above analysis is done without any consideration of the sampling period or equivalently the time between adjacent samples. We consider the frequency variable for this case as the normalized frequency (or digital frequency) with unit radians/sample.

If we have  $F_s$  samples per second, and we wish to find the corresponding analog frequency with units radians/second, we write

$$\omega_0$$
 radians/sample =  $\omega_0$  radians/ $\{(1/F_s) \text{ second}\} = F_s\omega_0 \text{rad/s}$ .

Similarly

$$\Omega_0 = \omega_0 F_s = \frac{\omega_0}{T}$$

16

where  $T = 1/F_s$  is the sampling period. Thus, the normalized frequency  $\omega$  and the corresponding analog frequency  $\Omega$  are related according to

$$\omega = \frac{\Omega}{F_s} = \Omega T$$

and

$$\Omega = \omega F_s = \frac{\omega}{T}$$

## 2.8 Problems

1. Consider the signal x[n] in Figure 2.10. Sketch the transformed signals  $x_1[n],\,x_2[n],\,x_3[n]$  and

![](_page_15_Figure_8.jpeg)

Figure 2.10: Input signal for Problem 1.

 $x_4[n]$  defined below.

- (a)  $x_1[n] = x[-n]$
- (b)  $x_2[n] = x[3n]$
- (c)  $x_3[n] = x[3n+1]$
- (d)  $x_4[n] = x[3-2n]$
- 2. Consider the signal x[n] shown in Figure 2.11. Sketch y[n] = x[-3n + 6].
- 3. Express the following in their simplest possible form:
  - (a)  $(n+1)\delta[n]$
  - (b)  $(n^2+1)\delta[n-1]$
  - (c)  $\cos(0.2\pi n + \pi/4)\delta[n]$
  - (d)  $\sum_{n=-\infty}^{\infty} (n+1)\delta[n]$
  - (e)  $\sum_{n=-\infty}^{\infty} \frac{n+1}{n^2-1} \delta[n-5]$
- 4. Evaluate the following sums:

2.8. PROBLEMS 17

![](_page_16_Figure_1.jpeg)

Figure 2.11: Input signal for Problem 2.

(a) 
$$\sum_{n=-\infty}^{\infty} \left\{ e^{0.3n+j2} u[n] \delta[n+2] + \cos(2n+\frac{\pi}{3}) \delta[n-2] \right\}$$

(b) 
$$\sum_{m=-\infty}^{n} \{\delta[m+1] - \delta[m-1]\}$$

*Hint*: The result should be a signal that is a function of t.

(c) 
$$\sum_{n=-6}^{6} (n^2 + nt + 1)\delta[3n - 6]$$

Hint: For what value of n is the impulse function non-zero in this problem?

(d) 
$$\sum_{m=-3}^{3} \cos(0.04n - 2)\delta[n - 5].$$

5. A unit ramp signal is defined as

$$r[n] = \begin{cases} n+1 & ; & n \ge 0 \\ 0 & l & \text{otherwise} \end{cases}$$

- (a) Show that r[n] = (n+1)u[n], where u[n] is the unit step function.
- (b) Sketch and label r[n] in the range  $-6 \le n \le 6$ .
- (c) Show that u[n] = r[n] r[n-1].
- (d) Show that  $r[n] = \sum_{m=-\infty}^{n} u[m]$ .

6. Consider a system whose input output relationship is given by

$$y[n] = x[-2n - 4]$$

- (a) Qualitatively explain the set of operations you have to perform to get the above transformations. (Be specific.)
- (b) Sketch the output of the system when its input is given by the signal in Figure 2.12.

![](_page_17_Figure_2.jpeg)

Figure 2.12: Input signal to the system in Problem 6.

7. Let

$$x(t) = -\delta[2n + 10] + 2\delta[n] - 0.5\delta[n - 2]$$

Evaluate and sketch y[n] given by

$$y[n] = \sum_{m - \infty}^{n} x[m]$$

8. Express  $5\cos(0.25\pi n + 0.3\pi) + 2\sin(0.25\pi n - 0.1\pi)$  in the form  $A\cos(0.25\pi n + \varphi)$ . Explicitly determine A and  $\varphi$ . You may use a calculator to simplify the numbers.