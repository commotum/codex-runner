# Contents

| 6 | Fou | rier analysis of discrete-time signals                     |
|---|-----|------------------------------------------------------------|
|   | 6.1 | Discrete-time Fourier transform                            |
|   | 6.2 | Some properties of discrete-time Fourier transform         |
|   |     | 6.2.1 Linearity                                            |
|   |     | 6.2.2 Periodicity                                          |
|   |     | 6.2.3 Symmetry                                             |
|   |     | 6.2.4 Delay                                                |
|   |     | 6.2.5 Convolution                                          |
|   |     | 6.2.6 Modulation                                           |
|   | 6.3 | Fourier series expansion of periodic discrete-time signals |
|   |     | 6.3.1 Computing the Fourier series coefficients            |
|   | 6.4 | Problems                                                   |

2 CONTENTS

# Chapter 6

# Fourier analysis of discrete-time signals

Much of our studies have so far been devoted to the analysis and design of linear, time-invariant systems. We now change gear and focus on the analysis of signals. In particular, we consider the representation of signals in the frequency domain. Such representations involve expressing a signal x[n] as a sum or, in the limiting case, an integral of complex sinusoids of the form  $e^{j\omega n}$  with different frequencies and amplitudes. We prefer to use complex sinusoids rather than real sinusoids in these representations because the calculations involved are often easier to perform with complex-valued sinusoids than real-valued sinusoids.

## 6.1 Discrete-time Fourier transform

We learned about discrete-time Fourier transform (dtFt) without using the term Fourier transform when we studied about frequency response of linear time-invariant systems. Given the impulse response h[n] of a linear, time-invariant system, its frequency response is given by

$$H(e^{j\omega}) = \sum_{n=-\infty}^{\infty} h[n]e^{-j\omega n}$$
(6.1)

This is also the discrete-time Fourier transform of h[n]. We extend the notion of discrete-time Fourier transform to all discrete-time signals and define the dtFt of a signal x[n] to be

$$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n]e^{-j\omega n}$$
(6.2)

Given  $X(e^{j\omega})$  of a discrete-time signal x[n], we can calculate x[n] using the inverse dtFt formula

$$x[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega}) e^{j\omega n} d\omega \tag{6.3}$$

The expressions in (6.2) and (6.3) define the discrete-time Fourier transform and the inverse discrete-time Fourier transform, respectively. Effectively, the Fourier transform relationship tells us how we can express a time-domain signal as a linear combination (in this case, an integral) of complex sinusoidal signals. This representation is particularly important because complex sinusoids are eigensignals of linear, time-invariant systems. Because of this, we can compute the Fourier

transform of the output of a linear, time-invariant system as the product of the Fourier transform of its input signal and the frequency response of the system. We will prove this statement later when we consider the properties of Fourier transform.

The two representations of the same signal x[n] and  $X(e^{j\omega})$  are known as the Fourier transform pair, and is usually denoted using

$$x[n] \Leftrightarrow X(e^{j\omega})$$
 (6.4)

or

$$X(e^{j\omega}) = \mathcal{F}\{x[n]\}\tag{6.5}$$

or

$$x[n] = \mathcal{F}^{-1}\{X(e^{j\omega})\}\tag{6.6}$$

#### Proof of inverse dtFt formula

In (6.3), we introduced the inverse dtFt formula without offering any proof. Now we show that the right-hand side of this equation does compute the time domain sample x[n]. For this, we substitute (6.2) in (6.3) after changing the time variable n to m to avoid any confusion. That is,

$$\frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega}) e^{j\omega n} d\omega = \frac{1}{2\pi} \int_{-\pi}^{\pi} \left( \sum_{m=-\infty}^{\infty} x[m] e^{-j\omega m} \right) e^{j\omega n} d\omega$$
 (6.7)

We change the order in which the integration and summation are performed on the right-hand side of the above equation, combine the two exponential terms and take x[m] outside the integration operation to get

$$\frac{1}{2\pi} \int_{-\pi}^{\pi} \left( \sum_{m=-\infty}^{\infty} x[m] e^{-j\omega m} \right) e^{j\omega n} d\omega = \sum_{m=-\infty}^{\infty} x[m] \left( \frac{1}{2\pi} \int_{-\pi}^{\pi} e^{j\omega(n-m)} d\omega \right)$$
 (6.8)

Now, when m = n,

$$\frac{1}{2\pi} \int_{-\pi}^{\pi} e^{j\omega(n-m)} d\omega = \frac{1}{2\pi} \int_{-\pi}^{\pi} e^{j\omega(0)} d\omega = \frac{1}{2\pi} \int_{-\pi}^{\pi} (1) d\omega = 1$$
 (6.9)

When  $m \neq n$ 

$$\frac{1}{2\pi} \int_{-\pi}^{\pi} e^{j\omega(n-m)} d\omega = \frac{1}{2\pi} \left. \frac{e^{j\omega(n-m)}}{j(n-m)} \right|_{\pi}^{\pi} = 0 \tag{6.10}$$

The result of the integration is zero because  $e^{j\pi(n-m)} = e^{-j\pi(n-m)}$  for all integer values of n and m. Combining (6.9) and (6.10), we can see that the multiplier for x[m] in (6.8) is 1 when m=n and zero for all other values of m. Using this, we see that the inverse dtFt formula becomes

$$\frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega}) e^{j\omega n} d\omega = \sum_{m=-\infty}^{\infty} x[m] \left( \frac{1}{2\pi} \int_{-\pi}^{\pi} e^{j\omega(n-m)} d\omega \right)$$

$$= x[n](1) + \sum_{m=-\infty}^{\infty} x[m](0)$$

$$m \neq n$$

$$= x[n] \tag{6.11}$$

demonstrating that the inverse dtFt formula correctly computes the time domain signal x[n] from its discrete-time Fourier transform.

#### Exercise 6.1

Find the discrete-time Fourier transform of

$$x[n] = \begin{cases} 1 & ; & -N \le n \le N \\ 0 & ; & \text{otherwise} \end{cases}$$

Answer: Using the definition of dtFt,

$$X(e^{j\omega}) = \sum_{n=-N}^{N} (1)e^{-j\omega n}$$

$$= e^{j\omega N} \sum_{n=0}^{2N} e^{-j\omega n}$$

$$= e^{j\omega N} \frac{1 - e^{-j\omega(2N+1)}}{1 - e^{-j\omega}}$$

We can further simplify this by taking out (as a common factor  $e^{-j\frac{\omega}{2}}$  from both terms of the denominator and  $e^{-j\frac{\omega(2N+1)}{2}}$  from both terms of the numerator. This gives

$$X(e^{j\omega}) = \left(e^{j\omega N}\right) \left(\frac{e^{-j\frac{\omega(2N+1)}{2}}}{e^{-j\frac{\omega}{2}}}\right) \left(\frac{e^{j\frac{\omega(2N+1)}{2}} - e^{-j\frac{\omega(2N+1)}{2}}}{e^{j\frac{\omega}{2}} - e^{-j\frac{\omega}{2}}}\right)$$

The exponentials within the first two parentheses cancel each other. Dividing both the numerator and demoninator within the third parenthesis with 2j results in

$$X(e^{j\omega}) = \frac{\sin\left(\frac{\omega(2N+1)}{2}\right)}{\sin\left(\frac{\omega}{2}\right)}$$

Something to think about: Recall from our study of frequency response that frequency response of a discrete-time, linear time-invariant system is periodic with period  $2\pi$  radians/sample. Since the definition of dtFt is identical to the definition of frequency response, discrete-time Fourier transform is also periodic in  $\omega$  with period  $2\pi$  radians/sample. You should verify that the dtFt in Exercise 6.1 is periodic and that the period is  $2\pi$  radians/sample.

# 6.2 Some properties of discrete-time Fourier transform

We consider only the most important properties of dtFt, from the perspective of our class.

#### 6.2.1 Linearity

It is left as an exercise for the student to show that

$$\mathcal{F}\left\{\alpha x_1[n] + \beta x_2[n]\right\} = \alpha \mathcal{F}\left\{x_1[n]\right\} + \beta \mathcal{F}\left\{x_2[n]\right\}$$
(6.12)

where  $\alpha$  and  $\beta$  are two arbitrary constants.

#### 6.2.2 Periodicity

The discrete-time Fourier transform  $X(e^{j\omega})$  is periodic with period  $2\pi$  radians/sample. To derive this property, we write

$$X(e^{j(\omega+2\pi)}) = \sum_{n=-\infty}^{\infty} x[n]e^{-j(\omega+2\pi)n}$$

$$= \sum_{n=-\infty}^{\infty} x[n]e^{-j\omega n}e^{-j2\pi n}$$

$$= \sum_{n=-\infty}^{\infty} x[n]e^{-j\omega n} = X(e^{j\omega})$$
(6.13)

This shows that dtFt is periodic and that the period is  $2\pi$ .

## 6.2.3 Symmetry

If x[n] is real-valued, its Fourier transform is a complex conjugate even function of frequency, i. e.,

$$X(e^{j\omega}) = X^*(e^{-j\omega}) \tag{6.14}$$

We have seen this property in the context of frequency response of linear, time-invariant systems. The derivation is left as an exercise for the students. As a consequence of the symmetry property, we can show that the real part and also the magnitude of the discrete-time Fourier transform of a real-valued signal are even functions of frequency. Similarly, the imaginary part and the phase of the discrete-time Fourier transform of real-valued signals are odd functions of frequency.

#### 6.2.4 Delay

Let  $\mathcal{F}\{x[n]\} = X(e^{j\omega})$ . We now show that

$$\mathcal{F}\{x[n-m]\} = X(e^{j\omega})e^{-j\omega m} \tag{6.15}$$

That is, when a signal is delayed by m samples, only the phase of its Fourier transform changes, and the change itself is directly proportional to the frequency. To derive the above result, we start with

$$\mathcal{F}\{x[n-m]\} = \sum_{n=-\infty}^{\infty} x[n-m]e^{-j\omega n}$$
(6.16)

We use a change of variables k = n - m in the above expression to rewrite it as

$$\mathcal{F}\{x[n-m]\} = \sum_{k=-\infty}^{\infty} x[k]e^{-j\omega(k+m)}$$

$$= e^{-j\omega m} \sum_{k=-\infty}^{\infty} x[k]e^{-j\omega k}$$

$$= e^{-j\omega m} X(e^{j\omega})$$
(6.17)

proving the result.

#### Exercise 6.2

Find the Fourier transform of

$$x[n] = \begin{cases} 1 & ; & 0 \le n \le 2N \\ 0 & ; & \text{otherwise} \end{cases}$$

Answer: We note that the above signal is the signal in Exercise 6.1 delayed by N samples. Using the delay property, and using the results from that exercise, we can immediately show that

$$X(e^{j\omega}) = \frac{\sin\left(\frac{\omega(2N+1)}{2}\right)}{\sin\left(\frac{\omega}{2}\right)} e^{-j\omega N}$$

We verify this result by direct evaluation of the Fourier transform.

$$X(e^{j\omega}) = \sum_{n=0}^{2N} e^{-j\omega n}$$
$$= \frac{1 - e^{-j\omega(2N+1)}}{1 - e^{-j\omega}}$$

Simplifying as in Exercise 6.1, we get

$$X(e^{j\omega}) = \left(\frac{e^{-j\frac{\omega 2N+1}{2}}}{e^{-j\frac{\omega}{2}}}\right) \left(\frac{e^{j\frac{\omega(2N+1)}{2}} - e^{-j\frac{\omega(2N+1)}{2}}}{e^{j\frac{\omega}{2}} - e^{-j\frac{\omega}{2}}}\right) = e^{-j\omega N} \left(\frac{\sin\left(\frac{\omega(2N+1)}{2}\right)}{\sin\left(\frac{\omega}{2}\right)}\right)$$

which is identical to the initial results we obtained.

Transforming a non-causal filter to a causal filter: Let h[n] be the unit impulse response of a non-causal filter, and let the first non-zero sample of h[n] be at time  $n_0$ . Then,  $h[n-n_0]$  represents the unit impulse signal of a causal filter. Its output will arrive  $n_0$  samples later than the outout of the original (non-causal) filter. The magnitudes of the Fourier transforms of the outputs will be identical for both filters. By the delay property of dtFt, the phase of the Fourier transform of the output signal of the causal filter will be shifted by  $-n_0\omega$  radians when compared with the phase of the Fourier transform of the output signal of the non-causal filter.

#### 6.2.5 Convolution

Let the discrete-time Fourier transforms of x[n] and h[n] be given by  $X(e^{j\omega})$  and  $H(e^{j\omega})$ , respectively. Then

$$y[n] = x[n] * h[n] \Leftrightarrow X(e^{j\omega})H(e^{j(\omega)})$$
(6.18)

In the above, \* denotes the convolution operation. We have seen and used this result in the context of linear, time-invariant systems in the past. We just verify the results here.

$$Y(e^{j\omega}) = \sum_{n=-\infty}^{\infty} y[n]e^{-j\omega n}$$

$$= \sum_{n=-\infty}^{\infty} \sum_{m=-\infty}^{\infty} x[m]h[n-m]e^{-j\omega n}$$
(6.19)

We change the order of summations to express the above result as

$$Y(e^{j\omega}) = \sum_{m=-\infty}^{\infty} x[m] \left\{ \sum_{n=-\infty}^{\infty} h[n-m]e^{-j\omega n} \right\}$$
(6.20)

The inner sum is the discrete-time Fourier transform of h[n-m] and can be evaluated as  $H(e^{j\omega})e^{-j\omega m}$  using the delay property. Then,

$$Y(j\omega) = \sum_{m=-\infty}^{\infty} x[m]H(e^{j\omega})e^{-j\omega m}$$

$$= H(e^{j\omega})\sum_{m=-\infty}^{\infty} x[m]e^{-j\omega m}$$

$$= X(e^{j\omega})H(e^{j\omega})$$
(6.21)

giving us the desired result.

#### Exercise 6.3

Find the Fourier transform of the triangular signal given by

$$x[n] = \begin{cases} 2N + 1 - |n| & ; \quad -2N \le n \le 2N \\ 0 & ; \quad \text{otherwise} \end{cases}$$

Answer: You should verify that the signal defined above is triangular in shape. It is also left as an exercise for the student to show that x[n] can be obtained by convolving

$$x_1[n] = \begin{cases} 1 & ; & -N \le n \le N \\ 0 & ; & \text{otherwise} \end{cases}$$

with itself. We saw in Exercise 6.1 that the dtFt  $X_1(e^{j\omega})$  of  $x_1[n]$  is given by

$$X_1(e^{j\omega}) = \frac{\sin\left(\frac{\omega(2N+1)}{2}\right)}{\sin\left(\frac{\omega}{2}\right)}$$

By the convolution property of the Fourier transform

$$X(e^{j\omega}) = X_1^2(e^{j\omega}) = \left(\frac{\sin\left(\frac{\omega(2N+1)}{2}\right)}{\sin\left(\frac{\omega}{2}\right)}\right)^2$$

An alternate approach to computing the convolution of two signals: The convolution property of discrete-time Fourier transform provides us an alternate method for computing the convolution of two signals. Let  $H(e^{j\omega})$  and  $X(e^{j\omega})$  represent the dtFt of h[n] and x[n], respectively. Then, by the convolution property, we can compute y[n], the convolution of h[n] and x[n], as

$$y[n] = \mathcal{F}^{-1} \left\{ H(e^{j\omega}) X(e^{j\omega}) \right\}$$

Here,  $\mathcal{F}^{-1}$  represents the inverse dtFt operation.

#### 6.2.6 Modulation

Let 
$$X(e^{j\omega}) = \mathcal{F}\{x[n]\}$$
. Then,  

$$\mathcal{F}\left\{x[n]e^{j\omega_0 n}\right\} = X(e^{j(\omega - \omega_0)}) \tag{6.22}$$

That is, we can shift the frequency contents of a signal by  $\omega_0$  radians/sample by simply multiplying the signal with a complex sinusoid of frequency  $\omega_0$  radians/sample. To see this result, we evaluate the discrete-time Fourier transform of  $x[n]e^{j\omega_0 n}$  directly to get

$$\mathcal{F}\left\{x[n]e^{j\omega_0 n}\right\} = \sum_{n=-\infty}^{\infty} x[n]e^{j\omega_0 n}e^{-j\omega_n}$$

$$= \sum_{n=-\infty}^{\infty} x[n]e^{-j(\omega-\omega_0)n}$$

$$= X(e^{j(\omega-\omega_0)})$$
(6.23)

#### Exercise 6.4

Find the discrete-time Fourier transform of

$$x[n] = 0.8^n \cos(0.5\pi n)u[n]$$

Answer: We first find the Fourier transform of  $x_1[n] = 0.8^n u[n]$  to get

$$X_1(e^{j\omega}) = \sum_{n=0}^{\infty} 0.8^n e^{-j\omega n}$$
$$= \frac{1}{1 - 0.8e^{-j\omega}}$$

Since  $\cos(\omega_0 n) = (e^{j\omega_0 n} + e^{-j\omega_0 n})/2$ , we can use the modulation and linearity properties of the Fourier transform to obtain the Fourier transform of x[n] as

$$X(e^{j\omega}) = \mathcal{F}\left\{\frac{x_1[n]e^{j0.5\pi n} + x_1[n]e^{-j0.5\pi}}{2}\right\}$$
$$= \frac{0.5}{1 - 0.8e^{-j(\omega - 0.5\pi)}} + \frac{0.5}{1 - 0.8e^{-j(\omega + 0.5\pi)}}$$

An application of the modulation property of discrete-time Fourier transform involves design of bandpass and highpass filters.

Transforming lowpass filter designs to bandpass and highpass filter designs: Let the real-valued signal h[n] represent the unit impulse response of a lowpass filter. Then, the frequency response of a new filter with impulse response  $2h[n]\cos(\omega_0 n)$  will have bandpass characteristics with the passband centered around  $+\omega_0$  radians/sample for positive frequencies and around  $-\omega_0$  radians/sample for negative frequencies. A filter with impulse response signal  $h[n]e^{j\pi n} = (-1)^n h[n]$  will have highpass characteristics with the passpand shifting to center around  $\pm \pi$  radians/sample.

# 6.3 Fourier series expansion of periodic discrete-time signals

If we attempt to directly compute the discrete-time Fourier transform of a periodic signal, we will run into problems because the summation in the dtFt formula will not converge. For periodic signals, we need an alternate approach, provided by the concept of Fourier series expansion of periodic signals.

Let x[n] be a periodic signal with period N samples. Then, the frequency  $f_0 = 1/N$  cycles/sample is said to be the fundamental frequency of this signal. When expressed in radians/sample, the fundamental frequency is given by  $\omega_0 = 2\pi/N$  radians/sample. Integer multiples of the fundamental frequency are known as the harmonics of the fundamental frequency. The periodic signal x[n] can be expressed as a sum of sinusoidal components with the fundamental frequency and its harmonics in the form

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X(k)e^{j\frac{2\pi}{N}kn}$$
(6.24)

The constants X(k) are known as the discrete-time Fourier series coefficients of x[n]. The expansion given above is known as the Fourier series expansion of x[n]. We may visualize X(k)/N as the amplitude of the kth harmonic frequency of the fundamental frequency. The above statements suggest that any periodic signal with period = N samples can be written as a combination of N everlasting complex sinusoids with frequencies that are integer multiples of the fundamental frequency.

### 6.3.1 Computing the Fourier series coefficients

It is easy to show that

$$\sum_{n=0}^{N-1} e^{j\frac{2\pi}{N}kn} e^{-j\frac{2\pi}{N}mn} = \begin{cases} N & ; & k=m\\ 0 & ; & \text{otherwise} \end{cases}$$
 (6.25)

Here, the variables k and m are arbitrary integers between 0 and N-1. The proof of the above result is straightforward. If k=m,

$$\sum_{n=0}^{N-1} e^{j\frac{2\pi}{N}kn} e^{-j\frac{2\pi}{N}mn} = \sum_{n=0}^{N-1} e^{j\frac{2\pi}{N}(0)n} = \sum_{n=0}^{N-1} (1) = N$$

If  $k \neq m$ ,

$$\sum_{n=0}^{N-1} e^{j\frac{2\pi}{N}kn} e^{-j\frac{2\pi}{N}mn} = \sum_{n=0}^{N-1} e^{j\frac{2\pi}{N}(k-m)n} = \frac{1 - e^{j\frac{2\pi}{N}(k-m)N}}{1 - e^{j\frac{2\pi}{N}(k-m)}} = 0$$

The numerator in the closed form expression for the sum of the geometric series above is zero because  $e^{j\frac{2\pi}{N}(k-m)N} = 1$  when  $k \neq m$ .

This result gives us a way to calculate X(k) from the samples of the periodic signal x[n]. We will first state the result and then show that it is indeed correct. To evaluate X(k), we multiply x[n] with  $e^{-j\frac{2\pi}{N}kn}$  and add the product samples over one period to get

$$X(k) = \sum_{n=0}^{N-1} x[n]e^{-j\frac{2\pi}{N}kn}$$
(6.26)

Substituting the Fourier series expansion of x[n] in the right-hand side of the above equation, we get

$$\sum_{n=0}^{N-1} x[n] e^{-j\frac{2\pi}{N}kn} = \sum_{n=0}^{N-1} \left( \frac{1}{N} \sum_{m=0}^{N-1} X(m) e^{j\frac{2\pi}{N}mn} \right) e^{-j\frac{2\pi}{N}kn} 
= \sum_{m=0}^{N-1} X(m) \left( \frac{1}{N} \sum_{n=0}^{N-1} e^{j\frac{2\pi}{N}(m-k)n} \right)$$
(6.27)

While deriving the above expression, we interchanged the order of the summation operations. Now, of all the values of m, there is only one value for which the inner summation in the above expression is non-zero. This occurs when m = k. That is, X(m) is multiplied by zero for all values of m except when m = k, at which time, the multiplier is 1. Therefore, the summation above becomes

$$\sum_{n=0}^{N-1} x[n]e^{-j\frac{2\pi}{N}kn} = X(k)$$
(6.28)

giving us a way to compute the Fourier series coefficients of periodic discrete-time signals.

Something to remember: The summation in (6.26) extends from 0 to N-1. However, remember that, in addition to x[n] being periodic with period N samples, all of the everlasting complex sine waves of the form  $e^{j\frac{2\pi}{N}kn}$  are also periodic with period N samples. This means that the product terms of the form  $x[n]e^{j\frac{2\pi}{N}kn}$  are also periodic with the same period. This implies that the summation range can be any one complete period of length N samples. That is, the range of summation can be 0 to N-1, or 10N to 11N-1, or -5 to N-6 or any range of one full period.

#### Exercise 6.5

Find the Fourier series expansion for a periodic signal x[n] with period N samples when the first period is described by

$$x[n] = \begin{cases} 1 & ; & -M \le n \le M; & 2M+1 < N \\ 0 & ; & \text{otherwise} \end{cases}$$

Answer: The Fourier series coefficients are given by

$$X(k) = \sum_{n=-M}^{M} (1)e^{-j\frac{2\pi}{N}kn}$$

$$= e^{j\frac{2\pi}{N}kM} \sum_{n=0}^{2M+1} e^{-j\frac{2\pi}{N}kn}$$

$$= e^{j\frac{2\pi}{N}kM} \frac{1 - e^{-j\frac{2\pi}{N}k(2M+1)}}{1 - e^{j\frac{2\pi}{N}k}}$$

$$= \frac{e^{j\frac{2\pi}{N}kM} \frac{1 - e^{-j\frac{2\pi}{N}k(2M+1)}}{1 - e^{j\frac{2\pi}{N}k}}$$

$$= \frac{e^{j\frac{2\pi}{N}k(\frac{2M+1}{2})} - e^{-j\frac{2\pi}{N}k(\frac{2M+1}{2})}}{e^{j\frac{2\pi}{N}\frac{k}{2}} - e^{-j\frac{2\pi}{N}\frac{k}{2}}}$$

$$= \frac{\sin\left(\frac{2\pi}{N}k\frac{2M+1}{2}\right)}{\sin\left(\frac{2\pi}{N}\frac{k}{2}\right)}$$

The simplifications in the steps to calculate X(k) above followed the process used in Exercise 6.1. For example, in the second line of the calculations above, we changed the range of summation from -M-+M to 0-2M after taking out  $e^{j\frac{2\pi}{N}kn}$  as a common factor from all terms. In line 4, we took out  $e^{-j\frac{2\pi}{N}k\frac{2M+1}{2}}$  from both terms in the numerator and  $e^{-j\frac{2\pi}{N}\frac{k}{2}}$  from both terms in the denominator. Here, we also recognized that all the phase terms we extracted will cancel each other. Finally in line 5, we divided both the numerator and denominator by 2j and recognized the real-valued sinusoids that then resulted.

#### 6.4 Problems

- 1. Find the discrete-time Fourier transform of  $x[n] = (-0.9)^n \cos(\frac{\pi}{2}n)u[n]$ .
- 2. Two discrete-time, linear, time-invariant systems connected in series (cascade) have unit impulse responses given by  $h_1[n] = 0.9^n u[n] 0.5(0.9)^{n-1} u[n-1]$  and  $h_2[n] = 0.5^n u[n] 0.9(0.5)^{n-1} u[n-1]$ , respectively. Show that the series (cascade) connection of the system produces an identity system, *i.e.*, the unit impulse response function of the cascade is a unit impulse function, or equivalently, the output y[n] of the cascade is the same as the input signal x[n].

Hint: Use the convolution property of dictrete-time Fourier transform to solve this problem.

3. Consider a complex-valued impulse response function of a discrete-time linear, time-invariant system given by

$$h[n] = 0.8^n e^{j(\frac{\pi}{6})n} u[n]$$

- (a) Find the frequency response  $H(e^{j\omega})$  of this system.
- (b) Show that the frequency response above is periodic. If you do not agree with this statement, explain why this is incorrect. If periodic, find the period of the frequency response.
- (c) Does the frequency response exhibit odd and even symmetries? Why or why not?

6.4. PROBLEMS

- 4. Let x[n] be a signal of length N samples such that x[n] = 0 for n < 0 and for  $n \ge N$ .
  - (a) Write down the expression for  $X(e^{j\omega})$ , the discrete-time Fourier transform of x[n]. Make sure that the range of summation is over N samples only.
  - (b) Consider a new signal  $x_p[n]$  defined as

$$x_p[n] = \sum_{k=-\infty}^{\infty} x[n-kN]$$

That is, we get  $x_p[n]$  by shifting x[n] by integer multiples of N samples and adding all such shifted versions together. Show that  $x_p[n]$  is a periodic signal with period N samples, i. e.,  $x_p[n] = x_p[n+N]$ .

(c) Write down an expression for  $X_p(k)$ , the discrete-time Fourier series coefficients of  $x_p[n]$ . Recall from our knowledge of Fourier series expansion that

$$x_p[n] = \sum_{k=0}^{N-1} X_p(k) e^{j\frac{2\pi}{N}kn}$$

where  $X_p(k)$  is the amplitude of the everlasting complex sine wave with frequency  $\frac{2\pi}{N}k$  radians/sample. That is,  $X_p(k)$  is the amplitude of the kth harmonic of the fundamental frequency  $\frac{2\pi}{N}$  radians/sample.

(d) Show by comparing the results in (a) and (c) and show that

$$X_p(k) = \frac{1}{2\pi} \left. X(e^{j\omega}) \right|_{\omega = \frac{2\pi}{N}k}$$

This shows that the amplitude of the kth harmonic frequency in the Fourier series expansion of a periodic signal can be computed by finding the value of the discrete-time Fourier transform of the first period of the periodic signal at the kth harmonic frequency and scaling it by  $2\pi$ .

- 5. Let h[n] be the impulse response of a discrete-time, linear, time-invariant system. It is known that h[n] is real-valued, and non-zero over some finite range of values of time n. It is also known that h[n] is an odd function of time, i. e., h[n] = -h[-n].
  - (a) Is the system causal? Why or why not?
  - (b) Write an expression that defines the frequency response  $H(e^{j\omega})$  of this system.
  - (c) Now consider a linear, time-invariant system with unit impulse response signal g[n], given by g[n] = -h[-n]. Keep in mind that because h[n] is an odd function of time, g[n] = h[n]. Show, using the fact that g[n] = -h[-n] that the frequency response  $G(e^{j\omega})$  of this system is  $-H^*(e^{j\omega})$ , where  $(\cdot)^*$  denotes the complex conjugate of  $(\cdot)$ .
  - (d) Now, because g[n] = h[n],  $G(e^{j\omega})$  must also be equal to  $H(e^{j\omega})$ . Combining this result and the previous results, show that  $H(e^{j\omega}) = -H^*(e^{j\omega})$ .
  - (e) Using the above result, show that the real part of the frequency response  $H(e^{j\omega})$  is zero, that is,  $H(e^{j\omega})$  is a purely imaginary function when h(t) is real-valued and odd.
- 6. Consider a discrete time signal x[n] with dtFt given by

$$X(e^{j\omega}) = \begin{cases} 1 & ; & |\omega| \le \omega_0 \\ 0 & ; & \text{otherwise} \end{cases}$$

(a) Show by direct calculation of the inverse discrete-time Fourier transform that

$$x[n] = \frac{\sin(\omega_0 n)}{\pi n}$$

(b) Let h[n] be a different signal given by

$$h[n] = \frac{\sin(\omega_1 n)}{\pi n}$$

where  $\omega_1 < \omega_0$ . Show, using the convolution property of discrete-time Fourier transform, that the result of convolving x[n] with h[n] is h[n].

7. Find the discrete-time Fourier series expansion of a periodic signal x[n] with period 30 samples when

$$x[n] = \begin{cases} 1 & ; & 0 \le n \le 9 \\ 0 & ; & \text{otherwise in the first period.} \end{cases}$$