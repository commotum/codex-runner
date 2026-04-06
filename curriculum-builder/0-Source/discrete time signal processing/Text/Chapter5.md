# Contents

| 5 | Free | quency response of linear, time-invariant systems | 3 |
|---|------|---------------------------------------------------|---|
|   | 5.1  | Output of lti systems for real-valued sinusoids   | 5 |
|   | 5.2  | Examples on frequency response                    | 6 |
|   | 5.3  | Problems                                          | 9 |

2 CONTENTS

# Chapter 5

# Frequency response of linear, time-invariant systems

Let us consider a discrete-time lti system with unit impulse response signal h[n]. We wish to find its output when the input is an everlasting, complex sinusoid of the form

$$x[n] = Ae^{j\omega_0 n} \tag{5.1}$$

In the above equation  $\omega_0$  represents an arbitrary frequency value.

Something to think about: The complex sinusoid in (5.1) is periodic in frequency with period  $2\pi$ . This is very easy to see since

$$Ae^{j\omega_0 + 2\pi)n} = Ae^{j\omega_0 n}e^{j2\pi} = Ae^{j\omega_0 n}$$

 $\sin \alpha \circ i 2\pi - 1$ 

It is common to consider the first period to be between  $-\pi$  and  $\pi$  radians/sample.<sup>1</sup> Thus the maximum normalized frequency of a discrete-time signal is  $\pi$  radians/sample or  $\pi/(2\pi) = 1/2$  cycles/sample.

How does the normalized frequency relate to the actual frequency of a continuous time signal that may have been sampled to generate the discrete-time signal? Let T be the sampling period so that  $F_s = 1/T$  is the sampling frequency in Hz. or cycles/s. Let  $f_0$  denote the normalized frequency. Then

$$f_0 \text{ cycles/sample} = f_0 \text{ cycles/T seconds} = f_0 F_s \text{ cycles/second}$$
 (5.2)

That is, we can get the true frequency from the normalized frequency by simply multiplying the normalized frequency with the sampling frequency.

It is important to make one more point at this time. Since the maximum normalized frequency is 1/2 cycle/sample, the highest frequency a discrete-time signal can have is  $F_s/2$  cycles/s. Multiplying this by  $2\pi$ , we can also see that the maximum frequency is  $\pi F_s$  radians/s.

<sup>&</sup>lt;sup>1</sup>Note the difference between the units of frequency in the discrete-time and continuous-time cases. Since we have omitted the sampling period in the representation of the signal (i.e., we write x[n] rather than x[nT] where T is the sampling period.), the concept of seconds as the unit of time is not appropriate. Rather, we would simply say so many samples instead of so many seconds. Essentially we are normalizing the time unit to samples instead of seconds. The range of frequencies in the first period between  $-\pi$  and  $\pi$  is also the range of normalized frequency values.

Calculation of the system output: The output of an lti system to any input is the convolution of input signal with the unit impulse response signal. Let y[n] be the output to the sinusoidal input. Then,

$$y[n] = \sum_{m=-\infty}^{\infty} h[m]x[n-m]$$

$$= \sum_{m=-\infty}^{\infty} h[m]Ae^{j\omega_0(n-m)}$$

$$= \sum_{m=-\infty}^{\infty} h[m]Ae^{j\omega_0n}e^{-j\omega_0m}$$

$$= Ae^{j\omega_0n} \left\{ \sum_{m=-\infty}^{\infty} h[m]e^{-j\omega_0m} \right\}$$
(5.3)

This result is an extremely important one, and true for all lti systems (continuous-time and discrete-time, as well as FIR and IIR systems).

The output of a linear, time-invariant system to an everlasting complex sinusoid is the input times a constant that depends only on the frequency of the sinusoid.

The complex constant is given by

$$H(e^{j\omega_0}) = \sum_{m=-\infty}^{\infty} h[m]e^{-j\omega_0 m}$$

$$\tag{5.4}$$

and is known as the frequency response of the filter. A consequence of this property is the following:

Linear time-invariant systems are frequency selective.

What this statement means is this: Linear, time-invariant systems treats sinusoids at different frequencies differently. We may be able to design an lti systems that attenuates or eliminates certain frequencies and amplifies or retains certain frequencies.

Something for the future: The frequency response of a linear, time-invariant system given by

$$H(e^{j\omega}) = \sum_{n=-\infty}^{\infty} h[n]e^{-j\omega n}$$
(5.5)

is the discrete-time Fourier transform of h[n].

Terminology: We found from this discussion that the output of a linear, time-invariant system to a complex sine wave is the same sine wave times a constant. Input signals that are changed only by a constant multiplier (This multiplier may be complex valued.) are known as eigensignals of the system. This concept is similar to that of eigenvectors of matrices. Recall that a vector  $\mathbf{v}$  is an eigenvector of the matrix  $\mathbf{A}$  if  $\mathbf{A}\mathbf{v} = \sigma\mathbf{v}$ .

## 5.1 Output of lti systems for real-valued sinusoids

In this Subsection, we consider lti systems with real-valued impulse response signals and evaluate their output signals when the input is a real-values sinusoid, say  $x[n] = \cos(\omega_0 n)$ . First, we recognize that the cosine function can be written as a sum of two complex-valued sine waves using Euler's identity:

$$x[n] = \cos(\omega_0 n) = \frac{1}{2} \left( e^{j\omega_0 n} + e^{-j\omega_0 n} \right)$$

$$(5.6)$$

This means that we can invoke linearity of the system to find the output as a sum of two signals, one of which is the output of the system to  $(1/2)e^{j\omega_0 n}$  and the other the response of the system to  $(1/2)e^{-j\omega_0 n}$ . Therefore, if we know the frequency response of the system at the frequencies  $\omega_0$  and  $-\omega_0$ , we can explicitly find the output of the system for our cosine signal.

Now, the frequency response for the frequency  $\omega_0$  is

$$H(e^{j\omega_0}) = \sum_{n=-\infty}^{\infty} h[n]e^{-j\omega_0 n}$$
(5.7)

and the frequency response for the frequency  $-\omega$  is

$$H(e^{-j\omega_0}) = \sum_{n=-\infty}^{\infty} h[n]e^{j\omega_0 n}$$
(5.8)

Since h[n] is real-valued, we can show that the complex conjugate of  $H(e^{j\omega_0})$  is identical to  $H(e^{-j\omega_0})$  as follows:

$$H^*(e^{j\omega_0}) = \left(\sum_{n = -\infty}^{\infty} h[n]e^{-j\omega_0 n}\right)^* = \sum_{n = -\infty}^{\infty} (h[n])^* e^{j\omega_0 n} = \sum_{n = -\infty}^{\infty} h[n]e^{j\omega_0 n} = H(e^{-j\omega_0})$$
 (5.9)

Let us rewrite the frequency response in polar form as

$$H(e^{j\omega_0}) = |H(e^{j\omega_0})|e^{j\theta(\omega_0)}$$

$$\tag{5.10}$$

where  $\theta(\omega_0)$  is the phase of the frequency response. Then, since  $H(e^{-j\omega_0})$  is the complex conjugate of  $H(e^{j\omega_0})$ , we have

$$H(e^{-j\omega_0}) = H^*(e^{j\omega_0}) = |H(e^{j\omega_0})|e^{-j\theta(\omega_0)}$$
(5.11)

Terminology: The magnitude of the frequency response function is known as the magnitude response. The phase of the frequency response function is known as the phase response.

The symmetry of the phase response described above is true only for lti systems with real-valued impulse response signals. This symmetry property is something we will make use of time and again, and is worth repeating.

For systems with real-valued impulse response signals, the magnitude response is an even function of frequency and the phase response is an odd function of frequency.

Calculation of the system output for real sinusoidal inputs: We make use of the linearity of the system and the concept of frequency response of lti systems to write

$$\mathcal{H}\left\{\cos(\omega_{0}n)\right\} = \frac{1}{2} \left(\mathcal{H}\left\{e^{j\omega_{0}n}\right\} + \mathcal{H}\left\{e^{-j\omega_{0}n}\right\}\right)$$

$$= \frac{1}{2} \left(|H(e^{j\omega_{0}})|e^{j\theta(\omega_{0})}e^{j\omega_{0}n} + |H(e^{j\omega_{0}})|e^{-j\theta(\omega_{0})}e^{-j\omega_{0}n}\right)$$

$$= |H(e^{j\omega_{0}})|\cos(\omega_{0}n + \theta(\omega_{0}))$$
(5.12)

The above result shows that the difference between how an lti system treats a complex sinusoid and how an lti system with real-valued impulse response signal treats a real-valued sine wave is not substantial. Specifically:

For systems with real-valued impulse response signals, the system output for a real-valued sinusoidal input is the same sinusoid with a different amplitude and a different phase. The amplitude of the input sine wave gets multiplied by the magnitude response of the system at the frequency of the input signal, and the phase phase of the sinusoidal input is changed at the output by the phase response of the system at the frequency of the input signal.

# 5.2 Examples on frequency response

#### Exercise 5.1

Find the frequency response of a discrete-time, lti system with unit impulse response

- 1.  $h[n] = 0.9^n u[n]$
- 2.  $h[n] = (-0.9)^n u[n]$
- 3.  $h[n] = (-0.9)^n \cos(\frac{\pi}{2}n)u[n]$

4. 
$$h[n] = \begin{cases} 1 & ; n = 0 \\ 0.81 & ; n = 2 \\ 0 & ; \text{ otherwise.} \end{cases}$$

Answer: Each system in the list has real-valued impulse response signals, and therefore, we know from earlier discussion that the frequency response will exhibit symmetry about the y axis. Specifically, the magnitude response will be symmetric about the y axis, and the phase response will be anti-symmetric about the y axis.

1. From the discussions above, we know that the frequency response is given by

$$H\{e^{j\omega}\} = \sum_{n=-\infty}^{\infty} h[n]e^{-j\omega n}$$
$$= \sum_{n=0}^{\infty} 0.9^n e^{-j\omega n}$$
$$= \frac{1}{1 - 0.9e^{-j\omega}}$$

Figure 5.1 displays the magnitude response and the phase response of this filter in the frequency range  $-\pi$  through  $\pi$  radians/sample. We can see from the results that the magnitude response is an even function of frequency, meaning that if we reflect the magnitude response about the y axis, we will get the same function back. The phase response, on the other hand, is an odd function of frequency in that if we reflect the phase response about the y axis, we will get the negative of the phase response. Because of this symmetry, it is common to plot the frequency response only the positive values of frequencies in the range  $[0, \pi]$ . For the rest of the exercise, we will plot the frequency response in this manner.

![](_page_6_Figure_2.jpeg)

Figure 5.1: Magnitude (left panel) and phase (right panel) responses of the system in Exercise 4.6, Problem 1.

![](_page_6_Figure_4.jpeg)

Figure 5.2: Magnitude (left panel) and phase (right panel) responses of the system in Exercise 4.6, Problem 2.

### 2. A similar analysis to Problem 1 will show that

$$H\{e^{j\omega}\} = \sum_{n=0}^{\infty} (-0.9)^n e^{-j\omega n}$$
$$= \frac{1}{1+0.9e^{-j\omega}}$$

Figure 5.2 displays the magnitude response and the phase response of this filter in the frequency range 0 through  $\pi$  radians/sample.

#### 3. The frequency response in this case is given by

$$H\{e^{j\omega}\} = \sum_{n=0}^{\infty} (-0.9)^n \cos(\frac{\pi}{2}n) e^{-j\omega n}$$

$$= \frac{1}{2} \sum_{n=0}^{\infty} (-0.9)^n e^{j\frac{\pi}{2}n} e^{-j\omega n} + \frac{1}{2} \sum_{n=0}^{\infty} (-0.9)^n e^{-j\frac{\pi}{2}n} e^{-j\omega n}$$

$$= \frac{1}{2} \left\{ \frac{1}{1 + 0.9e^{j\frac{\pi}{2}} e^{-j\omega}} + \frac{1}{1 + 0.9e^{-j\frac{\pi}{2}} e^{-j\omega}} \right\}$$

![](_page_7_Figure_2.jpeg)

Figure 5.3: Magnitude (left panel) and phase (right panel) responses of the system in Exercise 4.6, Problem 3.

![](_page_7_Figure_4.jpeg)

Figure 5.4: Magnitude (left panel) and phase (right panel) responses of the system in Exercise 4.6, Problem 4.

Combining the two terms in the last expression, we can show that

$$H(e^{j\omega}) = \frac{1 + 0.9\cos(\frac{\pi}{2})e^{-j\omega}}{1 + 1.8\cos(\frac{\pi}{2})e^{-j\omega} + 0.81e^{-j2\omega}}$$

Since  $\cos(\frac{\pi}{2}) = 0$ , the expression for the frequency response of the system in this problem becomes

$$H(e^{j\omega}) = \frac{1}{1 + 0.81e^{-j2\omega}}$$

Figure 5.3 displays the magnitude response and the phase response of this filter in the frequency range 0 through  $\pi$  radians/sample.

4. The frequency response in this case is given by

$$H(e^{j\omega}) = 1 + 0.81e^{-j2\omega}$$

Figure 5.4 displays the magnitude response and the phase response of this filter in the frequency range 0 through  $\pi$  radians/sample.

Something to think about: The four examples above clearly illustrates the frequency selective nature of linear time-invariant filters since each of them treat input signals differently based on their

5.3. PROBLEMS

![](_page_8_Figure_1.jpeg)

Figure 5.5: Magnitude response of four types of ideal filters.

frequencies. Furthermore, we can see that four filters have very different characteristics. For example, the first filter attenuates high frequency signals while processing low-frequency signals with less attenuation. This is the characteristic of a lowpass filter. The second filter behaved in exactly the opposite manner - it passed through the high frequency signals with less attenuation while severely attenuating the low-frequency signals. This is how a highpass filter behaves. The third system attenuated both low-frequency and high-frequency signals more while processing signals with frequencies in between low and high values with less attenuation. This is what bandpass filters do. Finally, the last filter attenuated mid-frequency signals more while retaining low and high frequency signals with less attenuation at the output. This represents the characteristic of a bandstop filter.

The magnitude responses of *ideal* lowpass, highpass, bandpass and bandstop filters are as shown in Figure 5.5. Unfortunately, it is impossible to design practical systems that shows ideal behavior. In practice, we try to *design* implementable filters that can approximate the ideal characteristics. Keep in mind for all these discussions that the highest frequency a discrete-time signal can have is  $\pi$  radians/sample.

#### 5.3 Problems

1. Suppose that we wish to sample a sine wave of the form

$$x(t) = \cos(\omega_0 t)$$

every T seconds. Therefore the sampling frequency is  $F_s = \frac{1}{T}$  samples/s. The discrete-time signal so obtained is given by

$$x[n] = \cos(\omega_0 T n) = \cos(\Omega_0 n)$$

where  $\Omega_0$  is the normalized frequency of the discrete-time signal. We say that the signal is normalized because we have completely suppressed information about the sampling rate in the description of the signal, and it looks as if the sampling rate was one sample/second. Find the normalized frequency of the signal x[n] for the following cases:

- (a)  $\omega_0 = 2\pi 100 \text{ radians/s.}$  and T = 0.002 s.
- (b)  $\omega_0 = 2\pi 400 \text{ radians/s.}$  and T = 0.002 s.
- (c)  $\omega_0 = 2\pi 400 \text{ radians/s.}$  and T = 0.0005 s.
- (d)  $\omega_0 = 2\pi 1600 \text{ radians/s.}$  and T = 0.0005 s.

Whenever you have a normalized frequency outside the range  $[-\pi, \pi]$ , show that your signal will not change if you change the normalized frequency to a number between  $-\pi$  and  $\pi$  obtained by subtracting an appropriate integer multiple of  $2\pi$  from the original result. If two of the answers are the identical among the four cases, convince yourself that the answers are correct by verifying that the signals x[n] in both cases are also the same, and then explain why this is so.

2. Consider a discrete-time linear time-invariant system with unit impulse response function

$$h[n] = 3(0.5^n u[n]) + 2(0.9^n u[n])$$

- (a) Find the frequency response of this filter.
- (b) Using the above result, find the output of the system when its input is

$$x[n] = \cos(0.4\pi n) + \sin(0.3\pi)$$

simplify as much as possible. You may use a calculator if necessary.

- 3. Graph the magnitude response of the system in Problem 2. You may use Matlab or another software for this. However, please do not use canned routines to find the frequency response. Sketch the magnitude response for both negative and positive frequencies to verify that the magnitude response is an even function of frequency. You may need to experiment a little to determine the range of frequencies where the magnitude response is negligible. Is the frequency response periodic in these cases? Why or why not?
- 4. Consider a unit impulse response signal given by

$$h[n] = 0.9^n \cos\left(\frac{\pi}{6}n\right) u[n]$$

Find the frequency response  $H(e^{j\omega})$  of this system. Is the result periodic in frequency? Why or why not? Sketch the magnitude response of this system. (You may use Matlab or another computer program to do this.) Is the magnitude response an even function of frequency? Why or why not?

5. Two discrete-time, linear, time-invariant systems connected in series (cascade) have unit impulse responses given by  $h_1[n] = 0.9^n u[n] - 0.5(0.9)^{n-1} u[n-1]$  and  $h_2[n] = 0.5^n u[n] - 0.9(0.5)^{n-1} u[n-1]$ , respectively. Show that the series connection of the system produces an identity system.

*Note*: This problem appeared in Chapter 3 also. This time, show that the frequency response of the overall system created by the series connection of the two systems is 1 for all frequencies.