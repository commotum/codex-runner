1. **Recognizing a complex exponential signal** – read (z(t)=Ae^{j(\omega_0 t+\phi)}) as a signal with magnitude, angular frequency, and phase.

2. **Converting a complex exponential to rectangular form** – expand the signal into explicit real and imaginary pieces.

3. **Extracting the real and imaginary waveforms** – identify the cosine waveform as the real part and the sine waveform as the imaginary part.

4. **Interpreting a complex exponential in the complex plane** – view the signal as a complex quantity with magnitude and angle that change meaningfully over time.

5. **Reading phasors at (t=0)** – interpret (z(0)=Ae^{j\phi}) as the initial phasor with starting angle (\phi).

6. **Interpreting angular frequency as rotation rate** – connect (\omega_0) to how fast the phasor turns in the complex plane.

7. **Real-axis projection of a phasor** – understand the cosine component as the projection of the rotating phasor onto the real axis.

8. **Imaginary-axis projection of a phasor** – understand the sine component as the projection of the rotating phasor onto the imaginary axis.

9. **Reading projection values at special angles** – determine when the real or imaginary component becomes zero or reaches an extreme from the total angle.

10. **Connecting phasor motion to time-domain sinusoids** – understand why a rotating phasor traces sine and cosine waves over time.

11. **Using complex exponentials as compact sinusoid representations** – treat exponential form as a cleaner representation for analysis and later signal-processing work.


NEEDS CLEANING:

* **Reading sinusoid parameters** – identify amplitude, angular frequency, and phase in expressions like (A\cos(\omega t+\phi)) or (A\sin(\omega t+\phi)).
* **Radians and standard angles** – interpret angles like (\pi/2) and (\pi) as positions in a rotation.
* **Real and imaginary parts of a complex number** – distinguish the real component from the imaginary component.
* **Complex plane basics** – interpret a complex number as a point or vector on real and imaginary axes.
* **Magnitude and angle of a complex quantity** – describe a complex number by its length and direction.
* **Using Euler’s formula** – rewrite exponential form as cosine plus (j) sine, and recognize the reverse connection.

* **Reading continuous-time signal notation** – understand expressions like (x(t)), (x(t+T)), and (x(t+nT)).
* **Basic sine and cosine fluency** – recognize sinusoidal expressions as repeating oscillatory signals.
* **Period–frequency conversion** – convert between period (T) and angular frequency (\omega).
* **Working with integer multiples** – reason about quantities like (nT) and (k\omega_0).
* **Comparing ratios** – determine whether a ratio is rational and what that implies.
* **Using the cosine product-to-sum identity** – rewrite a product of cosines as a sum of cosine terms.
* **Determining whether a continuous-time signal is periodic** – check whether a signal repeats after some positive time (T).
* **Finding the fundamental period** – identify the smallest repeating interval of a periodic signal.
* **Finding the period of a pulse train** – read the repeat interval of a repeating pulse waveform.
* **Reading the parameters of a sinusoid** – extract amplitude, angular frequency, and phase from (A\sin(\omega_0 t+\phi)).
* **Converting between sinusoid period and angular frequency** – use the relationship between (T) and (\omega_0) for a sinusoid.
* **Interpreting phase** – understand phase as controlling where the sinusoid starts and how it is shifted in time.
* **Identifying harmonics** – recognize first, second, third, and higher harmonics as integer multiples of a fundamental frequency.
* **Computing harmonic frequencies** – use (\omega_k = k\omega_0) to find a harmonic’s frequency.
* **Recognizing harmonic components in examples** – classify specific sinusoidal terms as fundamental or higher harmonics.
* **Finding a common period for two periodic signals** – determine whether two repeating signals line up on a shared repeat interval.
* **Testing whether a sum of periodic signals is periodic** – use the rational-ratio condition on periods to decide whether a linear combination repeats.
* **Distinguishing linear and nonlinear operations on signals** – classify addition as linear and multiplication as nonlinear.
* **Expanding the product of two sinusoids** – use the product-to-sum identity to reveal sum- and difference-frequency components.
* **Analyzing the equal-frequency product case** – identify what happens when two same-frequency sinusoids are multiplied.
* **Identifying a DC component** – recognize a constant term as a zero-frequency part of a signal.
* **Recognizing harmonic generation from nonlinear operations** – understand that multiplication can create new frequencies not present in the original inputs.

1. **Use Euler’s formula** — Move between (e^{j\theta}) and (\cos(\theta)+j\sin(\theta)), so trig expressions and complex exponentials can be treated as equivalent views of the same object.

2. **Identify real and imaginary parts of a complex exponential** — Read the cosine part as the real part and the sine part as the imaginary part of (Ae^{j(\omega t+\phi)}).

3. **Interpret polar form of a complex number** — Understand (Ae^{j\phi}) as a magnitude-angle representation rather than just a symbolic expression.

4. **Convert a complex number between rectangular and polar form** — Rewrite a number like (a+jb) as (|V|e^{j\angle V}), using magnitude and angle.

5. **Add complex numbers in rectangular form** — Combine phasors by adding their real parts and imaginary parts correctly.

6. **Use special-angle values in the complex plane** — Recognize standard values such as (e^{j\pi/2}=j) and common reference angles that appear in phasor calculations.

7. **Rewrite a real cosine as the real part of a complex exponential** — Express (A\cos(\omega t+\phi)) as (\Re{Ae^{j(\omega t+\phi)}}).

8. **Factor a sinusoid into phasor form** — Separate (Ae^{j(\omega t+\phi)}) into a fixed phasor (Ae^{j\phi}) times the rotating term (e^{j\omega t}).

9. **Define and interpret a phasor** — Treat (V=Ae^{j\phi}) as the stationary complex quantity carrying amplitude and phase information.

10. **Relate the sign of frequency to direction of rotation** — Determine whether a complex exponential rotates clockwise or counterclockwise from the sign of (\omega).

11. **Rewrite sine and cosine using complex exponentials** — Use Euler-based identities to convert trigonometric functions into sums or differences of exponentials.

12. **Interpret a cosine as positive- and negative-frequency components** — Recognize that a real cosine is built from two counter-rotating exponentials, one with (+\omega) and one with (-\omega).

13. **Identify positive and negative frequency geometrically** — Understand frequency sign not just algebraically, but as a direction of motion in the complex plane.

14. **Add same-frequency cosines using phasors** — Convert each sinusoid to a phasor, add the phasors directly, and preserve the shared (e^{j\omega t}) factor.

15. **Form and interpret the resultant phasor** — Combine multiple same-frequency terms into one equivalent phasor that captures the net amplitude and phase.

16. **Convert a resultant phasor back to a real sinusoid** — Take the summed phasor in polar form and rewrite it as a single cosine in amplitude-phase form.

17. **Solve an end-to-end phasor combination problem** — Execute the full workflow: convert real sinusoids to phasors, do the complex-domain algebra, convert back to a single real sinusoid.

* **Complex-number form** — read a complex quantity as real part plus imaginary part.
* **Exponent laws for exponentials** — split and regroup expressions like (e^{a+b}) and (e^{at+bt}).
* **Euler’s formula** — convert between (e^{j\theta}) and (\cos \theta + j\sin \theta).
* **Real and imaginary parts** — extract (\Re{\cdot}) and (\Im{\cdot}) from a complex expression.
* **Basic sinusoidal interpretation** — understand cosine/sine as oscillations and (\omega) as angular frequency.
* **Real exponential growth and decay** — tell from the sign of an exponent whether a signal grows, decays, or stays constant.
* **Signal periodicity definition** — use (x(t+T)=x(t)) as the condition for repetition.

* **Define a generalized exponential signal** — interpret (x(t)=e^{st}) with (s=\sigma+j\omega).
* **Interpret the roles of (\sigma) and (\omega)** — know that (\sigma) controls growth/decay and (\omega) controls oscillation.
* **Expand a generalized exponential** — rewrite (e^{(\sigma+j\omega)t}) as (e^{\sigma t}e^{j\omega t}).
* **Convert a generalized exponential into cosine-sine form** — use Euler’s formula to express the signal in oscillatory components.
* **Extract the real and imaginary components of the signal** — identify the cosine-based real part and sine-based imaginary part.
* **Analyze the pure exponential case ((\omega=0))** — classify the signal as constant, growing, or decaying.
* **Analyze the pure oscillatory case ((\sigma=0))** — interpret (e^{j\omega t}) as a pure complex sinusoid.
* **Relate (\omega) to oscillation speed** — determine how changing (|\omega|) changes the rate of oscillation.
* **Analyze the mixed case ((\sigma \neq 0,\ \omega \neq 0))** — interpret simultaneous oscillation and amplitude growth/decay.
* **Identify damped versus growing sinusoids** — use the sign of (\sigma) to tell whether the oscillation shrinks or expands.
* **Interpret exponential envelopes** — understand (\pm e^{\sigma t}) as amplitude bounds on the oscillation.
* **Connect the model to physical systems** — recognize generalized exponentials as models of damped oscillatory behavior.
* **Apply the periodicity test to (e^{j\omega t})** — derive the condition for when a pure complex exponential repeats.
* **Compute the fundamental period of a complex exponential** — find (T_0 = \frac{2\pi}{\omega}).
* **Interpret general repetition times** — recognize that any integer multiple of the fundamental period is also a repeat time.
* **Use the rotating-phasor picture** — explain periodicity geometrically using motion on the unit circle.


* **Read continuous-time signal notation** — understand that (x(t)) names a signal as a function of time, and that changing the input changes the signal’s time behavior.
* **Interpret piecewise-defined signals** — read a signal whose formula changes across different time intervals.
* **Identify a signal’s support** — determine the interval where a signal is nonzero or where a particular branch applies.
* **Substitute expressions into a function argument** — replace (t) with expressions like (t-1), (3t), or (-t) inside a signal formula.
* **Simplify transformed expressions** — rewrite substituted formulas cleanly, especially when exponentials are involved.
* **Solve interval inequalities** — transform time intervals correctly after substitution, including reversing inequalities when multiplying by (-1).
* **Track key time points on the time axis** — recognize important times like start points, endpoints, and feature locations, then see how they move under a transformation.

* **Recognize the three basic time-domain signal operations** — distinguish between shifting, scaling, and reversal as three different ways of transforming a signal.

* **Interpret time shifting** — understand that shifting moves a signal left or right without changing its basic shape.

* **Delay a signal** — apply (x(t-T)) and interpret it as moving the signal to the right by (T).

* **Advance a signal** — apply (x(t+T)) and interpret it as moving the signal to the left by (T).

* **Read shift direction from the sign inside the argument** — determine whether a form like (x(t-a)) represents a delay or an advance.

* **Rewrite a shifted piecewise signal** — substitute a shifted argument into each branch and update the time intervals accordingly.

* **Interpret time shifting physically** — connect shifting to different arrival times of the same waveform, such as signals received earlier or later.

* **Interpret time scaling** — understand that scaling changes how fast or slow a signal unfolds in time, rather than when it starts.

* **Compress a signal in time** — apply (x(at)) with (a>1) and interpret it as making the signal run faster.

* **Expand a signal in time** — apply (x(t/a)) with (a>1) and interpret it as making the signal run slower.

* **Track feature times under scaling** — map important time locations like (T_1) and (T_2) to their new positions after compression or expansion.

* **Rewrite a scaled piecewise signal** — solve for the new time intervals after replacing (t) by (at) or (t/a).

* **Distinguish shifting from scaling** — tell the difference between moving a signal in time and changing its time rate.

* **Interpret time scaling physically** — connect scaling to playback speed, such as playing a recording faster or slower.

* **Interpret time reversal** — understand that (x(-t)) mirrors a signal across the vertical axis (t=0).

* **Map signal support under reversal** — determine how the interval where the signal exists flips from one side of the time axis to the other.

* **Rewrite a time-reversed piecewise signal** — replace (t) with (-t), simplify the formula, and transform the time interval correctly.

* **Reverse temporal behavior conceptually** — describe how the order of events flips when time reversal is applied.

* **Interpret time reversal physically** — connect reversal to playing a signal backward.

* **Compare the effects of the three operations** — identify whether an operation changes a signal’s start time, time scale, or time direction.

* **Analyze transformed exponential signals in piecewise form** — apply shifting, scaling, and reversal specifically to exponential signals while preserving the correct branch conditions.

* **Continuous-time signal notation** – reading (x(t)) as a signal whose amplitude changes over time.
* **Piecewise-defined signals** – determining which formula describes a signal on each time interval.
* **Periodicity and fundamental period** – recognizing when a signal repeats and identifying the repeating interval.
* **Magnitude-squared of a signal** – forming (|x(t)|^2) as the quantity used in energy and power calculations.
* **Definite integration** – computing accumulated quantity over a finite interval.
* **Improper integration** – evaluating or interpreting integrals over infinite intervals.
* **Basic limit-based averaging** – understanding averages defined by taking a time window to infinity.
* **Integrating common signal expressions** – handling constants, polynomials, and decaying exponentials inside integrals.

* **Signal size as a scalar concept** – understanding that signal “size” should reflect both how large a signal is and how long it lasts.
* **Instantaneous power from signal amplitude** – seeing why pointwise power is naturally tied to the square of the signal’s amplitude.
* **Power–energy relationship** – interpreting power as the rate of change of energy and energy as the accumulation of power over time.
* **Signal energy** – computing total signal energy with (\int_{-\infty}^{\infty}|x(t)|^2,dt).
* **When energy is not finite** – recognizing when a signal does not decay enough for total energy to converge.
* **Average signal power** – computing long-run average power for signals whose total energy is infinite.
* **Periodic-signal power over one period** – replacing the long-time average with a one-period average when the signal is periodic.
* **Choosing the right size measure** – deciding whether energy or average power is the meaningful descriptor for a given signal.
* **Classifying energy signals** – identifying signals for which total energy is finite and nonzero.
* **Classifying power signals** – identifying signals for which average power is finite and nonzero.
* **Splitting piecewise energy calculations** – breaking an energy integral into interval-by-interval pieces and summing the results.
* **Computing energy of a finite-energy signal** – applying the energy formula to a signal with finite support and/or exponential decay.
* **Computing power of a periodic signal** – applying the one-period power formula to a repeating waveform such as a ramp or sawtooth.
* **Using examples to distinguish decay vs repetition** – recognizing that decaying signals tend to be energy signals, while repeating signals tend to be power signals.
