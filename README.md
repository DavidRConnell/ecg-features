**ECG processing**

Provides standard linear time-domain, linear frequency-domain, and non-linear ECG processing functions (Support vector machine-based arrhythmia classification using reduced features of heart rate variability signal 2.2.2 https://pdfs.semanticscholar.org/0c5d/2c9a7540dd3ee6f708e3671d8c9352c2ff8b.pd ). All features are based on the R-R intervals.

The functions are designed to work with records from physionet.org. Records from physionet.org can be downloaded and saved using retrieve_physio_files.py.

*Linear time-domain functions*

  * Mean.
  * Root mean square successive difference (RMSSD).
  * Standard deviation between normal-normal (R-R) intervals (SDNN).
  * Standard deviation between successive differences (SDSD).
  * Probability successive normal-normal (R-R) intervals differ by greater than t (standard t = 50, 10, or 5) (pNNt).

*Linear frequency-domain functions*

  * Ratio of the low-frequency (LF) and high-frequency (HF) bands (LF_over_HF).

*Non-linear*

  * Cardiac-Sympathetic index (CSI).
  * Approximate entropy (ApEn).
  * Spectral entropy (SpEn).
  * Lyapunov exponent (LLE).
  * Detrended fluctuation analysis (DFA).
  * Sequential trend analysis (STA).

