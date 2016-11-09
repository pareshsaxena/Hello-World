# Multiple Maps t-SNE
A C++ multi-threaded implementation of multiple maps t-SNE. This code implements the algorithm described in the paper: '[Visualizing Non-Metric Similarities in Multiple Maps](https://lvdmaaten.github.io/publications/papers/MachLearn_2012.pdf)' by Maaten & Hinton (2012). The gradient descent optimization has been modified for better convergence properties.

The source files have been tested on Windows 10 using MSVC++ 14.0 compiler and the same can be built using the NMAKE tool by typing the following at the cmd prompt:
```
  nmake -f makefile.win all clean
```
Make sure the environment variable is set for command-line builds. For more information refer: https://msdn.microsoft.com/en-us/library/f2ccy3wt.aspx


