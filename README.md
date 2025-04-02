Code for Question 1 and Question 2 are in the file: [a1.ipynb](https://github.com/macs30113-s24/a1-Jessieliao2001/blob/main/a1.ipynb)

## Q1
First, let’s document the degree to which we can speed up the compute time of a program as we (pre-)compile computationally intensive portions of the code using numba.

### *a. (3 points) Rewrite this computationally intensive portion of the code as a separate function (returning $z_{mat}$) and compile it using the numba @jit decorator.*

```python
# Original nested loop function
def loop_original(eps_mat, z_mat, rho, mu, z_0, T, S):
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
    
    return z_mat

# JIT accelerated function
@jit(nopython=True)
def loop_jit(eps_mat, z_mat, rho, mu, z_0, T, S):
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
    return z_mat
```

### *Incorporate the function into the program above and compare how long it takes to run the original version of the code (as it is written above) with the time it takes to run your @jit-accelerated version. Report the speedup you observe.*

```python
# Set model parameters
rho = 0.5
mu = 3
sigma = 1
z_0 = mu

# Set simulation parameters
S = 1000
T = 4160
np.random.seed(25)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
z_mat = np.zeros((T, S))

# Time the original function
%time ori_results = loop_original(eps_mat, z_mat, rho, mu, z_0, T, S)
# Time the JIT-accelerated function
%time jit_results = loop_jit(eps_mat, z_mat, rho, mu, z_0, T, S)
```

 According to the return of the time function, the wall time reduced from 842ms to 128ms after utilizing the jit acceleration. 

```python
original_time = 842  # Original execution time in milliseconds
optimized_time = 128  # Optimized execution time in milliseconds
speedup = original_time / optimized_time  # Calculate speedup
print(f"Speedup: {speedup:.2f}x")
```
The Speedup is 6.58x.

### *b. (3 points) Now, pre-compile a version of your function ahead of time using numba. Incorporate this pre-compiled code into the program above and compare how long it takes to run the original version of the code (as it is written above) with the time it takes to run your pre-compiled version. Report the speedup you observe.*

*One hint:
Check the data types for each of your variables and make sure you are specifying them correctly in your numba signature. For instance, as defined in the code above, z_mat and eps_mat are 2d arrays that contain 64-bit (8-byte) floats and can thus be represented as f8[:,:] in the signature. Consult the numba documentation on types and signaturesLinks to an external site. for more detail on this notation.*

```python
from numba.pycc import CC

# name of compiled module to create:
cc = CC('test_aot')

# Correct the function signature if necessary (using 64-bit floats for all except indices):
@cc.export('loop_aot', 'f8[:,:](f8[:,:], f8[:,:], f8, f8, f8, i8, i8)')
def loop_aot(eps_mat, z_mat, rho, mu, z_0, T, S):
    for s_ind in range(S):  # Loop over each simulation
        z_tm1 = z_0  # Set initial value for each simulation
        for t_ind in range(T):  # Loop over each time period
            e_t = eps_mat[t_ind, s_ind]  # Random shock
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t  # Update equation
            z_mat[t_ind, s_ind] = z_t  # Store the result
            z_tm1 = z_t  # Prepare for next period
    return z_mat

cc.compile()
```
```python
import test_aot
%time aot_result = test_aot.loop_aot(eps_mat, z_mat, rho, mu, z_0, T, S)
```

The wall time is 21.6ms.

```python
precompiled_time = 21.6  
speedup_precompile = original_time / precompiled_time  # Calculate speedup
print(f"Speedup: {speedup_precompile:.2f}x")
```
The speedup is 38.98x.

### *c. (1 point) How does the pre-compiled code speedup compare to the @jit speedup? With this particular simulation application in mind, what contexts might it make sense to precompile this code ahead of time as opposed to using @jit?*

**Speedup comparison**:
* 6.58 times with @jit
* 38.98 times with precompiled code - aot.

From these results, we can see that the precompiled code has a much higher speedup than the JIT version. This indicates that for this particular simulation application, precompilation provides a more significant performance boost.

**Applicable scenarios**:

* Pre-compile (AOT): better suited for production environments or re-running scenarios. If you have finalized your code logic and need to run the same code multiple times on several different environments or platforms, precompilation can significantly reduce startup time and increase runtime efficiency. Additionally, if you are in an environment that does not allow on-the-fly compilation (such as certain servers or security-restricted environments), precompilation is a good option.

* In this particular simulation application, it may make more sense to use pre-compiled code if you need to run this simulation over and over again frequently, or if you need to deploy the code to a strict production environment. 

* A particular example is that precompilation can ensure consistent performance since the compilation step has already been completed. With JIT, the first run might include a performance penalty due to compilation time.

## Q2

### *(3 points) Let’s imagine that this code is still not fast enough and we wish to speed the simulation up further via parallelization. In words, **describe the portions of the above code that are potentially parallelizable**. Then, **calculate the overall theoretical speedup you might expect by parallelizing the code in these spots** (be sure to consider **both Amdahl and Gustafson’s Laws**!). Based on your calculations, **do you expect a linear speedup as you increase parallelism** (e.g. from 1 process to 10 processes to 100 processes)? **Explain your reasoning**. Note: Assume that the generation of eps_mat via sts.norm.rvs() cannot be parallelized and must occur on a single process.*

**Ans**：

1. the portions of the above code that are potentially parallelizable

   * Since each life's simulation is independent and does not require sequential execution, this part is suitable for parallel processing.
    ```python
    start = time.perf_counter()
    # Set model parameters
    rho = 0.5
    mu = 3.0
    sigma = 1.0
    z_0 = mu

    # Set simulation parameters, draw all idiosyncratic random shocks,
    # and create empty containers
    S = 1000  # Set the number of lives to simulate
    T = 4160  # Set the number of periods for each simulation
    np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
    z_mat = np.zeros((T, S))
    end = time.perf_counter()
    interval = end - start
    interval
    ```
    `0.13084141699800966`

    ```python
    S = 0.13/(0.13+0.84)
    P = 1-S
    P
    ```
    `0.865979381443299`

2. Theoretical Speedup Calculation:

   a. **Amdahl's Law**: assume the parallelizable portion is P, then the theoretical maximum speedup is:

   $$Speedup = \frac{1}{S + \frac{P}{N}}$$

   ```python
   def amdahls_law(P, N):
       S = 1 - P
       return 1 / (S + (P / N))
    ```


   b. **Gustafson’s Law** suggests that with scaled problem sizes, the speedup can be expressed as:

   $$Speedup' = N - S \times (N-1)$$
    ```python
    def gustafsons_law(S, N):
        return N - S * (N - 1)
   ```

3. Expectation of Linear Speedup:
    ```python
    P = 0.866
    S = 1 - P

    amdahl_speedup_1 = amdahls(P, 1)
    amdahl_speedup_10 = amdahls(P, 10)
    amdahl_speedup_100 = amdahls(P, 100)

    gustafson_speedup_1 = gustafsons(S, 1)
    gustafson_speedup_10 = gustafsons(S, 10)
    gustafson_speedup_100 = gustafsons(S, 100)

    print("Amdahl's Law, Speedup N = 1:", amdahl_speedup_1)
    print("Amdahl's Law, Speedup N = 10:", amdahl_speedup_10)
    print("Amdahl's Law, Speedup N = 100:", amdahl_speedup_100)

    print("Gustafson's Law, Speedup N = 1:", gustafson_speedup_1)
    print("Gustafson's Law, Speedup N = 10:", gustafson_speedup_10)
    print("Gustafson's Law, Speedup N = 100:", gustafson_speedup_100)
    ```

   a. **According to Amdahl's Law**, increasing the number of parallel processes will not increase the speed indefinitely because there will always be a portion that needs to be executed serially (for example, the generation of `eps_mat`). Therefore, even if you increase from 1 processor to 10 or 100 processors, the speedup will not increase linearly and will eventually encounter a limit.

   b. **According to Gustafson’s Law**, if you can increase the size of the problem (e.g., simulate more lives), then you may see a more linear growth in speedup as the number of processors increases. However, in practice, this depends on whether the additional workload can be efficiently distributed and whether the communication overhead between processors does not consume too much time.

    ```
    Amdahl's Law, Speedup N = 1: 1.0
    Amdahl's Law, Speedup N = 10: 4.533091568449683
    Amdahl's Law, Speedup N = 100: 7.009673349221925
    Gustafson's Law, Speedup N = 1: 1.0
    Gustafson's Law, Speedup N = 10: 8.794
    Gustafson's Law, Speedup N = 100: 86.734
    ```