`#pragma omp parallel for` - Basic parallelization of the outer loop. 1.33

`#pragma omp parallel for schedule(static)` - Splits iterations evenly ahead of time; often good for image rows and cache locality.

`#pragma omp parallel for schedule(static, 8)` - Static scheduling with chunk size 8; each thread gets blocks of 8 iterations.

`#pragma omp parallel for schedule(static, 32)` - Same as above, but larger chunks; can reduce scheduling overhead.

`#pragma omp parallel for schedule(dynamic)` - Threads grab work as they finish; useful if work per iteration is uneven.

`#pragma omp parallel for schedule(dynamic, 8)` - Dynamic scheduling with chunk size 8; balances load with less overhead than fully dynamic.

`#pragma omp parallel for schedule(guided)` - Starts with large chunks, then smaller ones; often a middle ground between static and dynamic.

`#pragma omp parallel for collapse(2)` - Merges two nested loops into one iteration space for more parallelism.

`#pragma omp parallel for collapse(2) schedule(static)` - Collapses both loops, then distributes work statically.

`#pragma omp parallel for collapse(2) schedule(dynamic, 32` - Collapses both loops and distributes combined iterations dynamically in chunks of 32.

Best first tests for image processing are usually schedule(static), schedule(static, 32), and collapse(2) schedule(static).