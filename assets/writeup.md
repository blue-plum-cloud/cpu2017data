## Introduction

This document aims to investigate SPEC CPU2017 speed and rate results and attempt to relate those results to hardware specifications. The goal is to identify key characteristics that influence a system's single-thread performance versus its overall throughput.



## Data Processing
The analysis was conducted by sourcing publicly available benchmark data at https://www.spec.org/cpu2017/.

### Filters
The following filters were used to preprocess the data.
1. Processor Generations: Only the newest server processors were considered
    - AMD EPYC (1st - 5th Generation)
    - Intel Xeon Scalable (1st - 6th Generation)
2. Number of chips: 1-8
3. Systems with Base Result > 0. As many Peak Results had a score of 0, the scores were not considered.

### Calculating Scaling and Efficiency
```
scaling = rate_score / speed_score

efficiency = rate_score / (speed_score * num_cores)
```
### Calculating System Memory Bandwidth
Memory is calculated with the following formula:
```
bandwidth = number_of_memory_modules * data_rate (MT/s) / bus_width (8 bits)
```

### Calculating System Socket Bandwidth
System Socket Bandwidth is calculated by finding single socket bandwidth and multiplying it by the number of chips in the system.
#### AMD
AMD utilizes Infinity Fabric is built on top of the PCIe lanes for socket-to-socket communication. AMD EPYC only has dual-socket systems.
1. 1st Gen (Naples)
    - PCIe 3.0 x16, 3 links
2. 2nd Gen (Rome)
    - PCIe 4.0 x16, 4 links
3. 3rd Gen (Milan)
    - PCIe 4.0 x16, 4 links
4. 4th Gen (Siena/Genoa)
    - PCIe 5.0 x16, 3/4 links
5. 5th Gen (Turin)
    - PCIe 5.0 x16, 4 links
#### Intel
Intel Xeon Scalable processors use the [Ultra Path Interconnect (UPI)](https://en.wikipedia.org/wiki/Intel_Ultra_Path_Interconnect) for socket-to-socket communication.
Each UPI link contains 20 lanes per direction, and transfer speeds differ by generation. Scalability can range from 2 to 8 socket systems.
1. 1st Gen (Skylake)
    - 10.4GT/s, 2-3 UPI links per processor.
2. 2nd Gen (Cascade Lake)
    - 10.4GT/s, 2-3 UPI links per processor.
3. 3rd Gen (Cooper Lake)
    - 10.4GT/s, 6 UPI links per processor. This enables 2 links per connection in larger topologies. But for dual-socket systems, only up to 3 UPI links can be used.
4. 4th Gen (Sapphire Rapids)
    - 16GT/s, 2-3 UPI links per processor.
5. 5th Gen (Emerald Rapids)
    - 20GT/s, 2-3 UPI links per processor. This generation only scales up to dual-socket systems.
6. 6th Gen (Granite Rapids)
    - 24GT/s, up to 6 UPI links per processor. Can connect up to 6 UPI links for dual-socket systems, depending on processor model.