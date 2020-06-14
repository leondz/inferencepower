# Inference-time activation power consumption

The power that machine learning models consume when making predictions can be affected by a model's architecture. This repository accompanies a paper that presents various estimates of power consumption for a range of different activation functions, a core factor in neural network model architecture design. Substantial differences in hardware performance exist between activation functions. This difference informs how power consumption in machine learning models can be reduced.

For a dummy workload:

![power per instance per activation function](https://github.com/leondz/inferencepower/raw/master/dc_cpu.svg)

The performance spread across the group of activation functions (and also across the group of dropout functions) exists across platforms and across loads. The specific high-variation spots depend on the scale and complexity of the workload; our workload is simple, and other workloads may experience maximum variation at other relative scales. In our case, GPUs can experience up to an order of magnitude in difference between execution performance of the fastest- and the slowest-running activation functions. This impacts power consumption (and emissions).

![Spread in computation required per activation function](https://github.com/leondz/inferencepower/raw/master/groupspread.svg)

For full details, see the paper: "[Power Consumption Variation over Activation Functions](https://www.derczynski.com/papers/Activation_Function_Power_Consumption.pdf)", Derczynski, L. (2020)


## Files

* `rand_test.py` -- The workload. First execution will generate test models and store in a cache dir.
* `power_table.py` -- For reporting the power used per function over `rand_test.py` results.
* `spread_graph.py` -- For showing the extent of spread within a group of functions.
* `mnist_power.py`, `mnist_graph.py` -- For the MNIST training experiment and graph.

## License

CC-BY 4.0 Leon Derczynski, ld@itu.dk 2020

## Reference

Derczynski, Leon. "Power Consumption Variation over Activation Functions" (2020). arXiv. [pdf](https://www.derczynski.com/papers/Activation_Function_Power_Consumption.pdf)

```
@article{activationpower,
  title={{Power Consumption Variation over Activation Functions}},
  author={Leon Derczynski},
  year={2020},
  journal={arXiv}
}
```
