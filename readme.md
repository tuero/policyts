# policyts

A C++23 wrapper project containing training and testing of various tree search algorithms which utilize learned policies and heuristics.
Algorithm implementations are in [libpolicyts](https://github.com/tuero/libpolicyts) which we pull in as a dependency,
and parameterize using the implemented neural policy/heuristics and environments.

## Supported Algorithms
- __Best First Search__: A general search algorithm with controlled weights on the g-cost and h-cost
- __LevinTS__: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information Processing Systems 31 (2018).
- __PHS*__: Orseau, Laurent, and Levi HS Lelis. "Policy-guided heuristic search with guarantees." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 14. 2021.

## Supported Environments
- [BoulderDash](https://github.com/tuero/boulderdash_cpp)
- [CraftWorld](https://github.com/tuero/craftworld_cpp_v2)
- [Sokoban](https://github.com/tuero/sokoban_cpp)
- [TSP Gridworld](https://github.com/tuero/tsp_cpp)


## Building
This project utilizes C++23 features, so you need to ensure your compiler is supported. 
The compiler used to build and test this project is`g++-15.2`.

All dependencies are managed through [vcpkg](https://vcpkg.io/en/), except for `libtorch` (pytorch's C++ frontend). 
The easiest way to get `libtorch` is through the python package.
First, create a virtual environment and install pytorch:
```shell
conda create -n policyts python=3.12
conda activate policyts
pip3 install torch torchvision
```

Next, the following environment variables are required for the toolchain packages to be found:
- `CC`: The path to your C compiler
- `CXX`: The path to your C++23 compliant compiler
- `FC`: The path to your Fortran compiler
- `LIBTORCH_ROOT`: The path to the libtorch package, which we will point towards the just installed python package

For example:
```shell
export CC=gcc-15.2
export CXX=g++-15.2
export FC=gfortrain-15.2
# Ensure you activated the policyts virtual environment
export LIBTORCH_ROOT=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

Finally, we use `CMakePresets.json` which sets all the required CMake variables.
```shell
cmake --preset=release
cmake --build --preset=release -- -j8
```

## Usage
For an example of how to train a policy network for PHS*:
```shell
./build/release/src/phs/phs_train --environment=boulderdash --problems_path=problems/bd_train.txt --output_dir=experiments/phs/bd_s0 --model_path=models/twoheaded_convnet.json --search_budget=4000 --inference_batch_size=32 --mix_epsilon=0.01 --seed=0 --num_train=10000 --num_validate=1000 --num_threads=8 --learning_batch_size=512 --device_num=0 --validation_solved_ratio=0.95 --time_budget=1000000 
```
Then, to test the trained policy:
```shell
./build/release/src/phs/phs_test --environment=boulderdash --problems_path=problems/bd_test.txt --output_dir=experiments/phs/bd_s0 --model_path=models/twoheaded_convnet.json --search_budget=4000 --inference_batch_size=1 --mix_epsilon=0.01 --num_threads=8 --device_num=0
```
