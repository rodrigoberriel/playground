## About the cs231n

The scripts are most likely related to the material available [here](http://cs231n.github.io/).

#### Optimization 2

In [Backpropagation, Intuitions](http://cs231n.github.io/optimization-2/), 3 circuits are shown and the gradients are calculated. In addition, they provided code written in Python using NumPy to perform the computations. Here, code written in Python using TensorFlow is provided.

##### Example 1

Command:

    python backprop-example-1.py

Output:

    Result (f): -12
	Gradients:
	- variable: x	grad: -4	value: -2
	- variable: y	grad: -4	value: +5
	- variable: z	grad: +3	value: -4

Graph:

![Backprop Example 1](https://github.com/rodrigoberriel/playground/blob/master/cs231n/images/backprop-example-1.png)

##### Example 2

Command:

    python backprop-example-2.py

Output:

	Result (f): +0.73
	Gradients:
	- variable: w0	grad: -0.20	value: +2.00
	- variable: x0	grad: +0.39	value: -1.00
	- variable: w1	grad: -0.39	value: -3.00
	- variable: x1	grad: -0.59	value: -2.00
	- variable: w2	grad: +0.20	value: -3.00

Graph:

![Backprop Example 2](https://github.com/rodrigoberriel/playground/blob/master/cs231n/images/backprop-example-2.png)

##### Example 3

Command:

    python backprop-example-3.py

Output:

	Result (f): -20.00
	Gradients:
	- variable: x	grad: -8.00	value: +3.00
	- variable: y	grad: +6.00	value: -4.00
	- variable: z	grad: +2.00	value: +2.00
	- variable: w	grad: +0.00	value: -1.00

Graph:

![Backprop Example 3](https://github.com/rodrigoberriel/playground/blob/master/cs231n/images/backprop-example-3.png)