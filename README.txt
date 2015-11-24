Part of UBC CPSC311 Final Group Project

nn.c - This is a naive implementation of neural network in C, using simple
gradient regression method.

nn_realderv.c - This improves from nn.c. It uses a real methmatical
differentiation algorithm. Note it sometimes produces “nan” as results,
which stands for “not a number” in C and I think it is just related to the
absence of an activation function. For now, I don’t see any noticeable
difference in accuracy between the two version, but things may change
after we add a sigmoid, who knows...
