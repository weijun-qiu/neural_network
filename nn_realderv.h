/* nn.h - a naive implementation of neural network */

#define RATE 0.00000008        // Learning rate
#define LAYER_NUM 2           // number of hidden layers
#define NEURON_PER_LAYER 5   // number of neurons per layer
#define DOMAIN_MIN 0          // domain lower bound
#define DOMAIN_MAX 50         // domain upper bound

typedef struct neuron{
  double value;
  struct neuron *inputs[NEURON_PER_LAYER];
  double weights[NEURON_PER_LAYER];
  double delta;
} Neuron;

double input;
Neuron output;
Neuron hlayers[LAYER_NUM][NEURON_PER_LAYER];
int    trset[100]; // trainning set

double feed_forward(void);
void   train(void);
double f(int);
void   initialize();
double derv(double *);
double sigmoid(double);
void   printw(void);
void   printe(void);
