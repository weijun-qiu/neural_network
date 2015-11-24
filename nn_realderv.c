/* nn_realderv.c - a naive implementation of neural network using
   real mathmetical differentiaion algorithm */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "nn.h"

int main(int argc, char *argv[]){
  initialize();
  train();
  return 0;
}


/* initialize all data struct need for the neural network */

void initialize(void) {
  srand(time(NULL));

  // initialize trainning set
  
  int i, j, k;
  printf("test set initialized to: { ");
  for (i = 0; i < sizeof trset / sizeof trset[0]; i++) {
    trset[i] = rand() % DOMAIN_MAX;
    printf("%i ", trset[i]);
  }
  printf("}\n");

  // initialize neurons

  for (i = 0; i < LAYER_NUM; i++) {
    for (j = 0; j < NEURON_PER_LAYER; j++) {
      if (i == 0) {
        hlayers[0][j].weights[0] = (double)(rand() % 2000) / 1000 - 1;
      }else {
        for (k = 0; k < NEURON_PER_LAYER; k++) {
          hlayers[i][j].weights[k] = (double)(rand() % 2000) / 1000 - 1;
          hlayers[i][j].inputs[k] = &hlayers[i - 1][k];
        }
      }
    }
  }

  for (i = 0; i < NEURON_PER_LAYER; i++) {
    output.inputs[i] = &hlayers[LAYER_NUM - 1][i];
    output.weights[i] = (double)(rand() % 2000) / 1000 - 1;
  }
  // printw();
}


/* train the neural network using the trainning set */

void train(void) {
  int i, j, k, l;
  for (i = 0; i < sizeof trset / sizeof trset[0]; i++) {
    printf("\nTrainning Round %d\n", i);

    // feed an input into the network
    
    input = trset[i];
    feed_forward();
    printe();

    // back-propagate

    for (j = 0; j < LAYER_NUM; j++) {
      for (k = 0; k < NEURON_PER_LAYER; k++) {
        hlayers[j][k].delta = 0;
      }
    }

    output.delta = output.value - f(input);
    for (j = 0; j < NEURON_PER_LAYER; j++) {
      output.inputs[j]->delta = output.delta *
        output.weights[i];
    }
    for (j = LAYER_NUM - 1; j > 0; j--) {
      for (k = 0; k < NEURON_PER_LAYER; k++) {
        if (j == 0) {
          hlayers[0][k].inputs[0]->delta += hlayers[0][k].delta *
            hlayers[0][k].weights[0];
        }else {
          for (l = 0; l < NEURON_PER_LAYER; l++) {
            hlayers[j][k].inputs[l]->delta += hlayers[j][k].delta *
              hlayers[j][k].weights[l];
          }
        }
      }
    }

    Neuron cpy[LAYER_NUM][NEURON_PER_LAYER];
    memcpy(cpy, hlayers, sizeof cpy);

    for (j = 0; j < NEURON_PER_LAYER; j++) {
      output.weights[j] -= output.delta * output.inputs[j]->value * RATE;
    }
    for (j = 0; j < LAYER_NUM; j++) {
      for (k = 0; k < NEURON_PER_LAYER; k++) {
        if (j == 0) {
          cpy[0][k].weights[0] = hlayers[0][k].weights[0] -
            hlayers[0][k].delta * input * RATE;
          // derv(&hlayers[0][k].weights[0]) * RATE;
        }else {
          for (l = 0; l < NEURON_PER_LAYER; l++) {
            cpy[j][k].weights[l] = hlayers[j][k].weights[l] -
              hlayers[j][k].delta * hlayers[j][k].inputs[l]->value * RATE;
              //derv(&hlayers[j][k].weights[l]) * RATE;
          }
        }
      }
    }

    Neuron outcpy = output;

    for (j = 0; j < NEURON_PER_LAYER; j++) {
      outcpy.weights[j] = output.weights[j] -
        derv(&output.weights[j]) * RATE;
    }
    
    memcpy(hlayers, cpy, sizeof hlayers); // update weights
    output = outcpy;

    // printw();
  }
}


/* feed forward stage of neural network */

double feed_forward(void) {
  int i, j, k;
  double ret = 0;
  for (i = 0; i < LAYER_NUM; i++) {
    for (j = 0; j < NEURON_PER_LAYER; j++) {
      if (i == 0) {
        hlayers[0][j].value = hlayers[0][j].weights[0] * input;
      }else {
        hlayers[i][j].value = 0;
        for (k = 0; k < NEURON_PER_LAYER; k++) {
          hlayers[i][j].value += hlayers[i][j].weights[k] *
            hlayers[i][j].inputs[k]->value;
        }
        // can add a sigmoid here
      }
    }
  }
  output.value = 0;
  for (i = 0; i < NEURON_PER_LAYER; i++) {
    output.value += output.inputs[i]->value *
      output.weights[i];
  }

  // can add a sigmoid here
  return output.value;
}


/* calculate dE/dw */

double derv(double *w) {
  *w += 0.0000001;
  double x = pow(feed_forward() - f(input), 2); // E(x+0.00001)
  *w -= 0.000002;
  double y = pow(feed_forward() - f(input), 2); // E(x-0.00001)
  *w += 0.0000001;
  double diff = x - y;
  return diff / 0.0000001;
}


/* an arbitrary polynomial function */

double f(int x) {
  return - x * x + 9 * x - 193;
}


/* the sigmoid function */

double sigmoid(double x) {
  return 100/(1 + exp(-0.05 * (x - 28))) + 3;
}


/* print weights */

void printw() {
  int i, j, k;
  for (i = 0; i < LAYER_NUM; i++) {
    for (j = 0; j < NEURON_PER_LAYER; j++) {
      printf("Neuron [%d, %d] weights:", i, j);
      if (i == 0) {
        printf(" %f", hlayers[0][j].weights[0]);
      }else {
        for (k = 0; k < NEURON_PER_LAYER; k++) {
          printf(" %f", hlayers[i][j].weights[k]);
        }
      }
      printf("\n");
    }
  }
}


/* print error */

void printe() {
  printf("Input = %f\nTarget = %f\nActual = %f\n", input,
         f(input), output.value);
  printf("Error = %f\n", f(input) - output.value);
}
