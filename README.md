### TwoEyesModelNeuralNetwork
Simple Neural Network with a simulation of 2 eyes and simple feature attention

### Input Context
Each Character of the context size text is represented with a square signal with different frequency and a shift in time to represent the position of the character in the context text.


### The Model Architecture

The model contains 2 parallel layers named eyes and one learnable attention layer and a layer to combines features of the eyes with attention applied then pass the result and the input to a final classificaion layer.


### Some generated Presentation after 100 training iterations:
The model generate coherent presentation that sample from the training set.

=====
My name is Moa, I'm 37 years old, and I'm an astronaut. My hobbies include meditation and skydiving. My favorite color is peam.
My name is Ethan, I'm 26 years old, and I work as a robotics engineer. In my spare time, I like to experiment with fusion cuisines. My favorite color is forest green.
My name is William, I'm 48 years old, and I'm a fashion shows. My favorite color is pearl.My name is Lucas, I'm 48 years old, and I'm a lashist oluer.
My name is Ethan, I'm 37 years old, and I'm an ovent and my favorite color is poral.
My name is Evan, I'm 53 years old, and I'm a professional music. My favorite color is sky dark blue.
My name is Aan, I'm 22 years old, and I'm a fashion blogger. In my free time, I like to build model airplanes and my favorite color is lavender.
My name is Abacde, I'm 26 years old, and I'm an interior designer. I enjoy purfing megal vhice. My favorite color is eora, I'm 31 years old, and I'm a hidscuup arist. In my spare time, I like to design sustainable cities and playing squash. My favorite color is crimson red.
=====
