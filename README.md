# quantum_stuff

Quantum Image Classifier for Fashion-MNIST

This project is a proof-of-concept for image classification using a hybrid quantum-classical model built with Qiskit. It classifies clothing items from the Fashion-MNIST dataset.

Summary of Work

    Quantum Encoder: Developed a circuit to encode Fashion-MNIST images into quantum states using Amplitude Encoding.

    Variational Quantum Classifier (VQC): Designed a trainable, parameterized quantum circuit to act as the classifier.

    Hybrid Training: Implemented a complete training pipeline that uses a classical optimizer (COBYLA) to teach the quantum circuit's parameters.

    Optimization & Debugging: Iteratively improved the model by resolving library-specific errors and scaling up the training process to achieve a better final score.
