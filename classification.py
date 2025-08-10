import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from skimage.transform import resize
import tensorflow as tf
from scipy.optimize import minimize

def encoder(image: np.ndarray) -> QuantumCircuit:
    image_resized = resize(image, (32, 32), anti_aliasing=True)
    pixel_vector = image_resized.flatten()
    norm = np.linalg.norm(pixel_vector)
    if norm == 0:
        normalized_vector = np.zeros(1024)
        normalized_vector[0] = 1.0
    else:
        normalized_vector = pixel_vector / norm
    num_qubits = 10
    qc = QuantumCircuit(num_qubits)
    qc.initialize(normalized_vector, range(num_qubits))
    return qc

def histogram_to_label(histogram: dict, num_classes: int = 10) -> int:
    valid_counts = {bitstring: count for bitstring, count in histogram.items() if int(bitstring, 2) < num_classes}
    if not valid_counts:
        return 0
    most_likely_bitstring = max(valid_counts, key=valid_counts.get)
    return int(most_likely_bitstring, 2)

def create_classifier_ansatz(num_qubits: int, num_layers: int) -> (QuantumCircuit, list):
    params = ParameterVector('θ', num_qubits * num_layers)
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        for i in range(num_qubits):
            qc.ry(params[layer * num_qubits + i], i)
        if num_qubits > 1:
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
        qc.barrier()
    return qc, params

def train_classifier(train_images, train_labels, num_samples, max_iter, num_layers):
    print("--- Starting Scaled-Up Quantum Classifier Training (COBYLA) ---")
    num_qubits = 10
    ansatz, params = create_classifier_ansatz(num_qubits, num_layers=num_layers)
    initial_params = np.random.uniform(0, 2 * np.pi, len(params))
    simulator = AerSimulator()
    X_train_small = train_images[:num_samples] / 255.0
    y_train_small = train_labels[:num_samples]
    iteration_count = 0

    def objective_function(current_params):
        nonlocal iteration_count
        iteration_count += 1
        total_cost = 0
        for image, label in zip(X_train_small, y_train_small):
            bound_ansatz = ansatz.assign_parameters(current_params)
            encoder_circuit = encoder(image)
            full_circuit = encoder_circuit.compose(bound_ansatz)
            full_circuit.measure_all()
            result = simulator.run(full_circuit, shots=1024).result()
            counts = result.get_counts()
            target_label_str = f'{label:0{num_qubits}b}'
            prob_correct = counts.get(target_label_str, 0) / 1024
            total_cost += (1 - prob_correct)
        avg_cost = total_cost / num_samples
        print(f"Iteration {iteration_count}/{max_iter} | Cost: {avg_cost:.4f}")
        return avg_cost

    print(f"Training on {num_samples} samples with a {num_layers}-layer ansatz...")
    res = minimize(objective_function, initial_params, method='COBYLA', options={'maxiter': max_iter})
    print("\n--- Training Complete ---")
    print(f"Final cost: {res.fun:.4f}")
    optimal_params = res.x
    trained_circuit = ansatz.assign_parameters(optimal_params)
    return trained_circuit

def run_part2(image: np.ndarray, classifier_circuit: QuantumCircuit) -> tuple:
    encoder_circuit = encoder(image)
    if encoder_circuit.num_qubits != classifier_circuit.num_qubits:
        raise ValueError("Mismatch between encoder and classifier qubit count!")
    full_circuit = encoder_circuit.compose(classifier_circuit)
    full_circuit.measure_all()
    simulator = AerSimulator()
    job = simulator.run(full_circuit, shots=4096)
    result = job.result()
    histogram = result.get_counts()
    predicted_label = histogram_to_label(histogram)
    return predicted_label, full_circuit

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    trained_classifier_circuit = train_classifier(train_images, train_labels, num_samples=500, max_iter=100, num_layers=3)
    print("\nClassifier trained and held in memory.")
    print("\n--- Running Part 2 Demonstration ---")
    sample_index = 9
    sample_image = test_images[sample_index]
    actual_label = test_labels[sample_index]
    sample_image_normalized = sample_image / 255.0
    predicted_label, final_circuit = run_part2(sample_image_normalized, trained_classifier_circuit)
    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    print("\n--- Prediction Results ---")
    print(f"Actual Label:    {actual_label} ({label_names[actual_label]})")
    print(f"Predicted Label: {predicted_label} ({label_names[predicted_label]})")
    print("\n--- Calculating Score on a Test Batch ---")
    num_test_samples = 100
    correct_predictions = 0
    total_2q_gates = 0
    for i in range(num_test_samples):
        test_image_norm = test_images[i] / 255.0
        pred_label, full_circ = run_part2(test_image_norm, trained_classifier_circuit)
        if pred_label == test_labels[i]:
            correct_predictions += 1
        decomposed_circuit = qiskit.transpile(full_circ, basis_gates=['u', 'cx'])
        total_2q_gates += decomposed_circuit.count_ops().get('cx', 0)
    accuracy = correct_predictions / num_test_samples
    avg_2q_gates = total_2q_gates / num_test_samples
    noise_overhead = 0.999 ** avg_2q_gates
    score = accuracy * noise_overhead
    print("\n--- Final Score Report ---")
    print(f"Accuracy on {num_test_samples} test samples: {accuracy:.4f}")
    print(f"Average 2-Qubit Gates per circuit: {avg_2q_gates:.1f}")
    print(f"Noise Overhead Factor: {noise_overhead:.4f}")
    print(f"Final Estimated Score: {score:.4f}")
