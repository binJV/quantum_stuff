import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from skimage.transform import resize
import tensorflow as tf

def image_mse(image_a, image_b, normalized=True):
    """
    Calculates the Mean Squared Error (MSE) between two images.

    Args:
        image_a (np.ndarray): The first image.
        image_b (np.ndarray): The second image.
        normalized (bool): Flag indicating if the images are normalized.

    Returns:
        float: The MSE value.
    """
    if normalized:
        return np.mean((image_a - image_b) ** 2)

    image_a = image_a / np.linalg.norm(image_a) if np.linalg.norm(image_a) != 0 else image_a
    image_b = image_b / np.linalg.norm(image_b) if np.linalg.norm(image_b) != 0 else image_b
    return np.mean((image_a - image_b) ** 2)

def encoder(image: np.ndarray) -> QuantumCircuit:
    """
    Encodes a 28x28 image into a 10-qubit quantum circuit using amplitude encoding.

    Args:
        image (np.ndarray): A 28x28 numpy array representing the grayscale image.

    Returns:
        qiskit.QuantumCircuit: A quantum circuit with the image encoded.
    """
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
    qc.measure_all()
    return qc

def decoder(histogram: dict) -> np.ndarray:
    """
    Decodes a quantum measurement histogram back into an image.

    Args:
        histogram (dict): A dictionary of measurement outcomes (e.g., {'00...': 100, ...}).

    Returns:
        np.ndarray: A 32x32 numpy array representing the reconstructed image.
    """
    num_qubits = len(next(iter(histogram.keys())))
    total_shots = sum(histogram.values())
    reconstructed_vector = np.zeros(2**num_qubits)
    for bitstring, count in histogram.items():
        index = int(bitstring, 2)
        probability = count / total_shots
        amplitude = np.sqrt(probability)
        reconstructed_vector[index] = amplitude
    reconstructed_image = reconstructed_vector.reshape(32, 32)
    return reconstructed_image

def run_part1(image: np.ndarray) -> tuple:
    """
    Executes the full encode-simulate-decode pipeline.

    Args:
        image (np.ndarray): The original 28x28 image.

    Returns:
        tuple: A tuple containing the generated QuantumCircuit and the reconstructed 28x28 image.
    """
    circuit = encoder(image)
    shots = 8192
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    histogram = result.get_counts(circuit)
    reconstructed_image_32x32 = decoder(histogram)
    reconstructed_image_28x28 = resize(reconstructed_image_32x32, (28, 28), anti_aliasing=True)
    return circuit, reconstructed_image_28x28

if __name__ == '__main__':
    (train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
    sample_image = train_images[0]
    sample_image_normalized = sample_image / 255.0
    generated_circuit, reconstructed_image = run_part1(sample_image_normalized)
    original_norm = np.linalg.norm(sample_image_normalized)
    original_for_comparison = sample_image_normalized / original_norm if original_norm != 0 else sample_image_normalized
    mse = image_mse(original_for_comparison, reconstructed_image, normalized=True)
    fidelity = 1 - mse
    try:
        num_2_qubit_gates = generated_circuit.count_ops().get('cx', 0)
    except Exception:
        decomposed_circuit = qiskit.transpiler.transpile(generated_circuit, basis_gates=['u', 'cx'])
        num_2_qubit_gates = decomposed_circuit.count_ops().get('cx', 0)
    noise_overhead_factor = 0.999 ** num_2_qubit_gates
    score = fidelity * noise_overhead_factor
    print("\n--- Results ---")
    print(f"Number of Qubits: {generated_circuit.num_qubits}")
    print(f"Number of 2-Qubit Gates (CNOTs): {num_2_qubit_gates}")
    print(f"Image Reconstruction Fidelity: {fidelity:.6f}")
    print(f"Noise Overhead Factor: {noise_overhead_factor:.6f}")
    print(f"Final Score: {score:.6f}")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(sample_image, cmap='gray')
        axes[0].set_title("Original Image (28x28)")
        axes[0].axis('off')
        axes[1].imshow(reconstructed_image, cmap='gray')
        axes[1].set_title("Reconstructed Image (28x28)")
        axes[1].axis('off')
        fig.suptitle("Quantum Image Reconstruction using Amplitude Encoding")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Please install it (`pip install matplotlib`) to visualize the images.")
