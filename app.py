import onnx
import onnxruntime as ort
import numpy as np
import cv2

# Especifica la ruta al archivo ONNX del modelo
model_path = "discord_pfp_generator_model.onnx"

# Cargar el modelo ONNX
ort_session = ort.InferenceSession(model_path)

# Obtener las dimensiones de entrada requeridas por el modelo
input_name = ort_session.get_inputs()[0].name
input_shape = ort_session.get_inputs()[0].shape
print(f"Nombre de entrada: {input_name}")
print(f"Dimensiones de entrada requeridas: {input_shape}")

# Ajustar los datos de entrada para que coincidan con las dimensiones requeridas
input_data = np.random.rand(1, 100, 1, 1).astype(np.float32)  # Ajusta las dimensiones según tu modelo

# Ejecutar el modelo para generar la salida
output = ort_session.run(None, {input_name: input_data})

# Ajustar la forma de la salida según tu modelo
output = output[0].squeeze()

# Asegúrate de que la salida esté en el rango adecuado (por ejemplo, 0-255 para imágenes)
output = (output * 255).astype(np.uint8)

# Muestra la imagen generada o guárdala en disco
cv2.imshow('Imagen Generada', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
# O bien, guarda la imagen en disco
cv2.imwrite('imagen_generada.png', output)
