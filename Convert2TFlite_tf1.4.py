# -*- coding:utf-8 -*-
import tensorflow as tf

in_path = "model.pb"
out_path = "model.tflite"
# out_path = "./model/quantize_frozen_graph.tflite"

# 模型输入节点
input_tensor_name = ["input"]
input_tensor_shape = {"input":[1 ,50, 50, 3]}
# 模型输出节点
output_tensor_name = ["output"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path,
                                            input_tensor_name, output_tensor_name,
                                            input_shapes = input_tensor_shape)
converter.post_training_quantize = True
tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)
