from model_utils import define_model
from keras.utils.vis_utils import plot_model

model = define_model()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)