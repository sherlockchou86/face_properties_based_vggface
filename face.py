

from vggface import VGGFace
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

BASE_WEIGHTS_PATH='./vggface_weights/rcmalli_vggface_tf_notop_resnet50.h5'

'''
create new networks based on vggface(resnet50 inside)
you can also use vgg16 or senet50 inside
'''
def Face(train=False):
    base_model = VGGFace(input_shape=(200, 200, 3), include_top=False, model='resnet50', weights=None, pooling='avg')

    # load pre-trained weights if used for fine-tuning
    if train:
        base_model.load_weights(BASE_WEIGHTS_PATH)
        for layer in base_model.layers[:len(base_model.layers)-50]:
            layer.trainable = False
    
    base_output = base_model.output
    # age 1~93, treat age as classifications task
    output_a = Dense(93, activation='softmax', name='predications_age')(base_output)
    # gender 0 or 1
    output_g = Dense(2, activation='softmax', name='predications_gender')(base_output)
    # race 0~4
    output_r = Dense(5, activation='softmax', name='predications_race')(base_output)

    new_model = Model(inputs=base_model.input, outputs=[output_a, output_g, output_r], name='network_based_vggface')

    return new_model

if __name__ == '__main__':
    model = Face(train=True)
    for layer in model.layers:
        print('layer_name:{0}=====trainable:{1}'.format(layer.name, layer.trainable))
    model.summary()