from decimal import InvalidContext
import torch
from torch._C import DeserializationStorageContext
from torchvision import transforms
from random import randint
import pickle
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import tenseal as ts
from typing import Dict

##################
# Client Helpers #
##################
# Create the TenSEAL security context
def create_ctx():
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = 2 ** 21
    ctx.generate_galois_keys()
    return ctx

# Sample an image
def load_input():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    idx = randint(1, 7)
    img_name = "data/mnist-samples/img_{}.jpg".format(idx)
    print(img_name)
    img = Image.open(img_name)
    return transform(img).view(28, 28).tolist(), img

# Helper for encoding the image
def prepare_input(ctx, plain_input):
    enc_input, windows_nb = ts.im2col_encoding(ctx, plain_input, 7, 7, 3)
    assert windows_nb == 64
    return enc_input

################
# Server Model #
################
# Load a pretrained model and adapt the forward call for encrypted input
class ConvMNIST():
    def __init__(self, parameters: Dict[str, list]):
        self.conv1_weight = parameters["conv1_weight"]
        self.conv1_bias = parameters["conv1_bias"]
        self.fc1_weight = parameters["fc1_weight"]
        self.fc1_bias = parameters["fc1_bias"]
        self.fc2_weight = parameters["fc2_weight"]
        self.fc2_bias = parameters["fc2_bias"]
        self.windows_nb = parameters["windows_nb"]

    def forward(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
        channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, self.windows_nb) + bias
            channels.append(y)
        out = ts.CKKSVector.pack_vectors(channels)
        out.square_()
        out = out.mm_(self.fc1_weight) + self.fc1_bias
        out.square_()
        out = out.mm_(self.fc2_weight) + self.fc2_bias
        return out

    @staticmethod
    def prepare_input(context: bytes, ckks_vector: bytes) -> ts.CKKSVector:
        try:
            ctx = ts.context_from(context)
            enc_x = ts.ckks_vector_from(ctx, ckks_vector)
        except:
            raise DeserializationStorageContext("cannot deserialize context or ckks_vector")
        try:
            _ = ctx.galois_keys()
        except:
            raise InvalidContext("the context doesn't hold galois keys")
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

##################
# Server helpers #
##################
def load_parameters(file_path: str) -> dict:
    try:
        parameters = pickle.load(open(file_path, "rb"))
        print(f"Model loaded from '{file_path}'")
    except OSError as ose:
        print("error", ose)
        raise ose
    return parameters

parameters = load_parameters("parameters/ConvMNIST-0.1.pickle")
model = ConvMNIST(parameters)

################
# Client Query #
################
# CKKS context generation.
context = create_ctx()

# Random image sampling
image, orig = load_input()

# Image encoding
encrypted_image = prepare_input(context, image)
print("Encrypted image ", encrypted_image)
# Image original
plt.figure()
plt.title("Original Image")
imshow(np.asarray(orig), cmap='gray')
plt.show()

# We prepare the context for the server, by making it public(we drop the secret key)
server_context = context.copy()
server_context.make_context_public()

# Context and ciphertext serialization
server_context = server_context.serialize()
encrypted_image = encrypted_image.serialize()

client_query = {
    "data": encrypted_image,
    "context": server_context,
}

####################
# Server inference #
####################
encrypted_query = model.prepare_input(client_query["context"], client_query["data"])
encrypted_result = model(encrypted_query).serialize()

server_response = {
    "data" : encrypted_result
}

###########################
# Client process response #
###########################
result = ts.ckks_vector_from(context, encrypted_result).decrypt()
probs = torch.softmax(torch.tensor(result), 0)
label_max = torch.argmax(probs)
print("Maximum probability for label {}".format(label_max))
