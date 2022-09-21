def print_model(model):
    for layer in model.layers:
        if layer.name == 'model':
            for nested_layer in layer.layers:
                print(nested_layer.name, nested_layer.input, nested_layer.output)
        else:
            print(layer.name, layer.input, layer.output)
