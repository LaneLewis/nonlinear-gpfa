import dill as pickle

def save_model(file_name,model):
    with open(f"./models/{file_name}", 'wb') as handle:
        pickle.dump(model, handle)

def load_model(file_name):
    with open(f"./models/{file_name}", 'rb') as handle:
        return pickle.load(file_name, handle)