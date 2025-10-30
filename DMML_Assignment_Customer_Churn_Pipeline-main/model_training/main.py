from model_training.model import Model

def train_model(data_path, label_column, drop_columns, model_dir, artifacts_dir):
    xtrain, xtest, ytrain, ytest = Model.load_data(data_path=data_path,
                                                   label_column=label_column, drop_columns=drop_columns)
    model, report = Model.train(xtrain, xtest, ytrain, ytest)
    Model.save_model(model_dir=model_dir,
                     artifacts_dir=artifacts_dir,
                     model=model, report=report)
