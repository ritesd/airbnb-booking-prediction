
from booking_model.prediction_model import PredictionModel, load_data


if __name__=='__main__':
    train_data, test_data = load_data()
    prepros = PredictionModel()
    X, y = prepros.preprocessing(data=train_data)
    X_train, X_test, y_train, y_test = prepros.split_train_test(X, y)

    X_train_transformed, X_test_transformed, pre_process = prepros.get_transformed_data(X_train, X_test)

    # creating and storing model
    prepros.create_model(pre_process, X_train_transformed, y_train, X_train)

    output = prepros.predict_contries(test_data)





