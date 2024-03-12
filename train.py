from dataloader import prepare_train_set, prepare_test_set
from cnn import build_model, train, plot_history


DATA_TRAIN_PATH = "train-sets/train-synthetic.json"
DATA_TEST_PATH = "test-sets/test-synthetic.json"
SAVED_MODEL_PATH = "model.h5"
EPOCHS = 50
BATCH_SIZE = 32 
PATIENCE = 35
LEARNING_RATE = 0.001
VAL_SIZE = 0.1

# generate train, validation and test sets
X_train, y_train, X_validation, y_validation = prepare_train_set(DATA_TRAIN_PATH, VAL_SIZE)
X_test, y_test, _ = prepare_test_set(DATA_TEST_PATH)

# create network
input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = build_model(input_shape, learning_rate=LEARNING_RATE)

# train network
history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)

# plot accuracy/loss for training/validation set as a function of the epochs
plot_history(history)

# evaluate network on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100 * test_acc))

# save model
model.save(SAVED_MODEL_PATH)
