def get_training_steps(data, epochs, batch_size):
    if data == "mnist":
        return 60000 * epochs // batch_size
    elif data in ["linear", "exponential"]:
        return 1000 * epochs // batch_size
    elif data == "xrmb":
        return 1429236 * epochs // batch_size
    else:
        raise NotImplementedError
