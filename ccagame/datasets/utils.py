def get_training_steps(data,epochs,batch_size):
    if data=='mnist':
        return 60000 * epochs // batch_size
    else:
        raise NotImplementedError