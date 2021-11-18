from ..ccagame.pls.experiments.game





# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    X, _, X_te, _ = mnist()
    input_data_iterator = data_stream(X[:, :400], Y=X[:, 400:], batch_size=None)
    k_per_device = 5
    FLAGS.config = get_config(input_data_iterator, CORES, [400,384],k_per_device=k_per_device)
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, Game))