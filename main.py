def main():
    """This method should be called when the program is run from the command line.
    The aim of the method is to run the complete, automated workflow you developed
    to solve the assignment.

    This function will be called by the automated test suite, so make sure that
    the function signature is not changed, and that it does not require any
    user input.

    If your workflow requires mongoDB (or any other) credentials, please commit them to
    this repository.
    Remember that if the workflow pushed new data to a mongo database without checking
    if the data is already present, the database will contain copies of the data and
    skew the results.

    After having implemented the method, please delete this docstring and replace
    it with a description of what your main method does.

    Hereafter, we provide a **volountarily suboptimal** example of how to structure
    your code. You are free to use this structure, and encouraged to improve it.

    Example:
        def main():
            # acquire the necessary data
            data = acquire()

            # store the data in MongoDB Atlas or Oracle APEX
            store(data)

            # format, project and clean the data
            proprocessed_data = preprocess(data)

            # perform exploratory data analysis
            statistics = explore(proprocessed_data)

            # show your findings
            visualise(statistics)

            # create a model and train it, visualise the results
            model = fit(proprocessed_data)
            visualise(model)
    """
    raise NotImplementedError()


if __name__ == "__main__":
    main()
