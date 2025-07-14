from __future__ import annotations

import abcdstats

# Most configuration can be done via the supplied YAML file which, in turn, optionally
# refers to a CSV file.

# However, it is also easy to override any function in the abcdstats module's
# `BasicWorkflow` class, by subclassing `BasicWorkflow` and overriding the methods of
# interest.


class MyWorkflow(abcdstats.BasicWorkflow):
    def __init__(self, *, yaml_file: str) -> None:
        super().__init__(yaml_file)
        # Add any additional initialization here

    # If you wish to override any class methods in a way that changing the configuration
    # within the YAML file (and optional CSV file) won't accomplish, write them here.


# Configure the BasicWorkflow: Which inputs, what processing, which outputs?
myWorkflow: MyWorkflow = MyWorkflow(yaml_file="config.yaml")
# Alternatively, construct with no arguments and then invoke
#   myWorkflow.configure(yaml_file="config.yaml")

# Now that it is configured, run it.
myWorkflow.run()
