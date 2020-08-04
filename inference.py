import os
import sys
# For the following openvino import statement to work, perform the following steps prior to Intellij launch:
# 1. Launch a Windows cmd terminal
# 2. Enter C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat
# 3. Enter C:\Program Files\JetBrains\IntelliJ IDEA Community Edition <version>\bin\idea.bat
from openvino.inference_engine import IENetwork, IECore

class Network:
    """
    Load and store information for working with the Inference Engine,
    and any loaded models.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None ):
        '''
        Load the model given in form of the OpenVINO IR files.
        '''
        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        #if cpu_extension and "CPU" in device:
        #    self.plugin.add_extension(cpu_extension, device)

        # Load the Intermediate Representation files
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.network = IENetwork(model=model_xml, weights=model_bin)

        self.network.batch_size = 1

        # In case of CPU, check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        if "CPU" in device:
            supported_layers = self.plugin.query_network(network=self.network, device_name= "CPU")
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                print("Unsupported layers found: {}".format(unsupported_layers))
                print("Check whether extensions are available to add to IECore.")
                sys.exit(1)

        assert len(self.network.inputs.keys()) == 1 # YOLOv3-based single input topologies supported only

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input and output layers
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape


    def async_inference(self, frame):
        '''
        Makes an asynchronous inference request, given an input frame.
        '''
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: frame})
        return


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[0].wait(-1)
        return status


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        # return self.exec_network.requests[0].outputs[self.output_blob]
        # return self.exec_network.requests[0].output_blobs
        return self.exec_network.requests[0].outputs