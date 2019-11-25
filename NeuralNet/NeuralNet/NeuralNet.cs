using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class Pulse
    { 
        public double value { get; set; }
    }

    public class Dendrite
    {
        public Pulse InputPulse { get; set; }
        public double SynapticWeight { get; set; }
        public bool Learnable { get; set; } = true;
    }

    public class Neuron
    {
        public List<Dendrite> dendrites { get; set; }
        public Pulse pulseOutput { get; set; }
        double weight;

        public Neuron()
        {
            dendrites = new List<Dendrite>();
            pulseOutput = new Pulse();
        }
        public void UpdateWeights(double new_weights)
        {
            foreach (var terminal in dendrites)
            {
                terminal.SynapticWeight = new_weights;
            }
        }

        public void Fire()
        {
            pulseOutput.value = Sum();
            pulseOutput.value = Activation(pulseOutput.value);
        }

        public void Compute(double learningRate, double delta)
        {
            weight += learningRate * delta;
            foreach (var terminal in dendrites)
            {
                terminal.SynapticWeight = weight;
            }
        }

        public double Sum()
        {
            double cValue = 0.0f;
            foreach (var d in dendrites)
            {
                cValue += d.InputPulse.value * d.SynapticWeight;
            }

            return cValue;
        }

        double Activation(double input)
        {
            double threshold = 1;
            return input >= threshold ? 0 : threshold;
        }
    }


    public class NeuralLayer
    {
        public List<Neuron> Neurons { get; set; }

        public string Name { get; set; }

        public double Weight { get; set; }

        public NeuralLayer(int count, double initialWeight, string name = "")
        {
            Neurons = new List<Neuron>();
            for (int i = 0; i < count; i++)
            {
                Neurons.Add(new Neuron());
            }

            Weight = initialWeight;
            Name = name;
        }

        public void Forward()
        {
            foreach (var neuron in Neurons)
            {
                neuron.Fire();
            }
        }

        public void Compute(double learningRate, double delta)
        {
            foreach (var neuron in Neurons)
            {
                neuron.Compute(learningRate, delta);
            }
        }

        public void Log()
        {
            Console.WriteLine("{0}, Weight: {1}", Name, Weight);
        }

        public void Optimize(double learningRate, double delta)
        {
            Weight += learningRate * delta;
            foreach (var neuron in Neurons)
            {
                neuron.UpdateWeights(Weight);
            }
        }

    }


    public class NetworkModel
    {
        public List<NeuralLayer> layers { get; set; }

        public NetworkModel()
        {
            layers = new List<NeuralLayer>();
        }

        public void AddLayer(NeuralLayer layer)
        {
            int denriteCount = 1;
            if (layers.Count > 0)
            {
                denriteCount = layers[layers.Count - 1].Neurons.Count;
            }

            foreach (var element in layer.Neurons)
            {
                for (int i = 0; i < denriteCount; i++)
                {
                    element.dendrites.Add(new Dendrite());
                }
            }
        }


        public void Build()
        {
            int i = 0;
            foreach (var layer in layers)
            {
                if (i >= layers.Count - 1)
                {
                    break;
                }

                var nextLayer = layers[i + 1];
                CreateNetwork(layer, nextLayer);

                i++;
            }
        }

        public void Train(NeuralData X, NeuralData Y, int iterations, double learningRate = 0.1)
        {
            int epoch = 1;
            //Loop till the number of iterations
            while (iterations >= epoch)
            {
                //Get the input layers
                var inputLayer = layers[0];
                List<double> outputs = new List<double>();

                //Loop through the record
                for (int i = 0; i < X.Data.Length; i++)
                {
                    //Set the input data into the first layer
                    for (int j = 0; j < X.Data[i].Length; j++)
                    {
                        inputLayer.Neurons[j].pulseOutput.value = X.Data[i][j];
                    }

                    //Fire all the neurons and collect the output
                    ComputeOutput();
                    outputs.Add(layers[layers.Count -1].Neurons[0].pulseOutput.value);
                }

                //Check the accuracy score against Y with the actual output
                double accuracySum = 0;
                int y_counter = 0;
                outputs.ForEach((x) =>
                {
                    if (x == Y.Data[y_counter][0])
                    {
                        accuracySum++;
                    }

                    y_counter++;
                });

                //Optimize the synaptic weights
                OptimizeWeights(accuracySum / y_counter);
                Console.WriteLine("Epoch: {0}, Accuracy: {1} %", epoch, (accuracySum / y_counter) * 100);
                epoch++;
            }
        }

        private void ComputeOutput()
        {
            bool first = true;
            foreach (var layer in layers)
            {
                //Skip first layer as it is input
                if (first)
                {
                    first = false;
                    continue;
                }

                layer.Forward();
            }
        }

        private void OptimizeWeights(double accuracy)
        {
            float lr = 0.1f;
            //Skip if the accuracy reached 100%
            if (accuracy == 1)
            {
                return;
            }

            if (accuracy > 1)
            {
                lr = -lr;
            }

            //Update the weights for all the layers
            foreach (var layer in layers)
            {
                layer.Optimize(lr, 1);
            }
        }

        private void CreateNetwork(NeuralLayer connectingFrom, NeuralLayer connectingTo)
        {
            foreach (var to in connectingTo.Neurons)
            {
                foreach (var from in connectingFrom.Neurons)
                {
                    to.dendrites.Add(new Dendrite() { InputPulse = to.pulseOutput, SynapticWeight = connectingTo.Weight });
                }
            }
        }
    }

    public class NeuralData
    {
        public double[][] Data { get; set; }

        int counter = 0;

        public NeuralData(int rows)
        {
            Data = new double[rows][];
        }

        public void Add(params double[] rec)
        {
            Data[counter] = rec;
            counter++;
        }
    }
}
