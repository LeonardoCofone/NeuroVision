const appContent = {
    explanation: {
        welcomeTitle: "Welcome to the Neural Network Visualizer!",
        welcomeText: `
            Hi there! I'm your guide to understanding how a neural network learns. Imagine taking a microscopic look inside the "brain" of an AI.
            To begin our journey, click the "Start Training" button. This will initiate a detailed, step-by-step visualization of a complete training cycle,
            covering both how the network makes predictions (forward propagation) and how it learns from its errors (backpropagation).
            You'll be able to advance step by step to truly grasp each concept.
        `,
        epochCompletedTitle: "Epoch Completed!",
        epochCompletedText: `
            Congratulations! The network has completed one training epoch. Weights and biases have been adjusted to reduce the error.
            You've just witnessed the fundamental learning process of a neural network!
            Click 'Reset Network' to restart with new random weights, or 'Start Training' again to see another epoch with the <i>updated</i> weights.
            <i>Remember, the goal of the training is to find the optimal weights and biases that minimize the loss function, which can be updated based on the 'experience' (past errors) of the model.</i>
        `,
    },

    microStepExplanations: {
        forwardPropInputLayerTitle: "Phase 1: Forward Propagation - Input Layer",
        forwardPropInputLayerText: `
            The journey begins! Our input data, where each value represents a feature (a small group of training set called a batch), is fed into the first layer of neurons, the 'Input Layer'.
            These neurons don’t perform any complex computation: they simply receive the data and pass it forward as their activations.
            Watch as each input neuron lights up with its corresponding value; the network is now awake and ready to learn.
        `,
        forwardPropLayerTitle: (layerName, neuronIndex) => `Phase 1: Forward Propagation - ${layerName} Layer, Neuron ${neuronIndex} (Activation Calculation)`,
        forwardPropLayerText: `
            Now, the neuron gets to work. Each neuron receives signals from all neurons in the previous layer.
            These signals are multiplied by their respective <b>weights (w)</b>, which represent how important each input is.
            Think of weights as the strength of each connection; some inputs matter more than others.

            <br><br>Next, the neuron computes its total input, called the <b>weighted sum</b> or <b>net input (z)</b>, by summing all weighted signals and adding a personal threshold called the <b>bias (b)</b>.
            The bias is like a neuron's personal adjustable threshold, allowing it to activate more or less easily.
            Remember: <i>z = w × x + b</i>.

            <br><br>Then, this net input (<i>z</i>) is passed through an <b>activation function</b>, a small mathematical operation that transforms <i>z</i> into the final output of the neuron.
            This function limits the value to a certain range (like 0 to 1) and adds non-linearity so the network can learn complex patterns.
            See above to discover the most common activation function.

            <br><br>The formula for activation is: <i>a = f(z)</i>, where 'f' is the activation function.

            <br><br>Watch now: the incoming connections light up in <span style='color: var(--forward-flow);'>green</span> as their weighted inputs are combined,
            then the neuron displays its <i>z</i> value, and finally its activation <i>a</i>.
        `,
        calculateLossTitle: "Phase 2: Calculate Loss – Measuring the Error",
        calculateLossText: `
            Forward propagation is complete – the network has made its prediction! Now it’s time to measure how accurate that prediction was.
            We compare the predicted value (the network’s output) with the correct value we expected, called the <b>target</b>, which comes from our labeled training data.
            To measure the difference, we use a <b>loss function</b>. In this case, it's Mean Squared Error (MSE), which tells us how far the prediction was from the target.
            A higher loss means the prediction was worse; a lower loss means it was closer to the truth.
            Our goal in training is to make this loss as small as possible.
            <br><br>The formula for MSE is: <i>Error = (Target - Prediction)²</i>.
            <br><br>Watch now: the output neuron highlights to show its prediction, and the calculated loss is displayed below.
        `,
        calculateLossResultTitle: "Phase 2: Calculate Loss - Result",
        calculateLossResultText: (loss) => `Current Loss (MSE): <b>${loss.toFixed(6)}</b>. This number tells us how far off our prediction was. Our goal is to make this number as small as possible!`,
        backpropOutputDeltaTitle: "Phase 3: Backpropagation - Output Layer Error Gradient (Delta)",
        backpropOutputDeltaText: `
            Now for the magic part: <b>Backpropagation</b>! This is how the network learns from its mistakes.
            We start by calculating the <b>error term</b> (often called 'delta' or 'Δ') for the neuron(s) in the output layer.
            This error term tells us how much the output neuron contributed to the overall loss.
            It's like figuring out who made the biggest mistake in a team project.
            It involves two things: (1) how much the loss changes with respect to the output, and (2) how much the output changes with respect to <i>z</i>.
            <br><br>The formula for Delta in the output layer is: <i>Δ = (Target - Prediction) × f'(z)</i>, where <i>f'(z)</i> is the derivative of the activation function.
            <br><br>Observe the output neuron highlighting as its error term is calculated and displayed.
        `,
        backpropHiddenDeltaTitle: (layerName, neuronIndex) => `Phase 3: Backpropagation – ${layerName} Layer, Neuron ${neuronIndex} - Error Gradient (Delta)`,
        backpropHiddenDeltaText: `
            Now the error is sent backward to the layer. Each neuron calculates its own error gradient (Delta), which tells how much it contributed to the final mistake.
            This depends on the errors from the next layer, weighted by the strength of the connections, and adjusted by how sensitive the neuron is, based on the derivative of its activation function.
            <br><br>This process lets the error "flow" backward, helping each neuron take responsibility for its part in the total loss.
            <br><br>The formula for Delta in a hidden layer is: <i>Δ_j = (Σ_k Δ_k × W_jk) × f'(z_j)</i>
            <br><br>Watch as the outgoing connections from this neuron light up in <span style="color: var(--backward-flow);">red</span>, showing how the error contribution is traced. Then, the neuron displays its Delta value.
        `,
        calculateGradientsTitle: "Phase 3: Backpropagation - Calculate Gradients for Weights and Biases",
        calculateGradientsText: `
            With the error terms (Delta) calculated for all neurons, we can now determine how to adjust each <b>weight</b> and <b>bias</b>.
            These adjustments are guided by <b>gradients</b>. A gradient tells us the direction and magnitude of the steepest increase in error.
            We want to move in the opposite direction to decrease the error.
            The weight gradient tells us how much to change a specific weight, and the bias gradient tells us how much to change a specific bias.
            It's like a teacher giving precise feedback to each connection and to each neuron's threshold.
            <br><br>The formula for weight gradient: <i>∂L/∂w_ij = a_i × Δ_j</i>
            <br>The formula for bias gradient: <i>∂L/∂b_j = Δ_j</i>
            <br><br>Observe connections and neurons highlighting in <span style="color: var(--backward-flow);">red</span> as their gradients are calculated and displayed.
        `,
        updateWeightsBiasesTitle: "Phase 4: Update Weights and Biases - Learning Happens!",
        updateWeightsBiasesText: `
            Finally, we arrive at the moment of learning! Using all the gradients we meticulously calculated,
            we now adjust the <b>weights</b> and <b>biases</b> of the entire network.
            Each parameter is updated by subtracting its gradient, scaled by the <b>learning rate (η)</b>.
            This effectively moves each weight and bias in the direction that reduces the overall prediction error.
            It's like fine-tuning every knob and dial in our complex machine to make it perform better next time.
            This completes an entire <b>epoch</b> of training.
            <br><br>The formula for weight update is: <i>w_new = w_old - η × ∂L/∂w</i>
            <br>The formula for bias update is: <i>b_new = b_old - η × ∂L/∂b</i>
            <br><br>Observe connections and neurons highlighting in <span style="color: var(--weight-update);">orange</span> as their values are updated.
        `,
    },

    exampleSteps: [
        {
            id: 'example-step-0',
            title: 'Start Example:',
            explanation: 'Click "Next Step" to see the calculations in action!',
        },
        {
            id: 'example-step-1',
            title: '1. Forward Propagation: Input to Hidden Layer',
            explanation: `The input values are passed to the neurons in the hidden layer. Each hidden neuron calculates its weighted sum (z) and activation (a) based on these inputs and its own weights and bias.`,
        },
        {
            id: 'example-step-2',
            title: '2. Forward Propagation: Hidden to Output Layer',
            explanation: `The activations from the hidden layer (a<sub>H1</sub>, a<sub>H2</sub>, a<sub>H3</sub>) now become the inputs for the output neuron. The output neuron calculates its weighted sum (z) and final activation (a).`,
        },
        {
            id: 'example-step-3',
            title: '3. Calculate Loss (Error)',
            explanation: `We compare the network's final output (a<sub>Out</sub>) with the desired target value. The Mean Squared Error (MSE) loss function quantifies how "wrong" the prediction is.`,
        },
        {
            id: 'example-step-4',
            title: '4. Backpropagation: Calculate Output Layer Error Gradient (Delta)',
            explanation: `To correct the network, we calculate the error gradient (Delta) for the output neuron. This tells us how much the error changes with respect to the net input (z) of the output neuron, considering the derivative of its activation function.`,
        },
        {
            id: 'example-step-5',
            title: '5. Backpropagation: Calculate Hidden Layer Error Gradients (Delta)',
            explanation: `Now, we propagate the error backward to the hidden layer. Each hidden neuron's error gradient (Delta) depends on its contribution to the output error, weighted by the connections to the output neuron, and the derivative of its own activation function.`,
        },
        {
            id: 'example-step-6',
            title: '6. Backpropagation: Calculate Gradients for Weights and Biases',
            explanation: `Now we calculate the specific gradients for each weight and bias. These indicate the direction and magnitude of the adjustment needed to reduce the error.`,
        },
        {
            id: 'example-step-7',
            title: '7. Update Weights and Biases',
            explanation: `Finally, we update the network's weights and biases using the calculated gradients and the learning rate (eta). This is the core of learning, where the network adapts to make better predictions.`,
        },
        {
            id: 'example-step-8',
            title: 'Epoch Conclusion:',
            explanation: `This cycle of calculation and updating is the heart of neural network learning. By repeating this process many times with different examples, the network learns to make increasingly accurate predictions.`,
        },
    ]
};

export { appContent };
