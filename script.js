import { appContent } from './content.js';

document.addEventListener('DOMContentLoaded', () => {
    const config = {
        inputSize: 2,
        hiddenSizes: [3],
        outputSize: 1,
        learningRate: 0.1,
        maxEpochs: 1,
        activationFn: 'sigmoid',
        lossFn: 'mse',
    };

    const DOMElements = {
        startBtn: document.getElementById('start-btn'),
        nextStepBtn: document.getElementById('next-step-btn'),
        resetBtn: document.getElementById('reset-btn'),
        fullNetworkView: document.getElementById('full-network-view'),
        neuronDetailView: document.getElementById('neuron-detail-view'),
        phaseExplanation: document.getElementById('phase-explanation'),
        neuronTooltip: document.getElementById('neuron-tooltip'),
        explanationTitle: document.getElementById('explanation-title'),
        explanationText: document.getElementById('explanation-text'),
        exampleStepsContainer: document.getElementById('example-steps'),
        exampleInput1: document.getElementById('example-input1'),
        exampleInput2: document.getElementById('example-input2'),
        exampleLr: document.getElementById('example-lr'),

        getExampleElement: (id) => document.getElementById(id),
    };

    let currentMicroStep = 0;
    let microSteps = [];
    let allNeuronsElements = [];
    let allConnectionsElements = [];
    let networkState = [];
    let inputData = [0.9, 0.8];
    let targetOutput = [1.0];
    let isProcessingStep = false;

    const ActivationFunctions = {
        sigmoid: (x) => 1 / (1 + Math.exp(-x)),
        sigmoid_derivative: (a) => a * (1 - a)
    };

    const LossFunctions = {
        mse: (predictions, targets) => {
            let sum = 0;
            for (let i = 0; i < predictions.length; i++) {
                sum += Math.pow(predictions[i] - targets[i], 2);
            }
            return sum / predictions.length;
        },
        mse_derivative: (prediction, target) => 2 * (prediction - target)
    };

    function getRandomWeight() {
        return (Math.random() * 2 - 1);
    }

    function updateExplanation(title, text) {
        DOMElements.explanationTitle.textContent = title;
        DOMElements.explanationText.innerHTML = text;
    }

    function displayValue(element, value) {
        let valueDisplay = element.querySelector('.value-display');
        if (!valueDisplay) {
            valueDisplay = document.createElement('div');
            valueDisplay.classList.add('value-display');
            element.appendChild(valueDisplay);
        }
        valueDisplay.textContent = value.toFixed(4);
        valueDisplay.classList.add('show');

        setTimeout(() => {
            valueDisplay.classList.remove('show');
        }, 1000);
    }

    function clearAllVisualCues() {
        allNeuronsElements.forEach(n => n.classList.remove('neuron-active'));
        allConnectionsElements.forEach(c => c.classList.remove('connection-active-forward', 'connection-active-backward', 'connection-weight-update'));
        document.querySelectorAll('.value-display').forEach(vd => vd.remove());
    }

    function initializeNetworkLayout() {
        DOMElements.fullNetworkView.innerHTML = '';
        allNeuronsElements = [];
        allConnectionsElements = [];
        networkState = [];

        const layers = ['input', ...config.hiddenSizes.map(() => 'hidden'), 'output'];

        layers.forEach((type, layerIndex) => {
            const layerDiv = document.createElement('div');
            layerDiv.classList.add('layer');
            layerDiv.setAttribute('data-layer-index', layerIndex);

            const layerBadge = document.createElement('div');
            layerBadge.className = 'layer-badge';
            layerBadge.textContent = layerIndex;
            layerDiv.appendChild(layerBadge);

            let neuronCount;
            if (type === 'input') neuronCount = config.inputSize;
            else if (type === 'output') neuronCount = config.outputSize;
            else neuronCount = config.hiddenSizes[layerIndex - 1];

            networkState[layerIndex] = { neurons: [] };

            for (let i = 0; i < neuronCount; i++) {
                const neuronElement = document.createElement('div');
                neuronElement.className = 'neuron';
                neuronElement.setAttribute('data-neuron-id', `${layerIndex}-${i}`);
                neuronElement.setAttribute('data-layer-index', layerIndex);
                neuronElement.setAttribute('data-neuron-index', i);
                neuronElement.setAttribute('tabindex', 0);

                const neuronData = {
                    activation: 0,
                    z: 0,
                    bias: getRandomWeight(),
                    error: 0,
                    bias_gradient: 0,
                    weights: [],
                    weight_gradients: []
                };
                networkState[layerIndex].neurons.push(neuronData);
                allNeuronsElements.push(neuronElement);
                layerDiv.appendChild(neuronElement);

                const valueDisplay = document.createElement('div');
                valueDisplay.classList.add('value-display');
                neuronElement.appendChild(valueDisplay);

                neuronElement.addEventListener('mouseenter', (e) => handleNeuronHover(e, layerIndex, i));
                neuronElement.addEventListener('mouseleave', hideTooltip);
                neuronElement.addEventListener('focus', (e) => handleNeuronHover(e, layerIndex, i));
                neuronElement.addEventListener('blur', hideTooltip);
            }
            DOMElements.fullNetworkView.appendChild(layerDiv);
        });

        setTimeout(() => {
            createAllConnections();
        }, 0);
    }

    function createAllConnections() {
        allConnectionsElements.forEach(conn => conn.remove());
        allConnectionsElements = [];

        for (let layerIndex = 0; layerIndex < networkState.length - 1; layerIndex++) {
            const currentLayerNeuronsElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === layerIndex);
            const nextLayerNeuronsElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === layerIndex + 1);

            currentLayerNeuronsElements.forEach((startNeuronElement, startNeuronIdx) => {
                nextLayerNeuronsElements.forEach((endNeuronElement, endNeuronIdx) => {
                    const conn = document.createElement('div');
                    conn.classList.add('connection');
                    conn.setAttribute('data-source-layer', layerIndex);
                    conn.setAttribute('data-source-neuron', startNeuronIdx);
                    conn.setAttribute('data-target-layer', layerIndex + 1);
                    conn.setAttribute('data-target-neuron', endNeuronIdx);

                    const weight = getRandomWeight();
                    networkState[layerIndex].neurons[startNeuronIdx].weights[endNeuronIdx] = weight;
                    networkState[layerIndex].neurons[startNeuronIdx].weight_gradients[endNeuronIdx] = 0;

                    const startRect = startNeuronElement.getBoundingClientRect();
                    const endRect = endNeuronElement.getBoundingClientRect();
                    const containerRect = DOMElements.fullNetworkView.getBoundingClientRect();

                    const x1 = startRect.left + startRect.width / 2 - containerRect.left;
                    const y1 = startRect.top + startRect.height / 2 - containerRect.top;
                    const x2 = endRect.left + endRect.width / 2 - containerRect.left;
                    const y2 = endRect.top + endRect.height / 2 - containerRect.top;

                    const length = Math.hypot(x2 - x1, y2 - y1);
                    const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

                    conn.style.width = `${length}px`;
                    conn.style.left = `${x1}px`;
                    conn.style.top = `${y1}px`;
                    conn.style.transformOrigin = '0 0';
                    conn.style.transform = `rotate(${angle}deg)`;
                    conn.style.setProperty('--angle', `${angle}deg`);

                    DOMElements.fullNetworkView.appendChild(conn);
                    allConnectionsElements.push(conn);

                    const valueDisplay = document.createElement('div');
                    valueDisplay.classList.add('value-display');
                    conn.appendChild(valueDisplay);
                });
            });
        }
    }

    function handleNeuronHover(event, layerIndex, neuronIndex) {
        const neuronData = networkState[layerIndex].neurons[neuronIndex];
        let tooltipContent = `
            <strong>Strato ${layerIndex}, Neurone ${neuronIndex + 1}</strong><br>
            Attivazione: ${neuronData.activation.toFixed(4)}<br>
            Bias: ${neuronData.bias.toFixed(4)}<br>
            Errore (Delta): ${neuronData.error.toFixed(4)}<br>
            Gradiente Bias: ${neuronData.bias_gradient.toFixed(4)}<br>
        `;
        if (layerIndex < networkState.length - 1) {
            tooltipContent += `Pesi al prossimo strato: [${neuronData.weights.map(w => w.toFixed(4)).join(', ')}]<br>`;
            tooltipContent += `Gradienti Pesi: [${neuronData.weight_gradients.map(wg => wg.toFixed(4)).join(', ')}]`;
        }

        DOMElements.neuronTooltip.innerHTML = tooltipContent;
        DOMElements.neuronTooltip.style.opacity = '1';
        DOMElements.neuronTooltip.style.display = 'block';

        positionTooltip(event.currentTarget);
        DOMElements.neuronTooltip.setAttribute('aria-hidden', 'false');
    }

    function hideTooltip() {
        DOMElements.neuronTooltip.style.opacity = '0';
        DOMElements.neuronTooltip.style.display = 'none';
        DOMElements.neuronTooltip.setAttribute('aria-hidden', 'true');
    }

    function positionTooltip(targetElement) {
        const rect = targetElement.getBoundingClientRect();
        const tooltipRect = DOMElements.neuronTooltip.getBoundingClientRect();
        const containerRect = DOMElements.fullNetworkView.getBoundingClientRect();

        let left = rect.left + rect.width / 2 - containerRect.left - tooltipRect.width / 2;
        let top = rect.top - containerRect.top - tooltipRect.height - 10;

        if (left < 0) left = 5;
        if (top < 0) top = rect.bottom - containerRect.top + 10;

        DOMElements.neuronTooltip.style.left = `${left}px`;
        DOMElements.neuronTooltip.style.top = `${top}px`;
    }

    function showFullNetworkView() {
        DOMElements.fullNetworkView.style.display = 'flex';
        DOMElements.neuronDetailView.style.display = 'none';
        DOMElements.phaseExplanation.textContent = 'Panoramica della Rete';
    }

    let currentExampleStep = 0;
    const exampleStepsData = appContent.exampleSteps.map((step, index) => ({
        ...step,
        updateExample: () => {
            const i1 = inputData[0];
            const i2 = inputData[1];
            const lr = config.learningRate;

            const updateElement = (id, value) => {
                const element = DOMElements.getExampleElement(id) || document.getElementById(id);
                if (element) {
                    element.innerHTML = value;
                }
            };

            switch (index) {
                case 0:
                    updateElement('example-input1', i1.toFixed(4));
                    updateElement('example-input2', i2.toFixed(4));
                    updateElement('example-lr', lr.toFixed(1));
                    updateElement('final-b-o', 'N/A');
                    updateElement('final-w-h1-o', 'N/A');
                    updateElement('final-w-h2-o', 'N/A');
                    updateElement('final-w-h3-o', 'N/A');
                    updateElement('final-b-h1', 'N/A');
                    updateElement('final-w-i1-h1', 'N/A');
                    updateElement('final-w-i2-h1', 'N/A');
                    break;
                case 1:
                    if (!networkState[0] || !networkState[1]) return;

                    updateElement('step1-input1', i1.toFixed(4));
                    updateElement('step1-input2', i2.toFixed(4));

                    const wI1H1 = networkState[0].neurons[0].weights[0];
                    const wI2H1 = networkState[0].neurons[1].weights[0];
                    const bH1 = networkState[1].neurons[0].bias;
                    const zH1 = (i1 * wI1H1) + (i2 * wI2H1) + bH1;
                    const aH1 = ActivationFunctions.sigmoid(zH1);
                    networkState[1].neurons[0].z = zH1;
                    networkState[1].neurons[0].activation = aH1;
                    updateElement('step1-i1-h1', i1.toFixed(4));
                    updateElement('step1-w1-h1', wI1H1.toFixed(4));
                    updateElement('step1-i2-h1', i2.toFixed(4));
                    updateElement('step1-w2-h1', wI2H1.toFixed(4));
                    updateElement('step1-b-h1', bH1.toFixed(4));
                    updateElement('step1-z-h1', zH1.toFixed(4));
                    updateElement('step1-a-h1', aH1.toFixed(4));

                    const wI1H2 = networkState[0].neurons[0].weights[1];
                    const wI2H2 = networkState[0].neurons[1].weights[1];
                    const bH2 = networkState[1].neurons[1].bias;
                    const zH2 = (i1 * wI1H2) + (i2 * wI2H2) + bH2;
                    const aH2 = ActivationFunctions.sigmoid(zH2);
                    networkState[1].neurons[1].z = zH2;
                    networkState[1].neurons[1].activation = aH2;
                    updateElement('step1-i1-h2', i1.toFixed(4));
                    updateElement('step1-w1-h2', wI1H2.toFixed(4));
                    updateElement('step1-i2-h2', i2.toFixed(4));
                    updateElement('step1-w2-h2', wI2H2.toFixed(4));
                    updateElement('step1-b-h2', bH2.toFixed(4));
                    updateElement('step1-z-h2', zH2.toFixed(4));
                    updateElement('step1-a-h2', aH2.toFixed(4));

                    const wI1H3 = networkState[0].neurons[0].weights[2];
                    const wI2H3 = networkState[0].neurons[1].weights[2];
                    const bH3 = networkState[1].neurons[2].bias;
                    const zH3 = (i1 * wI1H3) + (i2 * wI2H3) + bH3;
                    const aH3 = ActivationFunctions.sigmoid(zH3);
                    networkState[1].neurons[2].z = zH3;
                    networkState[1].neurons[2].activation = aH3;
                    updateElement('step1-i1-h3', i1.toFixed(4));
                    updateElement('step1-w1-h3', wI1H3.toFixed(4));
                    updateElement('step1-i2-h3', i2.toFixed(4));
                    updateElement('step1-w2-h3', wI2H3.toFixed(4));
                    updateElement('step1-b-h3', bH3.toFixed(4));
                    updateElement('step1-z-h3', zH3.toFixed(4));
                    updateElement('step1-a-h3', aH3.toFixed(4));
                    break;
                case 2:
                    if (!networkState[1] || !networkState[2]) return;

                    const current_aH1 = networkState[1].neurons[0].activation;
                    const current_aH2 = networkState[1].neurons[1].activation;
                    const current_aH3 = networkState[1].neurons[2].activation;

                    updateElement('step2-a-h1', current_aH1.toFixed(4));
                    updateElement('step2-a-h2', current_aH2.toFixed(4));
                    updateElement('step2-a-h3', current_aH3.toFixed(4));

                    const wH1O = networkState[1].neurons[0].weights[0];
                    const wH2O = networkState[1].neurons[1].weights[0];
                    const wH3O = networkState[1].neurons[2].weights[0];
                    const bO = networkState[2].neurons[0].bias;

                    const zO = (current_aH1 * wH1O) + (current_aH2 * wH2O) + (current_aH3 * wH3O) + bO;
                    const aO = ActivationFunctions.sigmoid(zO);
                    networkState[2].neurons[0].z = zO;
                    networkState[2].neurons[0].activation = aO;

                    updateElement('step2-a1-o', current_aH1.toFixed(4));
                    updateElement('step2-w1-o', wH1O.toFixed(4));
                    updateElement('step2-a2-o', current_aH2.toFixed(4));
                    updateElement('step2-w2-o', wH2O.toFixed(4));
                    updateElement('step2-a3-o', current_aH3.toFixed(4));
                    updateElement('step2-w3-o', wH3O.toFixed(4));
                    updateElement('step2-b-o', bO.toFixed(4));
                    updateElement('step2-z-o', zO.toFixed(4));
                    updateElement('step2-a-o', aO.toFixed(4));
                    break;
                case 3:
                    if (!networkState[2]) return;
                    const current_aO = networkState[2].neurons[0].activation;
                    const target = targetOutput[0];
                    const error = Math.pow(target - current_aO, 2);

                    updateElement('step3-target', target.toFixed(1));
                    updateElement('step3-a-o', current_aO.toFixed(4));
                    updateElement('step3-error', error.toFixed(4));
                    break;
                case 4:
                    if (!networkState[2]) return;
                    const outputNeuronDelta = networkState[2].neurons[0];
                    const current_aO_delta = outputNeuronDelta.activation;
                    const current_zO_delta = outputNeuronDelta.z;
                    const target_delta = targetOutput[0];
                    const deltaO = (target_delta - current_aO_delta) * ActivationFunctions.sigmoid_derivative(current_aO_delta);
                    outputNeuronDelta.error = deltaO;

                    updateElement('step4-target', target_delta.toFixed(1));
                    updateElement('step4-a-o', current_aO_delta.toFixed(4));
                    // Aggiungi qui l'aggiornamento per il valore di 'a' nella formula f'(z) = a * (1 - a)
                    updateElement('step4-a-o-formula-part1', current_aO_delta.toFixed(4)); // Nuovo ID per il primo 'a'
                    updateElement('step4-a-o-formula-part2', current_aO_delta.toFixed(4)); // Nuovo ID per il secondo 'a'
                    updateElement('step4-delta-o', deltaO.toFixed(4));
                    break;
                case 5:
                    if (!networkState[1] || !networkState[2]) return;
                    const current_deltaO = networkState[2].neurons[0].error;
                    const current_wH1O = networkState[1].neurons[0].weights[0];
                    const current_wH2O = networkState[1].neurons[1].weights[0];
                    const current_wH3O = networkState[1].neurons[2].weights[0];

                    const current_zH1 = networkState[1].neurons[0].z;
                    const current_aH1_delta = networkState[1].neurons[0].activation;
                    const deltaH1 = (current_deltaO * current_wH1O) * ActivationFunctions.sigmoid_derivative(current_aH1_delta);
                    networkState[1].neurons[0].error = deltaH1;
                    updateElement('step5-delta-o', current_deltaO.toFixed(4));
                    updateElement('step5-w-h1-o', current_wH1O.toFixed(4));
                    updateElement('step5-a-h1', current_aH1_delta.toFixed(4));
                    // Aggiungi qui l'aggiornamento per il valore di 'a' nella formula f'(z) = a * (1 - a)
                    updateElement('step5-a-h1-formula-part1', current_aH1_delta.toFixed(4)); // Nuovo ID per il primo 'a'
                    updateElement('step5-a-h1-formula-part2', current_aH1_delta.toFixed(4)); // Nuovo ID per il secondo 'a'
                    updateElement('step5-delta-h1', deltaH1.toFixed(4));

                    const current_zH2 = networkState[1].neurons[1].z;
                    const current_aH2_delta = networkState[1].neurons[1].activation;
                    const deltaH2 = (current_deltaO * current_wH2O) * ActivationFunctions.sigmoid_derivative(current_aH2_delta);
                    networkState[1].neurons[1].error = deltaH2;
                    updateElement('step5-delta-o2', current_deltaO.toFixed(4));
                    updateElement('step5-w-h2-o', current_wH2O.toFixed(4));
                    updateElement('step5-a-h2', current_aH2_delta.toFixed(4));
                    // Aggiungi qui l'aggiornamento per il valore di 'a' nella formula f'(z) = a * (1 - a)
                    updateElement('step5-a-h2-formula-part1', current_aH2_delta.toFixed(4)); // Nuovo ID per il primo 'a'
                    updateElement('step5-a-h2-formula-part2', current_aH2_delta.toFixed(4)); // Nuovo ID per il secondo 'a'
                    updateElement('step5-delta-h2', deltaH2.toFixed(4));

                    const current_zH3 = networkState[1].neurons[2].z;
                    const current_aH3_delta = networkState[1].neurons[2].activation;
                    const deltaH3 = (current_deltaO * current_wH3O) * ActivationFunctions.sigmoid_derivative(current_aH3_delta);
                    networkState[1].neurons[2].error = deltaH3;
                    updateElement('step5-delta-o3', current_deltaO.toFixed(4));
                    updateElement('step5-w-h3-o', current_wH3O.toFixed(4));
                    updateElement('step5-a-h3', current_aH3_delta.toFixed(4));
                    // Aggiungi qui l'aggiornamento per il valore di 'a' nella formula f'(z) = a * (1 - a)
                    updateElement('step5-a-h3-formula-part1', current_aH3_delta.toFixed(4)); // Nuovo ID per il primo 'a'
                    updateElement('step5-a-h3-formula-part2', current_aH3_delta.toFixed(4)); // Nuovo ID per il secondo 'a'
                    updateElement('step5-delta-h3', deltaH3.toFixed(4));
                    break;
                case 6:
                    if (!networkState[0] || !networkState[1] || !networkState[2]) return;

                    const inputNeuron1 = networkState[0].neurons[0];
                    const inputNeuron2 = networkState[0].neurons[1];
                    const hiddenNeuron1 = networkState[1].neurons[0];
                    const hiddenNeuron2 = networkState[1].neurons[1];
                    const hiddenNeuron3 = networkState[1].neurons[2];
                    const outputNeuron = networkState[2].neurons[0];

                    const current_aH1_grad = hiddenNeuron1.activation;
                    const current_aH2_grad = hiddenNeuron2.activation;
                    const current_aH3_grad = hiddenNeuron3.activation;
                    const current_deltaO_grad = outputNeuron.error;

                    const gradWH1O = current_aH1_grad * current_deltaO_grad;
                    const gradWH2O = current_aH2_grad * current_deltaO_grad;
                    const gradWH3O = current_aH3_grad * current_deltaO_grad;
                    const gradBO = current_deltaO_grad;
                    hiddenNeuron1.weight_gradients[0] = gradWH1O;
                    hiddenNeuron2.weight_gradients[0] = gradWH2O;
                    hiddenNeuron3.weight_gradients[0] = gradWH3O;
                    outputNeuron.bias_gradient = gradBO;

                    updateElement('step6-a-h1-o', current_aH1_grad.toFixed(4));
                    updateElement('step6-delta-o', current_deltaO_grad.toFixed(4));
                    updateElement('step6-grad-w-h1-o', gradWH1O.toFixed(4));
                    updateElement('step6-a-h2-o', current_aH2_grad.toFixed(4));
                    updateElement('step6-delta-o2', current_deltaO_grad.toFixed(4));
                    updateElement('step6-grad-w-h2-o', gradWH2O.toFixed(4));
                    updateElement('step6-a-h3-o', current_aH3_grad.toFixed(4));
                    updateElement('step6-delta-o3', current_deltaO_grad.toFixed(4));
                    updateElement('step6-grad-w-h3-o', gradWH3O.toFixed(4));
                    updateElement('step6-grad-b-o', gradBO.toFixed(4));

                    const current_deltaH1_grad = hiddenNeuron1.error;
                    const gradWI1H1 = inputNeuron1.activation * current_deltaH1_grad;
                    const gradWI2H1 = inputNeuron2.activation * current_deltaH1_grad;
                    const gradBH1 = current_deltaH1_grad;
                    inputNeuron1.weight_gradients[0] = gradWI1H1;
                    inputNeuron2.weight_gradients[0] = gradWI2H1;
                    hiddenNeuron1.bias_gradient = gradBH1;

                    updateElement('step6-i1-h1', inputNeuron1.activation.toFixed(4));
                    updateElement('step6-delta-h1', current_deltaH1_grad.toFixed(4));
                    updateElement('step6-grad-w-i1-h1', gradWI1H1.toFixed(4));
                    updateElement('step6-i2-h1', inputNeuron2.activation.toFixed(4));
                    updateElement('step6-delta-h1-2', current_deltaH1_grad.toFixed(4));
                    updateElement('step6-grad-w-i2-h1', gradWI2H1.toFixed(4));
                    updateElement('step6-grad-b-h1', gradBH1.toFixed(4));
                    break;
                case 7:
                    if (!networkState[0] || !networkState[1] || !networkState[2]) return;

                    const current_lr = config.learningRate;

                    const oldWH1O = networkState[1].neurons[0].weights[0];
                    const oldWH2O = networkState[1].neurons[1].weights[0];
                    const oldWH3O = networkState[1].neurons[2].weights[0];
                    const oldBO = networkState[2].neurons[0].bias;

                    const current_gradWH1O = networkState[1].neurons[0].weight_gradients[0];
                    const current_gradWH2O = networkState[1].neurons[1].weight_gradients[0];
                    const current_gradWH3O = networkState[1].neurons[2].weight_gradients[0];
                    const current_gradBO = networkState[2].neurons[0].bias_gradient;

                    const newWH1O = oldWH1O - current_lr * current_gradWH1O;
                    const newWH2O = oldWH2O - current_lr * current_gradWH2O;
                    const newWH3O = oldWH3O - current_lr * current_gradWH3O;
                    const newBO = oldBO - current_lr * current_gradBO;

                    networkState[1].neurons[0].weights[0] = newWH1O;
                    networkState[1].neurons[1].weights[0] = newWH2O;
                    networkState[1].neurons[2].weights[0] = newWH3O;
                    networkState[2].neurons[0].bias = newBO;

                    updateElement('step7-w-h1-o-old', oldWH1O.toFixed(4));
                    updateElement('step7-lr-1', current_lr.toFixed(1));
                    updateElement('step7-grad-w-h1-o', current_gradWH1O.toFixed(4));
                    updateElement('step7-w-h1-o-new', newWH1O.toFixed(4));

                    updateElement('step7-w-h2-o-old', oldWH2O.toFixed(4));
                    updateElement('step7-lr-2', current_lr.toFixed(1));
                    updateElement('step7-grad-w-h2-o', current_gradWH2O.toFixed(4));
                    updateElement('step7-w-h2-o-new', newWH2O.toFixed(4));

                    updateElement('step7-w-h3-o-old', oldWH3O.toFixed(4));
                    updateElement('step7-lr-3', current_lr.toFixed(1));
                    updateElement('step7-grad-w-h3-o', current_gradWH3O.toFixed(4));
                    updateElement('step7-w-h3-o-new', newWH3O.toFixed(4));

                    updateElement('step7-b-o-old', oldBO.toFixed(4));
                    updateElement('step7-lr-4', current_lr.toFixed(1));
                    updateElement('step7-grad-b-o', current_gradBO.toFixed(4));
                    updateElement('step7-b-o-new', newBO.toFixed(4));

                    const oldWI1H1 = networkState[0].neurons[0].weights[0];
                    const oldWI2H1 = networkState[0].neurons[1].weights[0];
                    const oldBH1 = networkState[1].neurons[0].bias;

                    const current_gradWI1H1 = networkState[0].neurons[0].weight_gradients[0];
                    const current_gradWI2H1 = networkState[0].neurons[1].weight_gradients[0];
                    const current_gradBH1 = networkState[1].neurons[0].bias_gradient;

                    const newWI1H1 = oldWI1H1 - current_lr * current_gradWI1H1;
                    const newWI2H1 = oldWI2H1 - current_lr * current_gradWI2H1;
                    const newBH1 = oldBH1 - current_lr * current_gradBH1;

                    networkState[0].neurons[0].weights[0] = newWI1H1;
                    networkState[0].neurons[1].weights[0] = newWI2H1;
                    networkState[1].neurons[0].bias = newBH1;

                    updateElement('step7-w-i1-h1-old', oldWI1H1.toFixed(4));
                    updateElement('step7-lr-5', current_lr.toFixed(1));
                    updateElement('step7-grad-w-i1-h1', current_gradWI1H1.toFixed(4));
                    updateElement('step7-w-i1-h1-new', newWI1H1.toFixed(4));

                    updateElement('step7-w-i2-h1-old', oldWI2H1.toFixed(4));
                    updateElement('step7-lr-6', current_lr.toFixed(1));
                    updateElement('step7-grad-w-i2-h1', current_gradWI2H1.toFixed(4));
                    updateElement('step7-w-i2-h1-new', newWI2H1.toFixed(4));

                    updateElement('step7-b-h1-old', oldBH1.toFixed(4));
                    updateElement('step7-lr-7', current_lr.toFixed(1));
                    updateElement('step7-grad-b-h1', current_gradBH1.toFixed(4));
                    updateElement('step7-b-h1-new', newBH1.toFixed(4));
                    break;
                case 8:
                    if (!networkState[0] || !networkState[1] || !networkState[2]) return;
                    updateElement('final-b-o', networkState[2].neurons[0].bias.toFixed(4));
                    updateElement('final-w-h1-o', networkState[1].neurons[0].weights[0].toFixed(4));
                    updateElement('final-w-h2-o', networkState[1].neurons[1].weights[0].toFixed(4));
                    updateElement('final-w-h3-o', networkState[1].neurons[2].weights[0].toFixed(4));
                    updateElement('final-b-h1', networkState[1].neurons[0].bias.toFixed(4));
                    updateElement('final-w-i1-h1', networkState[0].neurons[0].weights[0].toFixed(4));
                    updateElement('final-w-i2-h1', networkState[0].neurons[1].weights[0].toFixed(4));
                    break;
            }
        }
    }));

    function showExampleStep(stepIndex) {
        const allExampleSteps = DOMElements.exampleStepsContainer.querySelectorAll('.example-step');
        allExampleSteps.forEach((step, index) => {
            if (index === stepIndex) {
                step.classList.add('active');
                step.style.display = 'block';
            } else {
                step.classList.remove('active');
                step.style.display = 'none';
            }
        });

        if (exampleStepsData[stepIndex] && exampleStepsData[stepIndex].updateExample) {
            exampleStepsData[stepIndex].updateExample();
        }

        const activeStepElement = allExampleSteps[stepIndex];
        if (activeStepElement) {
            const highlights = activeStepElement.querySelectorAll('.highlight');
            highlights.forEach(span => {
                span.style.backgroundColor = 'yellow';
                setTimeout(() => {
                    span.style.backgroundColor = '';
                }, 1000);
            });
        }
    }

    async function executeMicroStep(stepFunction) {
        if (isProcessingStep) return;
        try {
            isProcessingStep = true;
            DOMElements.nextStepBtn.disabled = true;

            clearAllVisualCues();
            await stepFunction();
            currentMicroStep++;

            if (currentMicroStep < microSteps.length) {
                DOMElements.nextStepBtn.disabled = false;
            } else {
                updateExplanation(appContent.explanation.epochCompletedTitle, appContent.explanation.epochCompletedText);
                DOMElements.startBtn.disabled = false;
                DOMElements.nextStepBtn.disabled = true;
                showExampleStep(exampleStepsData.length - 1);
            }
        } finally {
            isProcessingStep = false;
            DOMElements.nextStepBtn.disabled = (currentMicroStep >= microSteps.length);
        }
    }

    function generateMicroSteps() {
        microSteps = [];

        microSteps.push(async () => {
            updateExplanation(appContent.microStepExplanations.forwardPropInputLayerTitle, appContent.microStepExplanations.forwardPropInputLayerText);
            showExampleStep(1);
            const inputLayer = networkState[0].neurons;
            const inputLayerElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === 0);
            for (let i = 0; i < inputLayer.length; i++) {
                inputLayer[i].activation = inputData[i];
                inputLayerElements[i].classList.add('neuron-active');
                displayValue(inputLayerElements[i], inputLayer[i].activation);
                await new Promise(r => setTimeout(r, 300));
                inputLayerElements[i].classList.remove('neuron-active');
            }
        });

        for (let l = 1; l < networkState.length; l++) {
            const currentLayer = networkState[l].neurons;
            const prevLayer = networkState[l - 1].neurons;
            const currentLayerElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === l);
            const prevLayerElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === l - 1);

            for (let n = 0; n < currentLayer.length; n++) {
                microSteps.push(async () => {
                    const layerName = l === 1 ? 'Hidden' : 'Output';
                    updateExplanation(
                        appContent.microStepExplanations.forwardPropLayerTitle(layerName, n + 1),
                        appContent.microStepExplanations.forwardPropLayerText
                    );
                    showExampleStep(l === 1 ? 1 : 2);

                    currentLayerElements[n].classList.add('neuron-active');
                    let weightedSum = 0;
                    for (let p = 0; p < prevLayer.length; p++) {
                        const weight = prevLayer[p].weights[n];
                        const prevActivation = prevLayer[p].activation;
                        const connectionElement = allConnectionsElements.find(c =>
                            parseInt(c.dataset.sourceLayer) === l - 1 &&
                            parseInt(c.dataset.sourceNeuron) === p &&
                            parseInt(c.dataset.targetLayer) === l &&
                            parseInt(c.dataset.targetNeuron) === n
                        );
                        if (connectionElement) {
                            connectionElement.classList.add('connection-active-forward');
                            displayValue(connectionElement, prevActivation * weight);
                        }
                        prevLayerElements[p].classList.add('neuron-active');
                        await new Promise(r => setTimeout(r, 200));
                        prevLayerElements[p].classList.remove('neuron-active');
                        if (connectionElement) connectionElement.classList.remove('connection-active-forward');
                        weightedSum += prevActivation * weight;
                    }

                    currentLayer[n].z = weightedSum + currentLayer[n].bias;
                    displayValue(currentLayerElements[n], currentLayer[n].z);
                    await new Promise(r => setTimeout(r, 500));

                    const activationFn = ActivationFunctions.sigmoid;
                    currentLayer[n].activation = activationFn(currentLayer[n].z);
                    displayValue(currentLayerElements[n], currentLayer[n].activation);
                    await new Promise(r => setTimeout(r, 500));
                    currentLayerElements[n].classList.remove('neuron-active');
                });
            }
        }

        microSteps.push(async () => {
            updateExplanation(appContent.microStepExplanations.calculateLossTitle, appContent.microStepExplanations.calculateLossText);
            showExampleStep(3);
        });

        microSteps.push(async () => {
            const outputLayer = networkState[networkState.length - 1].neurons;
            const outputLayerElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === networkState.length - 1);
            let predictions = outputLayer.map(n => n.activation);
            let loss = LossFunctions.mse(predictions, targetOutput);

            for (let i = 0; i < outputLayerElements.length; i++) {
                outputLayerElements[i].classList.add('neuron-active');
                displayValue(outputLayerElements[i], predictions[i]);
                await new Promise(r => setTimeout(r, 200));
            }
            updateExplanation(appContent.microStepExplanations.calculateLossResultTitle, appContent.microStepExplanations.calculateLossResultText(loss));
            await new Promise(r => setTimeout(r, 1000));
            outputLayerElements.forEach(n => n.classList.remove('neuron-active'));
        });

        microSteps.push(async () => {
            updateExplanation(appContent.microStepExplanations.backpropOutputDeltaTitle, appContent.microStepExplanations.backpropOutputDeltaText);
            showExampleStep(4);
            const outputLayerIndex = networkState.length - 1;
            const outputLayer = networkState[outputLayerIndex].neurons;
            const outputLayerElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === outputLayerIndex);

            for (let n = 0; n < outputLayer.length; n++) {
                outputLayerElements[n].classList.add('neuron-active');
                const activation = outputLayer[n].activation;
                const target = targetOutput[n];

                outputLayer[n].error = (target - activation) * ActivationFunctions.sigmoid_derivative(activation);
                displayValue(outputLayerElements[n], outputLayer[n].error);
                await new Promise(r => setTimeout(r, 300));
                outputLayerElements[n].classList.remove('neuron-active');
            }
        });

        for (let l = networkState.length - 2; l >= 1; l--) {
            const currentLayer = networkState[l].neurons;
            const nextLayer = networkState[l + 1].neurons;
            const currentLayerElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === l);
            const nextLayerElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === l + 1);

            for (let n = 0; n < currentLayer.length; n++) {
                microSteps.push(async () => {
                    const layerName = l === 1 ? 'Hidden' : `Hidden ${l}`;
                    updateExplanation(
                        appContent.microStepExplanations.backpropHiddenDeltaTitle(layerName, n + 1),
                        appContent.microStepExplanations.backpropHiddenDeltaText
                    );
                    showExampleStep(5);

                    currentLayerElements[n].classList.add('neuron-active');
                    let sumOfWeightedDeltas = 0;
                    for (let nextN = 0; nextN < nextLayer.length; nextN++) {
                        const weight = currentLayer[n].weights[nextN];
                        const nextDelta = nextLayer[nextN].error;
                        sumOfWeightedDeltas += nextDelta * weight;

                        const connectionElement = allConnectionsElements.find(c =>
                            parseInt(c.dataset.sourceLayer) === l &&
                            parseInt(c.dataset.sourceNeuron) === n &&
                            parseInt(c.dataset.targetLayer) === l + 1 &&
                            parseInt(c.dataset.targetNeuron) === nextN
                        );
                        if (connectionElement) {
                            connectionElement.classList.add('connection-active-backward');
                            displayValue(connectionElement, nextDelta * weight);
                            nextLayerElements[nextN].classList.add('neuron-active');
                            await new Promise(r => setTimeout(r, 200));
                            nextLayerElements[nextN].classList.remove('neuron-active');
                            connectionElement.classList.remove('connection-active-backward');
                        }
                    }

                    const activation = currentLayer[n].activation;
                    currentLayer[n].error = sumOfWeightedDeltas * ActivationFunctions.sigmoid_derivative(activation);
                    displayValue(currentLayerElements[n], currentLayer[n].error);
                    await new Promise(r => setTimeout(r, 500));
                    currentLayerElements[n].classList.remove('neuron-active');
                });
            }
        }

        microSteps.push(async () => {
            updateExplanation(appContent.microStepExplanations.calculateGradientsTitle, appContent.microStepExplanations.calculateGradientsText);
            showExampleStep(6);

            const outputLayerIndex = networkState.length - 1;
            const outputLayer = networkState[outputLayerIndex].neurons;
            const prevLayerOutput = networkState[outputLayerIndex - 1].neurons;
            const prevLayerOutputElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === outputLayerIndex - 1);

            for (let p = 0; p < prevLayerOutput.length; p++) {
                for (let n = 0; n < outputLayer.length; n++) {
                    const weightGradient = prevLayerOutput[p].activation * outputLayer[n].error;
                    prevLayerOutput[p].weight_gradients[n] = weightGradient;

                    const connectionElement = allConnectionsElements.find(c =>
                        parseInt(c.dataset.sourceLayer) === outputLayerIndex - 1 &&
                        parseInt(c.dataset.sourceNeuron) === p &&
                        parseInt(c.dataset.targetLayer) === outputLayerIndex &&
                        parseInt(c.dataset.targetNeuron) === n
                    );
                    if (connectionElement) {
                        connectionElement.classList.add('connection-active-backward');
                        displayValue(connectionElement, weightGradient);
                        prevLayerOutputElements[p].classList.add('neuron-active');
                        await new Promise(r => setTimeout(r, 100));
                        prevLayerOutputElements[p].classList.remove('neuron-active');
                        connectionElement.classList.remove('connection-active-backward');
                    }
                }
            }
            for (let n = 0; n < outputLayer.length; n++) {
                outputLayer[n].bias_gradient = outputLayer[n].error;
                const neuronElement = allNeuronsElements.find(ne => parseInt(ne.dataset.layerIndex) === outputLayerIndex && parseInt(ne.dataset.neuronIndex) === n);
                if (neuronElement) {
                    neuronElement.classList.add('neuron-active');
                    displayValue(neuronElement, outputLayer[n].bias_gradient);
                    await new Promise(r => setTimeout(r, 100));
                    neuronElement.classList.remove('neuron-active');
                }
            }

            const hiddenLayerIndex = 1;
            const hiddenLayer = networkState[hiddenLayerIndex].neurons;
            const inputLayer = networkState[0].neurons;
            const inputLayerElements = allNeuronsElements.filter(n => parseInt(n.dataset.layerIndex) === 0);

            for (let p = 0; p < inputLayer.length; p++) {
                for (let n = 0; n < hiddenLayer.length; n++) {
                    const weightGradient = inputLayer[p].activation * hiddenLayer[n].error;
                    inputLayer[p].weight_gradients[n] = weightGradient;

                    const connectionElement = allConnectionsElements.find(c =>
                        parseInt(c.dataset.sourceLayer) === 0 &&
                        parseInt(c.dataset.sourceNeuron) === p &&
                        parseInt(c.dataset.targetLayer) === hiddenLayerIndex &&
                        parseInt(c.dataset.targetNeuron) === n
                    );
                    if (connectionElement) {
                        connectionElement.classList.add('connection-active-backward');
                        displayValue(connectionElement, weightGradient);
                        inputLayerElements[p].classList.add('neuron-active');
                        await new Promise(r => setTimeout(r, 100));
                        inputLayerElements[p].classList.remove('neuron-active');
                        connectionElement.classList.remove('connection-active-backward');
                    }
                }
            }
            for (let n = 0; n < hiddenLayer.length; n++) {
                hiddenLayer[n].bias_gradient = hiddenLayer[n].error;
                const neuronElement = allNeuronsElements.find(ne => parseInt(ne.dataset.layerIndex) === hiddenLayerIndex && parseInt(ne.dataset.neuronIndex) === n);
                if (neuronElement) {
                    neuronElement.classList.add('neuron-active');
                    displayValue(neuronElement, hiddenLayer[n].bias_gradient);
                    await new Promise(r => setTimeout(r, 100));
                    neuronElement.classList.remove('neuron-active');
                }
            }
        });

        microSteps.push(async () => {
            updateExplanation(appContent.microStepExplanations.updateWeightsBiasesTitle, appContent.microStepExplanations.updateWeightsBiasesText);
            showExampleStep(7);
        });

        microSteps.push(async () => {
            for (let l = 0; l < networkState.length - 1; l++) {
                const currentLayer = networkState[l].neurons;
                const nextLayer = networkState[l + 1].neurons;

                for (let p = 0; p < currentLayer.length; p++) {
                    for (let n = 0; n < nextLayer.length; n++) {
                        const oldWeight = currentLayer[p].weights[n];
                        const weightGradient = currentLayer[p].weight_gradients[n];
                        const newWeight = oldWeight - config.learningRate * weightGradient;
                        currentLayer[p].weights[n] = newWeight;

                        const connectionElement = allConnectionsElements.find(c =>
                            parseInt(c.dataset.sourceLayer) === l &&
                            parseInt(c.dataset.sourceNeuron) === p &&
                            parseInt(c.dataset.targetLayer) === l + 1 &&
                            parseInt(c.dataset.targetNeuron) === n
                        );
                        if (connectionElement) {
                            connectionElement.classList.add('connection-weight-update');
                            displayValue(connectionElement, newWeight);
                            await new Promise(r => setTimeout(r, 100));
                            connectionElement.classList.remove('connection-weight-update');
                        }
                    }
                }
            }

            for (let l = 1; l < networkState.length; l++) {
                const currentLayer = networkState[l].neurons;
                const currentLayerElements = allNeuronsElements.filter(ne => parseInt(ne.dataset.layerIndex) === l);
                for (let n = 0; n < currentLayer.length; n++) {
                    const oldBias = currentLayer[n].bias;
                    const biasGradient = currentLayer[n].bias_gradient;
                    const newBias = oldBias - config.learningRate * biasGradient;
                    currentLayer[n].bias = newBias;

                    const neuronElement = currentLayerElements[n];
                    if (neuronElement) {
                        neuronElement.classList.add('neuron-active');
                        displayValue(neuronElement, newBias);
                        await new Promise(r => setTimeout(r, 100));
                        neuronElement.classList.remove('neuron-active');
                    }
                }
            }
        });
    }

    function startSingleEpoch() {
        if (isProcessingStep) return;
        resetVisualization();
        generateMicroSteps();
        currentMicroStep = 0;
        DOMElements.startBtn.disabled = true;
        DOMElements.nextStepBtn.disabled = false;
        setTimeout(() => executeMicroStep(microSteps[currentMicroStep]), 50);
    }

    async function nextStep() {
        if (isProcessingStep) return;
        DOMElements.nextStepBtn.disabled = true;
        if (currentMicroStep < microSteps.length) {
            await executeMicroStep(microSteps[currentMicroStep]);
        }
    }

    function resetVisualization() {
        currentMicroStep = 0;
        isProcessingStep = false;
        initializeNetworkLayout();
        clearAllVisualCues();
        DOMElements.startBtn.disabled = false;
        DOMElements.nextStepBtn.disabled = true;
        updateExplanation(appContent.explanation.welcomeTitle, appContent.explanation.welcomeText);
        showFullNetworkView();
        showExampleStep(0);
    }

    function setupEventListeners() {
        DOMElements.startBtn.addEventListener('click', startSingleEpoch);
        DOMElements.nextStepBtn.addEventListener('click', nextStep);
        DOMElements.resetBtn.addEventListener('click', resetVisualization);

        window.addEventListener('resize', () => {
            clearTimeout(window.resizeTimer);
            window.resizeTimer = setTimeout(() => {
                initializeNetworkLayout();
            }, 250);
        });

        DOMElements.exampleStepsContainer.addEventListener('click', (event) => {
            const clickedStep = event.target.closest('.example-step');
            if (clickedStep) {
                const stepIndex = Array.from(DOMElements.exampleStepsContainer.children).indexOf(clickedStep);
                if (stepIndex !== -1) {
                    showExampleStep(stepIndex);
                }
            }
        });
    }

    function initializeApp() {
        initializeNetworkLayout();
        setupEventListeners();
        resetVisualization();
    }

    initializeApp();
});
