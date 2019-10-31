clear; close all; clc

%% Performing clustering with only a "Competitive Layer" (based on LVQ)
% Competitive learning is a form of unsupervised learning in artificial 
% neural networks, in which nodes compete for the right to respond to a 
% subset of the input data. As a variant of Hebbian learning, competitive 
% learning works by increasing the specialization of each node in the 
% network. It is well suited to finding clusters within data. (from Wikipedia)
% 
% By the word "only", we mean that a one-layer neural net is the only thing
% that represents our model. For this pupose, the number of clusters should
% be given to the "competlayer" function as the first input argument.

%% Part 1- Creating a fake dataset
nRow_fake_dataset = 3000; % it should actually be 30000, but for elapsed 
% running time of the code, 3000 is chosen.  
nCol_fake_dataset = 600; % it should actually be 6000, but for elapsed 
% running time of the code, 600 is chosen.

temp0 = randn(ceil(nRow_fake_dataset/5), ceil(nCol_fake_dataset/5));
temp1 = repmat(temp0, 5, 5);
fake_dataset = temp1(1:nRow_fake_dataset, 1:nCol_fake_dataset);

%% Part 2- Performing linear dimensionality reduction (or noise suppression) alg.
[coeff, score, ~, ~, explained, ~] = pca(fake_dataset);
explained_variance_threshold = 99.5; % for the FFT dataset, a threshold of 
% 99 or 99.5 is thought to be more suitable.

cumulative_explained = cumsum(explained);
numFeatures = find(cumulative_explained >= explained_variance_threshold);
numFeatures = numFeatures(1);

reduced_dataset_by_pca = score(:, 1:numFeatures);
size(reduced_dataset_by_pca)

%% Part 3- Performing LVQ-based clustering 
%% Part 3-1- Preparing (transposing) the dataset matrix
inputs = reduced_dataset_by_pca.';

%% Part 3-2- Creating a "network" object and configuring its parameters
numClusters = 3;  numEpochs = 100;
net = competlayer(numClusters);
configure(net, inputs);
net.trainParam.epochs = numEpochs;

%% Part 3-3- Training the network and clustering the dataset
[net, tr] = train(net, inputs,'CheckpointFile', ...
    'compete_layer_chk', 'CheckpointDelay', 5);

view(net)

oneHotClasses = net(inputs);
classes = vec2ind(oneHotClasses);

%% Part 3-4- Visualization of the cluster proposals for some samples
numOfSamples = 5;
disp(['Proposals for the first ' num2str(numOfSamples) ' samples:'])
classes(1:numOfSamples)