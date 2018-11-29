import numpy as np
import math
import io
import sys
import subprocess
import time
from sklearn import preprocessing
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import collections



ps = PorterStemmer()
input_file_name = sys.argv[1]	# The name of the input file is being taken from the command line argument 

message_mapping = {}
message_mapping['spam'] = 0
message_mapping['ham'] 	= 1
frequency_of_words = {}



number_of_lines  = 0										# Number of lines in the input file
train_data_lines = 0										# Number of training messages
test_data_lines  = 0										# Number of test messages
x_train_data     = []										# Contains the training messages
y_train_data     = []										# Contains whether a training message is SPAM or a HAM
x_test_data      = []										# Contains the testing messages
y_test_data      = []										# Contains whether a test message is SPAM or a HAM 
distinct_words   = []										# Contains the distinct words after stopword removal and stemming
train_encoded_messages   = []								# Contains the messages of the training data in one hot encoded form.Each element is a binary vector.
train_encoded_message    = []								# This vector contains a single training messasge in encoded form
test_encoded_messages    = []								# Contains the messages of the test data in one hot encoded form. Each element is a binary vector.
test_encoded_message     = []								# This vector contains a single test messasge in encoded form
num_input_layer_neurons  = 0								# It is the number of neurons in the input layer.
num_first_layer_neurons  = 100								# It is the number of neurons in the 1st hidden layer
num_second_layer_neurons = 50								# It is the number of neurons in the 2nd hidden layer
num_output_layer_neurons = 1								# It is the number of neurons in the output layer
weight_matrix_1		     = np.zeros(shape = (10,10))		# Weight Matrix between input layer and first hidden layer. Dimensions updated once number of distinct words have been found out.
weight_matrix_2  		 = np.zeros(shape = (100,50))		# Weight Matrix between first hidden layer and second hidden layer
weight_matrix_3          = np.zeros(shape = (50,1))	 		# Weight Matrix between second hidden layer and output layer
input_layer_neurons 	 = np.zeros(10)						# Vector for storing the incoming input layer data. Dimensions upadted once number of distinct words have been found out.
first_layer_neurons      = np.zeros(100)					# Vector for storing the activated outputs of the first hidden layer					
second_layer_neurons     = np.zeros(50)						# Vector for storing the activated outputs of the second hidden layer
output_layer_neurons     = np.zeros(1)
predicted_output 		 = 0								# Output of the neural network
learning_rate			 = 0.1								# Learning Rate used in backpropagation algorithm
delta_1					 = np.zeros(100)					# Vector for storing delta values for the first hidden layer
delta_2					 = np.zeros(50)						# Vector for storing delta values for the second hidden layer
delta_3					 = np.zeros(1)						# Vector for storing delta values for the output layer.
X_train = []												# Vector to store number of iterations
Y_train = []												# Vector to store the in sample error
X_test  = []												# Vector to store the number of iterations
Y_test  = []												# Vector to store the out of sample error
threshold = 0.5												# Threshold used for classifying into SPAM and HAM






def sigmoid( x ,derivative = False ):

	if derivative:
		result = x * ( 1 - x )
		return result

	else:
		return 1/ (1 + np.exp(-x))




# remove_stopwords() takes a string as input and produces a filtered string with stopwords and stemmed.

def remove_stopwords( message ):

	global frequency_of_words

	stop_words        =  list(stopwords.words('english'))
	punctuation_list  =  [".",";","{","}","[","]","(",")","!","@","#","-","_","--",",","''","?","...",":","&","'",">","<","*"]
	stop_words.extend(punctuation_list)

	filtered_string = ""
	words = word_tokenize(message)


	for word in words :							# If the word is not one amongst the stopwords, then it is used to create the filtered string
		if word not in stop_words:
			word  = ps.stem(word)
			filtered_string += word + " "


	filtered_string = filtered_string.encode("ascii","ignore").rstrip()


	for item in filtered_string.split(" "):

		if frequency_of_words.get(item) is None:
			frequency_of_words[item] =  1

		else:
			frequency_of_words[item] += 1

	return  filtered_string





# perform_one_hot_encoding() performs the one hot encoding of the training and the test messages.

def perform_one_hot_encoding():

	global train_encoded_message,train_encoded_messages,test_encoded_message,test_encoded_messages,x_train_data,x_test_data


	for message in x_train_data:								# One hot encoding for training messages is being done here

		train_encoded_message = []
		words = message.split()

		for word in distinct_words:
			if word in words:
				train_encoded_message.append(1)					# Each encoded message is a vector of size |V|, where |V| is the number of distinct words
			else:
				train_encoded_message.append(0)

		train_encoded_messages.append(train_encoded_message)



	for message in x_test_data:									# One hot encoding for test messages is being done here

		test_encoded_message = []
		words = message.split()

		for word in distinct_words:								# Each encoded message is a vector of size |V|, where |V| is the number of distinct words
			if word in words:
				test_encoded_message.append(1)
			else:
				test_encoded_message.append(0)

		test_encoded_messages.append(test_encoded_message)






def forward_pass_sigmoid( message_number ,testing = False ):

	global input_layer_neurons,first_layer_neurons,second_layer_neurons,weight_matrix_1,weight_matrix_2,weight_matrix_3,output_layer_neurons,predicted_output

	# Beginning of forward pass

	if testing:
		input_layer_neurons = test_encoded_messages[message_number]					# Inputting the encoded messages from the input layer for test data 
	else:
		input_layer_neurons = train_encoded_messages[message_number]				# Inputting the encoded messages from the input layer for training data 


	first_layer_neurons = np.dot(np.transpose(input_layer_neurons),weight_matrix_1)
	first_layer_neurons = sigmoid(first_layer_neurons)


	second_layer_neurons = np.dot(first_layer_neurons,weight_matrix_2)
	second_layer_neurons = sigmoid(second_layer_neurons)


	output_layer_neurons = np.dot(second_layer_neurons,weight_matrix_3)
	predicted_output     = sigmoid(output_layer_neurons[0])						# This is the output obtained from the neural network after each forward pass		

	return predicted_output






def backward_pass_sigmoid( predicted_output , message_number ):

	global delta_1,delta_2,delta_3,weight_matrix_1,weight_matrix_2,weight_matrix_3,first_layer_neurons,second_layer_neurons,input_layer_neurons,num_input_layer_neurons,num_first_layer_neurons,num_second_layer_neurons,num_output_layer_neurons 

	# Beginning of a backward pass

	# Calculating the delta values

	actual_output = message_mapping[y_train_data[message_number]]
	delta_3[0] = 2 * ( predicted_output - actual_output ) * sigmoid( predicted_output , True ) 		 # Delta values of last layer calculated



	delta_2 = np.dot(weight_matrix_3,delta_3)
	second_layer_neurons = sigmoid(second_layer_neurons,True)
	delta_2 = second_layer_neurons * delta_2



	delta_1 = np.dot(weight_matrix_2,delta_2)
	first_layer_neurons = sigmoid(first_layer_neurons,True)							# Delta values of first layer calculated
	delta_1 = first_layer_neurons * delta_1


	# Delta value calculation finished

	# Backward pass completed


	# Updating the weight matrices

	for i in range(num_input_layer_neurons):
		for j in range(num_first_layer_neurons):
			weight_matrix_1[i][j] -= learning_rate * input_layer_neurons[i] * delta_1[j]		# Updating weight matrix_1



	for i in range(num_first_layer_neurons):
		for j in range(num_second_layer_neurons):
			weight_matrix_2[i][j] -= learning_rate * first_layer_neurons[i] * delta_2[j]		# Updating weight matrix_2



	for i in range(num_second_layer_neurons):
		for j in range(num_output_layer_neurons):
			weight_matrix_3[i][j] -= learning_rate * second_layer_neurons[i] * delta_3[j]		# Updating weight matrix_3

	# Updation of weight matrices completed	
	










def create_neural_network_sigmoid( ):			# Neural Network with sigmoid function as the activation function

	global num_input_layer_neurons,input_layer_neurons,weight_matrix_1,weight_matrix_2,weight_matrix_3,X_train,Y_train,X_test,Y_test,train_data_lines,predicted_output,threshold,test_data_lines

	num_input_layer_neurons = len(distinct_words)
	input_layer_neurons     = np.zeros(num_input_layer_neurons)
	weight_matrix_1         = np.random.rand(num_input_layer_neurons,num_first_layer_neurons)  # Dimensions of weight matrix 1 updated, once the number of distinct words have been found
	weight_matrix_2         = np.random.rand(num_first_layer_neurons,num_second_layer_neurons)
	weight_matrix_3         = np.random.rand(num_second_layer_neurons,num_output_layer_neurons)	
	in_sample_error	= 0


	for iterations in range(10):

		in_sample_error	 = 0
		out_sample_error = 0 

		for message_number in range(train_data_lines):

			predicted_output = forward_pass_sigmoid( message_number )
			backward_pass_sigmoid( predicted_output , message_number )	


	#************************************ In Sample Error Calculation ********************************#	

		for training_message in range(train_data_lines):
			output 	= forward_pass_sigmoid( training_message )
			
			if ( output > threshold ):
				output = 1
			else:
				output = 0
			
			actual_output    = message_mapping[y_train_data[training_message]]
			in_sample_error  +=  ( actual_output - output )**2


		X_train.append(iterations)
		Y_train.append(in_sample_error)

	#**************************************************************************************************#

	#************************************ Out Sample Error Calculation ********************************#


		for test_message in range(test_data_lines):
			output 	= forward_pass_sigmoid( test_message ,True )

			if ( output > threshold ):
				output = 1
			else:
				output = 0

			actual_output    = message_mapping[y_test_data[test_message]]
			out_sample_error  +=  ( actual_output - output )**2


		X_test.append(iterations)
		Y_test.append(out_sample_error)


	#**************************************************************************************************#


	plt.plot(X_train,Y_train,marker = 'o',markersize = 5, markerfacecolor = 'red')
	plt.xlabel('Iteration Number')
	plt.ylabel('In Sample Error ')
	plt.title('Plot for Sigmoid Activation Function')
	plt.show()
	
	
	plt.plot(X_test,Y_test,marker = 'o',markersize = 5, markerfacecolor = 'red')
	plt.xlabel('Iteration Number')
	plt.ylabel('Out Sample Error ')
	plt.title('Plot for Sigmoid Activation Function')
	plt.show()



			
command    = "wc -l " + input_file_name
proc       = subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)
(out, err) = proc.communicate()
out        = out.rstrip('\n')


number_of_lines  = int(out.split(" ",1)[0])
train_data_lines = int(0.8 * number_of_lines)
test_data_lines  = number_of_lines - train_data_lines
input_file 	     = io.open(input_file_name,'r',encoding = 'utf8')
count = 0


ticks = time.time()
print 'Started Reading Input File...'

for line in input_file:

	line = line.split("\t",1)

	if ( count < train_data_lines ):								# First 80% of the messages is being treated as training data 

		x_train_data.append(remove_stopwords(line[1].strip("\n")))
		y_train_data.append(line[0])

		count += 1

	else:

		x_test_data.append(remove_stopwords(line[1].strip("\n")))	# Last 20% of the messages is being trated as test data
		y_test_data.append(line[0])



print 'File Reading Complete...'
print 'Time taken:   ',time.time()-ticks
ticks = time.time()


od = collections.OrderedDict(sorted(frequency_of_words.items(), key = lambda x:x[1] , reverse = True ))
count = 0 


for item in od:
	if ( count < 2000 ):
		distinct_words.append(item)
		count += 1


print 'One Hot Encoding Started...'
perform_one_hot_encoding()


print 'One Hot Encoding Complete...'
print 'Time taken:   ',time.time()-ticks
ticks = time.time()



print 'Creating Neural Network Using Sigmoid Function...'
create_neural_network_sigmoid()
print 'Time taken:   ',time.time()-ticks
ticks = time.time()














