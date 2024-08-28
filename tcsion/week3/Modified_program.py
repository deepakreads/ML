import numpy as np 


class LinearRegression() : 
	
	def __init__( self, learning_rate, iterations ) : 
		self.learning_rate = learning_rate 
		self.iterations = iterations 
		3210
	# Function for model training 
	def fit( self, X, Y ) : 
		
		# no_of_training_examples, no_of_features 
		self.m, self.n = X.shape 
		
		# weight initialization 
		self.W = np.ones( self.n ) 
		
		self.b = 0
		self.X = X 
		self.Y = Y 
		print(f'Starting weights {self.W} ,bias {self.b}')
		Y_pred = self.predict( self.X )
		print(f' Predictions {np.round(Y_pred,2)}')
		e = self.Y - Y_pred
		se = e ** 2
		mse = np.sum(se) / self.m 
		print(f' Error {e}')
		print(f' Sqaured error {se}')
		print(f' Mean sqaured error {mse}')
		# gradient descent learning 
		for i in range( self.iterations ) : 
			print(f'\n\nIteration {i} \n Current weights : {self.W}')
			self.update_weights() 
			print(f'Updated weights : {self.W}')
			print(f'Updated bias : {self.b}')
 
		return self
	
	# Helper function to update weights in gradient descent 
	
	def update_weights( self ) : 

		Y_pred = self.predict( self.X ) 
		print(f' Predictions {np.round(Y_pred,2)}')
		e = self.Y - Y_pred
		se = e ** 2
		mse = np.sum(se) / self.m 
		print(f' Error {e}')
		print(f' Sqaured error {se}')
		print(f' Mean sqaured error {mse}')

		# calculate gradients 
		dW = - (  ( self.X.T ).dot( self.Y - Y_pred ) ) / self.m 
		db = -  np.sum( self.Y - Y_pred ) / self.m 
		print(f' gradients - dW {dW} \n db {db}')
		print()
		# update weights 
		self.W = self.W - self.learning_rate * dW 
		self.b = self.b - self.learning_rate * db 
		return self

	def predict( self, X ) : 
		return X.dot( self.W ) + self.b 


def main() : 

  # data
  X_train = np.array([[60,22],[67,24],[71,15],[75,20],[78,16]])
  Y_train = np.array([140,159,192,200,212])
  
  # Model training 
  model = LinearRegression( iterations = 2, learning_rate = 0.00002 ) 
  model.fit(X_train, Y_train ) 
	
if __name__ == "__main__":
    main()