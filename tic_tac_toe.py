import numpy as np
import pickle

BOARD_ROWS = 3;
BOARD_COLUMNS = 3;

class bot:
	def __init__(self,name,exp_rate=0.3,discount_factor=0.9,learning_rate=0.2):
		self.player_name = name;						#Name of the player
		self.history = [];    							#History of the game
		self.exp_rate = exp_rate;						#Exploration rate of the player
		self.discount_factor = discount_factor;			#Discount factor of the gain
		self.action_value = {};							#Estimated action value of the player
		self.learning_rate = learning_rate				#Learning rate of the player


	#Hash to save the state of the game in history
	def getHash(self,board):
		hash = str(board.reshape(BOARD_COLUMNS*BOARD_ROWS));
		return hash;


	def chooseAction(self,positions,current_board,player_symbol):
		if(np.random.uniform(0,1) <= self.exp_rate):
			#Exploring moves
			idx = np.random.choice(len(positions));
			action = positions[idx];
		else:
			max_val = -1000;
			for p in positions:
				next_board = current_board.copy();
				next_board[p] = player_symbol;
				next_boardHash = self.getHash(next_board);
				val = 0 if self.action_value.get(next_boardHash) is None else self.action_value.get(next_boardHash);
				if(val >= max_val):
					max_val = val;
					action = p;

		return action;


	#Append a hash state to the history
	def addHistory(self,recent_eventHash):
			self.history.append(recent_eventHash);


	def feedReward(self,reward):
		for hist in reversed(self.history):
			if(self.action_value.get(hist)) is None:
				self.action_value[hist] = 0;
			self.action_value[hist] += self.learning_rate * (self.discount_factor * reward - self.action_value[hist]);
			reward = self.action_value[hist];

	def resetPlayer(self):
		self.history = [];


	def savePolicy(self):
		with open('policy_'+str(self.player_name),'wb') as policy:
			pickle.dump(self.action_value,policy);

	def loadPolicy(self,file):
		with open(file,'rb') as policy:
			self.action_value = pickle.load(policy);

class State:
	def __init__(self,player1,player2):
		self.board = np.zeros((BOARD_ROWS,BOARD_COLUMNS));
		self.player1 = player1;
		self.player2 = player2;
		self.isEnd = False;
		self.boardHash = None;
		self.playerSymbol = 1					#Player1 plays first player1 symbol is 1 and player2 symbol is -1;

	def getHash(self):
		self.boardHash = str(self.board.reshape(BOARD_ROWS*BOARD_COLUMNS));
		return self.boardHash;

	def isGameOver(self):
		#Rows match
		for i in range(BOARD_ROWS):
			if (sum(self.board[i,:]) == 3):
				self.isEnd = True;
				return 1;
			if (sum(self.board[i,:]) == -3):
				self.isEnd = True;
				return -1;
		#Columns match
		for i in range(BOARD_COLUMNS):
			if (sum(self.board[:,i] == 3)):
				self.isEnd = True;
				return 1;
			if (sum(self.board[:,i]) == -3):
				self.isEnd = True;
				return -1;
		#Diagonal match
		principle_diag = sum(self.board[i,i] for i in range(BOARD_COLUMNS));
		off_diag = sum(self.board[i,BOARD_COLUMNS -i -1] for i in range(BOARD_COLUMNS));
		if(principle_diag == 3 or off_diag == 3):
			self.isEnd = True;
			return 1;
		if(principle_diag == -3 or off_diag == -3):
			self.isEnd = True;
			return -1;

		#Tie or No moves left
		if(len(self.available_positions()) == 0):
			self.isEnd = True;
			return 0;

		#Game is still on
		self.isEnd  = False;
		return None;


	def available_positions(self):
		positions = [];
		for i in range(BOARD_ROWS):
			for j in range(BOARD_COLUMNS):
				if(self.board[(i,j)] == 0):
					positions.append((i,j));
		return positions;

	def updateState(self,position):
		self.board[position] = self.playerSymbol;			#Make the move
		self.playerSymbol = self.playerSymbol * -1;		#Give turn to next player

	def giveReward(self):
		winner = self.isGameOver();
		if (winner == 1):
			self.player1.feedReward(1);
			self.player2.feedReward(0);
		elif (winner == -1):
			self.player1.feedReward(0);
			self.player2.feedReward(1);
		else:	
			self.player1.feedReward(0.1);
			self.player2.feedReward(0.5);

	#Gives reward to the bot for playing with human players
	def giveRewardHuman(self):
		winner = self.isGameOver();
		if (winner == 1):
			self.player1.feedReward(1);
		elif (winner == -1):
			self.player1.feedReward(0);
		else:	
			self.player1.feedReward(0.5);

	def resetBoard(self):
		self.board = np.zeros((BOARD_ROWS,BOARD_COLUMNS));
		self.isEnd = False;
		self.boardHash = None;
		self.playerSymbol = 1;


	#Train the computer to play tic-tac-toe
	def train(self,rounds = 100):
		for i in range(rounds):
			while (not self.isEnd):
				positions = self.available_positions();
				player1_action = self.player1.chooseAction(positions,self.board,self.playerSymbol);
				self.updateState(player1_action);
				board_hash = self.getHash();
				self.player1.addHistory(board_hash);

				winner = self.isGameOver();
				if winner is not None:						#Implies either Player1 wins or ends in draw
					self.giveReward();
					self.player1.resetPlayer();
					self.player2.resetPlayer();
					self.resetBoard();
					break;
				else:
					#Player2's turn now
					positions = self.available_positions();
					player2_action = self.player2.chooseAction(positions,self.board,self.playerSymbol);
					self.updateState(player2_action);
					board_hash = self.getHash();
					self.player2.addHistory(board_hash);

					winner = self.isGameOver();
					if (winner is not None):				#Implies either Player2 wins or ends in a draw
						self.giveReward();
						self.player1.resetPlayer();
						self.player2.resetPlayer();
						self.resetBoard();
						break;
			if (i%1000 == 0 and i != 0):
				print ("{0} rounds trained...".format(i));


	#Play with human with added learning from the game
	def play(self):
		while (not self.isEnd):
			#Computer plays first
			positions = self.available_positions();
			player1_action = self.player1.chooseAction(positions,self.board,self.playerSymbol);
			self.updateState(player1_action);
			self.showBoard();

			winner = self.isGameOver();
			if (winner is not None):
				if (winner == 1):
					print(self.player1.player_name,"wins!!!");
					self.giveRewardHuman();
					self.player1.resetPlayer();
				else:
					print("Tie");
					self.giveRewardHuman();
					self.player1.resetPlayer();
				self.resetBoard();
				self.player1.savePolicy();
				break;
			else:
				positions = self.available_positions();
				player2_action = self.player2.chooseAction(positions);
				self.updateState(player2_action);
				self.showBoard();

				winner = self.isGameOver();
				if (winner is not None):
					if (winner == -1):
						print(self.player2.player_name,"wins!!!");
						self.giveRewardHuman();
						self.player1.resetPlayer();
					else:
						print("Tie");
						self.giveRewardHuman();
						self.player1.resetPlayer();
					self.resetBoard();
					self.player1.savePolicy();
					break;
			self.isGameOver();

	def showBoard(self):
	    # player1: X  player2: O
	    for i in range(0, BOARD_ROWS):
	        print('-------------')
	        out = '| '
	        for j in range(0, BOARD_COLUMNS):
	            if self.board[i, j] == 1:
	                token = 'X'
	            if self.board[i, j] == -1:
	                token = 'O'
	            if self.board[i, j] == 0:
	                token = ' '
	            out += token + ' | '
	        print(out)
	    print('-------------')
	    #print(self.board);

class Human:
	def __init__(self,name):
		self.player_name = name;

	def chooseAction(self, positions):
	    while True:
	        tup = input("Input your action, i.e (row col):")
	        (row,col) = (int(tup.split(' ')[0]),int(tup.split(' ')[1]));
	        action = (row-1, col-1)
	        if action in positions:
	            return action



if __name__ == "__main__":

	#Train the bot
	your_name = input("Enter your name: ");

	check = input("Do you have a bot already trained and wanna continue playing with it?(y/n): ");
	if(check == 'y'):
		bot_policy = input("Enter the name of the policy file of the bot you have trained before: ");
		tictac = bot(str(bot_policy.split("_")[1]));
		tictac.loadPolicy(bot_policy);

		human = Human(your_name);

		st = State(tictac,human);
		st.play();
	else:
		name = input("Enter the name of the bot: ");
		# exp_rate = float(input("Enter the exploration rate of the bot: "));
		# dis_fac = float(input("Enter the discount factor for the bot: "));
		# lr = float(input("Enter the learning rate of the bot: "));
		player1 = bot(name,0.3,0.9,0.2);
		player2 = bot("Trainer_bot");

		st = State(player1,player2);
		rounds = int(input("Enter the number of rounds of training for the bot: "));
		print("Training bot for you...");
		st.train(rounds);
		print("Training Complete!!!\nYou are on now...");
		player1.savePolicy();


		#Play with human
		tictac = bot(name);
		tictac.loadPolicy("policy_"+name);

		human = Human(your_name);

		st = State(tictac,human);
		st.play();

