/* Copyright 2012 David Pearson.
 * BSD License.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * An AI that uses Monte Carlo Tree Search to play Hex.
 *
 * @author David Pearson
 */
public class MCAI extends AI {
	private int aiplayer = 1;
	private int minLen = 49;
	private Location lastPlayed;
	public int diffLevel = 75;

	/**
	 * The default constructor. Assumes that the player is 1.
	 */
	public MCAI() {
	}

	/**
	 * Creates a new instance of MCAI.
	 *
	 * @param player The color to play as (see Constants.java)
	 */
	public MCAI(int player) {
		aiplayer = player;
	}

	/**
	 * Gets the color this AI is playing as.
	 *
	 * @return The color that the AI is playing as (see Constants.java)
	 */
	public int getPlayerCode() {
		return aiplayer;
	}

	// copy board
	private int[][] copy(int[][] board) {
		int length = board.length;
		int[][] copy = new int[length][length];
		for (int x = 0; x < length; x++) {
			for (int y = 0; y < length; y++) {
				copy[x][y] = board[x][y];
			}
		}

		return copy;
	}

	// finds n minimum values in array "a"
	private void mins(double[] a, double[] m, int[] i, int index) {
		// find maximum in m
		double max = -Double.MAX_VALUE;
		int nth = 0;
		for (int n = 0; n < m.length; n++) {
			if (m[n] > max) {
				max = m[n];
				nth = n;
			}
		}

		// find minimum in a
		if (a[index] < max) {
			m[nth] = a[index];
			i[nth] = index;
		}

		index += 1;

		if (!(index < a.length)) {
			return;
		}

		mins(a, m, i, index);
	}

	// finds n maximum values in array "a"
	private void maxs(double[] a, double[] m, int[] i, int index) {
		// find minimum in m
		double min = Double.MAX_VALUE;
		int nth = 0;
		for (int n = 0; n < m.length; n++) {
			if (m[n] < min) {
				min = m[n];
				nth = n;
			}
		}

		// find maximum in a
		if (a[index] > min) {
			m[nth] = a[index];
			i[nth] = index;
		}

		index += 1;

		if (!(index < a.length)) {
			return;
		}

		maxs(a, m, i, index);
	}

	// We don't need a heuristic function to determine whether a player has won or
	// not. It's unnecessary and impractical. We will use the playerWon() deductive
	// function instead.
	// The "calcVal" function still returns a double value to avoid breaking the
	// code that relies on it.
	public double calcVal(int[][] board) { // overrides the original calcVal()
//		double value = calcVal_(board); // sometimes cannot find the winner
//		System.out.println("calcVal: " + value);
//		return value;

		if (playerWon(board, aiplayer)) {
			return Double.MAX_VALUE;
		}

		return -Double.MAX_VALUE;
	}

	private boolean traverseChain(int[][] visited, int[][] board, int player, Location cell) {
		int x = cell.x;
		int y = cell.y;

		// game on
		if (visited[y][x] == player || board[y][x] != player) {
			return false;
		}

		// game over
		if (y == 6 && player == 1) { // white border
			return true;
		} else if (x == 6 && player == 2) { // black border
			return true;
		}

		// visit cell
		visited[y][x] = player;

		// visit cell's neighbors
		for (Location neigbour : cell.getAdjacentLocations()) {
			if (traverseChain(visited, board, player, neigbour)) {
				return true;
			}
		}

		return false;
	}

	private boolean playerWon(int[][] board, int player) {
//		System.out.println("aiplayer: " + aiplayer + " game over?");

		if (player == 0) { // new MCAI(colour).calcVal(board)>Math.pow(board.length, 2)
			player = 2;
		}

//		System.out.println("Player: " + player + " game over?");

		int length = board.length;
		int i;
		int[][] visited;
		Location cell = null;
		for (i = 0; i < length; i++) {
			// declare inside the for loop because chains may cross
			visited = new int[length][length]; // board to mark visited cells

			if (player == 1) { // 1: white
				cell = new Location(i, 0); // white: traverse over board[0][i]
			} else if (player == 2) { // 2: black
				cell = new Location(0, i); // black: traverse over board[i][0]
			}

			if (traverseChain(visited, board, player, cell)) {
//				System.out.println("Player: " + player + " WONN!!!");

				return true;
			}
		}

		return false;
	}

	// play random games
	private double random(int[][] board, int player) {
		int wins = 0;

		// find how many empty board cells are there and fill them
		int length = board.length;
		ArrayList<Location> empty = new ArrayList<>();
		for (int x = 0; x < length; x++) {
			for (int y = 0; y < length; y++) {
				if (board[x][y] == 0) {
					empty.add(new Location(y, x));
				}
			}
		}

		// play random games
		int[][] G;
		int turn;
		int x;
		int y;
		for (int i = 0; i < diffLevel; i++) {
			G = copy(board); // copy board
			turn = player;

			// shuffle empty cells before iterating and filling them
			Collections.shuffle(empty);
			for (Location e : empty) {
//				System.out.println("fill: " + turn);
				x = e.x;
				y = e.y;

				G[y][x] = turn;

				if (turn == 1) {
					turn = 2;

				} else {
					turn = 1;
				}
			}

			if (playerWon(G, aiplayer)) {
				wins += 1;
			}
		}

		return ((double) wins / (double) diffLevel);
	}

	// We could build game tree in minimax and negamax, but it is very slow.
	// simulate game states
	private void buildGameTree(TreeNode parent, int turn, int depth) {
		parent.children = new ArrayList<>();
		int[][] board = parent.board;

		TreeNode node;
		int length = board.length;
		for (int x = 0; x < length; x++) {
			for (int y = 0; y < length; y++) {
				if (board[x][y] == 0) {

					node = new TreeNode();

					node.parent = parent;
					node.playLocation = new Location(y, x); // board[x][y]

					node.board = copy(board);
					node.board[x][y] = turn; // play

					parent.children.add(node);

					if (depth > 1) {
						// opposite player's turn
						if (turn == 1) {
							turn = 2;

						} else {
							turn = 1;
						}

						buildGameTree(node, turn, depth - 1);
					}
				}
			}
		}
	}

	// Minimax with Alpha-Beta pruning heuristic function
	private double minMax(TreeNode node, int turn, double alpha, double beta, boolean max) {
		if (node.children == null || node.children.isEmpty()) {
			node.score = random(node.board, turn);
			return node.score;
		}

		// opposite player's turn
		if (turn == 1) {
			turn = 2;

		} else {
			turn = 1;
		}

		double value;
		if (max) {
			value = -Double.MAX_VALUE; // fail-soft
//			value = alpha; // fail-hard
			for (TreeNode child : node.children) {
				value = Math.max(value, minMax(child, turn, alpha, beta, false));
//				alpha = Math.max(value, alpha); // fail-hard

				if (beta < value) {
//					System.out.println("β cutoff: " + beta + "<" + value + " on depth: " + depth);
					break;
				}

				alpha = Math.max(value, alpha); // fail-soft
			}

		} else {
			value = Double.MAX_VALUE; // fail-soft
//			value = beta; // fail-hard
			for (TreeNode child : node.children) {
				value = Math.min(value, minMax(child, turn, alpha, beta, true));
//				beta = Math.min(value, beta); // fail-hard

				if (alpha > value) {
//					System.out.println("α cutoff: " + alpha + ">" + value + " on depth: " + depth);
					break;
				}

				beta = Math.min(value, beta); // fail-soft
			}
		}

		return value;
	}

	// Negamax with Alpha-Beta pruning heuristic function
	private double negamax(TreeNode node, int turn, double alpha, double beta, int color) {
		if (node.children == null || node.children.isEmpty()) {
			node.score = (double) color * random(node.board, turn);
			return node.score;
		}

		// opposite player's turn
		if (turn == 1) {
			turn = 2;

		} else {
			turn = 1;
		}

		double value = -Double.MAX_VALUE;
		for (TreeNode child : node.children) {
			value = Math.max(value, -negamax(child, turn, -beta, -alpha, -color));
			alpha = Math.max(alpha, value);

			if (alpha > beta) {
//				System.out.println("α-β cutoff: " + alpha + ">" + value + " on depth: " + depth);
				break;
			}
		}

		return value;
	}

	private double[] αβParallel(ArrayList<TreeNode> tree, int turn) {
		int length = tree.size();
		Vector<Future<Double>> scores = new Vector<Future<Double>>(length);
		scores.setSize(length);
		ExecutorService exec = Executors.newFixedThreadPool(length);
		try {
			for (int i = 0; i < length; i++) {
				TreeNode child = tree.get(i);
				Future<Double> result = exec.submit(new Callable<Double>() {
					@Override
					public Double call() { // heuristic function
						// Minimax with Alpha-Beta pruning
//						return minMax(child, turn, -Double.MAX_VALUE, Double.MAX_VALUE, true);

						// Negamax with Alpha-Beta pruning
						return negamax(child, turn, -Double.MAX_VALUE, Double.MAX_VALUE, 1);

					}
				});
				scores.set(i, result);
			}
		} finally {
			exec.shutdown();
		}

		double[] values = new double[length];
		for (int i = 0; i < length; i++) {
			try {
				values[i] = scores.get(i).get();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return values;
	}

	private TreeNode αβpruning(TreeNode lastAINode, TreeNode lastPlayerNode) {
		diffLevel = 53;
		int depth = 3; // if (depth > 3 && diffLevel == 75) it's slow

		// create game tree
		TreeNode parent = lastPlayerNode;
		buildGameTree(parent, aiplayer, depth); // aiplayer's move is in parent.children

		// call heuristic function in parallel
		int turn = aiplayer;
		if (turn == 1) { // opposite player's turn
			turn = 2;
		} else {
			turn = 1;
		}
		
		// PLAY!!!
		double[] scores = αβParallel(parent.children, turn);

//		// call heuristic function in parallel, for n best nodes
		int n = 7;
		double[] bests = new double[n];
		Arrays.fill(bests, -Double.MAX_VALUE);
		int[] indixes = new int[n];
		ArrayList<TreeNode> children = parent.children;
		maxs(scores, bests, indixes, 0);
		children = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			children.add(parent.children.get(indixes[i]));
		}

		diffLevel = 313;
		scores = αβParallel(children, turn); // children.size() == n

		for (int i = 0; i < bests.length; i++) {
			System.out.println("best: " + bests[i] + " vs " + scores[i]);
		}

		// get the best node
		n = 1;
		bests = new double[n];
		Arrays.fill(bests, -Double.MAX_VALUE);
		indixes = new int[n];
		maxs(scores, bests, indixes, 0);

		double best = bests[0];
		int index = indixes[0];

		System.out.println(best);
		System.out.println(index);

		TreeNode bestNode = children.get(index);
		bestNode.score = best;

		return bestNode;
	}

	private double mctsAlgorithm(MCTSTreeNode parent) {
		// More time is allotted per move so that the AI can pick the best possible move
		// Thus, it can win the game faster
		long timeLimit = (long) 15000.0; // milliseconds 
		
		MCTSTreeNode node;

		long time = System.currentTimeMillis();
		long deadline = time + timeLimit;
		int simulations = 100000;
		int iterations = 0;
		
		boolean winner;
		while (iterations < simulations) {
			// select + expand
			node = parent.select();

			// playout + backpropagate
			winner = node.playout(); // simulation
			node.backpropagate(winner);

			// resources limits
			time = System.currentTimeMillis();
			iterations += 1;
			
			if (time > deadline) {
				System.out.println(
						"time: " + (deadline - time + timeLimit) + " milliseconds\n"
						+ "iterations: " + iterations);
				break;
			}
		}
		
		return 1.0; // useless value (see mctsParallel())
	}

	private MCTSTreeNode mctsParallel(MCTSTreeNode parent) {
		// expand parent children
		parent.children = new ArrayList<>();
		MCTSTreeNode node;
		int length = parent.moves.size();
		for (int i = 0; i < length; i++) {
			node = parent.expand();
			
//			int x = node.playLocation.x;
//			int y = node.playLocation.y;
//			System.out.println("Location: " + y + ":" + x);
//			
			// game over
			if (node.leaf) {
//				System.out.println("leaf node!!!");
				
				node.score = 1.0;
				return node;
			}
		}
		Collections.shuffle(parent.children);
		System.out.println("Parent's children.size(): " + parent.children.size());

		// There is a race condition in the process of backpropagating parent.visits,
		// but it will not affect the result. This is because parent.visits is only
		// used to select the best child in parent.children. Since we are exploring
		// all of the children, there is no need to update this property. As a result,
		// it is safe to "child.parent = null" since it is not necessary."
		Vector<Future<Double>> results = new Vector<Future<Double>>(length);
		results.setSize(length);
		ExecutorService exec = Executors.newFixedThreadPool(length);
		try {
			for (int i = 0; i < length; i++) {
				MCTSTreeNode child = parent.children.get(i);
				child.parent = null; // not used, we are exploring all of the children

				Future<Double> result = exec.submit(new Callable<Double>() {
					@Override
					public Double call() { // heuristic function
						return mctsAlgorithm(child);
					}
				});
				results.set(i, result);
			}
		} finally {
			exec.shutdown();
		}

		// We can read the children from results, but they are already present in
		// parent.children.
		// Instead, we can simply wait for the thread to complete.
		// read values
		double[] values = new double[length];
		for (int i = 0; i < length; i++) {
			try {
				values[i] = results.get(i).get(); // wait for the thread to complete
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		node = parent.bestChild(false); // not exploring
		return node;
	}

	// Monte Carlo Tree Search heuristic function
	private TreeNode mcts(TreeNode lastAINode, TreeNode lastPlayerNode) {

		int[][] B = lastPlayerNode.board; // parent.children will copy their parent's board
		MCTSTreeNode parent = new MCTSTreeNode(B);
		parent.player = aiplayer;
		parent.playLocation = lastPlayerNode.playLocation;

		// PLAY!!!
		MCTSTreeNode mctsBestNode = mctsParallel(parent);

		// create TreeNode from MCTSTreeNode
		TreeNode bestNode = new TreeNode();
		bestNode.score = mctsBestNode.score;
		bestNode.playLocation = mctsBestNode.playLocation;

		// debug
		int x = bestNode.playLocation.x;
		int y = bestNode.playLocation.y;
		System.out.println("Play location: " + y + " : " + x);

		return bestNode;
	}
	
	class MCTSTreeNode {
		double sqrt2 = Math.sqrt(2);

		public ArrayList<MCTSTreeNode> children = new ArrayList<>();
		public MCTSTreeNode parent;
		public double score;
		public int[][] board;
		public Location playLocation;

		public int player;
		public int wins = 0;
		public int visits = 0;
		public ArrayList<Location> moves = new ArrayList<>();
		public boolean leaf;

		public MCTSTreeNode(int[][] board) {
			this.board = board;

			int length = board.length;
			ArrayList<Location> empty = new ArrayList<>();
//			int total = 0;
//			int free = 0;
//			int full = 0;
			for (int x = 0; x < length; x++) {
				for (int y = 0; y < length; y++) {
					if (board[x][y] == 0) {
						empty.add(new Location(y, x));
						
//						free++;
					} 
//					else {
//						full++;
//					}
					
//					total++;
				}
			}
//			System.out.println(total + "=" + free + "+" + full);
//			System.out.println("empty: " + empty.size());

			Collections.shuffle(empty);
			moves = empty;
		}

		private MCTSTreeNode select() {
			MCTSTreeNode node = this;

			while (!node.leaf) { // game over, do not expand node
				if (node.moves.size() > 0) {
//					System.out.println("expand node...");

					return node.expand();

				} else {
					node = node.bestChild(true);
				}
			}
			return node;
		}

		public MCTSTreeNode expand() {
			int[][] C = copy(this.board); // do not play on parent's node

			Location move = this.moves.remove(0);
			int x = move.x;
			int y = move.y;

			C[y][x] = this.player;

//			System.out.println("Child move: " + y + " : " + x);

			MCTSTreeNode child = new MCTSTreeNode(C);
			child.parent = this;
			child.playLocation = move;
			
			// game over
			child.leaf = playerWon(child.board, this.player);

//			if (child.leaf) {
//				System.out.println("Child leaf: " + y + " : " + x);
//			}

			int turn = this.player;
			if (turn == 1) { // opposite player's turn
				turn = 2;
			} else {
				turn = 1;
			}
			child.player = turn;

			this.children.add(child);

			return child;
		}

		public boolean playout() {
			// find how many empty board cells are there and fill them
			int length = this.board.length;
			ArrayList<Location> empty = new ArrayList<>();
			for (int x = 0; x < length; x++) {
				for (int y = 0; y < length; y++) {
					if (board[x][y] == 0) {
						empty.add(new Location(y, x));
					}
				}
			}

			int[][] G = copy(board); // copy board
			int turn = this.player;
			int x;
			int y;

			// shuffle empty cells before iterating and filling them
			Collections.shuffle(empty);
			for (Location e : empty) {
//				System.out.println("fill: " + turn);
				x = e.x;
				y = e.y;

				if (turn == 1) {
					turn = 2;

				} else {
					turn = 1;
				}

				G[y][x] = turn;
			}

			if (playerWon(G, aiplayer)) { // checks if computer won
				return true;
			}

			return false;
		}

		public void backpropagate(boolean winner) {
			this.visits += 1;

			if (winner) {
				this.wins += 1;
			}

			if (this.parent != null) {
				this.parent.backpropagate(winner);
			}

//			System.out.println("select: parent: break!" + this.visits);
		}

		public MCTSTreeNode bestChild(boolean explore) {
			MCTSTreeNode node = this;

			double[] scores = new double[node.children.size()];
			double score;
			double w;
			double n;
			double N;
			MCTSTreeNode child;
			for (int i = 1; i < node.children.size(); i++) {
				child = node.children.get(i);

				// c * Math.sqrt(Math.log(N) / n);
				w = (double) child.wins;
				n = (double) child.visits;
				N = (double) node.visits;

				try {
					if (explore) { // Upper Confidence bounds applied to Trees (UTC)
//						double c = Math.sqrt(2);
//						double x = w / n;
//						double y = Math.log(N);
//						double z = y / n;
//						score = x + c * Math.sqrt(z);

						score = (w / n) + (sqrt2 * (Math.sqrt(Math.log(N) / n)));
						child.score = score;

//						System.out.println("Score: " + score);
//						System.out.println("w: " + w);
//						System.out.println("n: " + n);
//						System.out.println("N: " + N);
//						System.out.println("sqrt2: " + sqrt2);

					} else {
						score = w / n;
						child.score = score;

//						System.out.println("Score: " + score);
//						System.out.println("w: " + w);
//						System.out.println("n: " + n);
					}

				} catch (ArithmeticException e) {
					e.printStackTrace();

					throw new ArithmeticException("wins: " + child.wins + "visits: " + child.visits);
				}

				scores[i] = score;
			}

			double[] bests = new double[1];
			Arrays.fill(bests, -Double.MAX_VALUE);
			int[] indixes = new int[1];
			maxs(scores, bests, indixes, 0);

			int index = indixes[0];

			return node.children.get(index);
		}
	}

	// Deductible function
	private Double deductible(TreeNode lastAINode, TreeNode lastPlayerNode) {
		// TODO Auto-generated method stub
		return null;
	}

	private TreeNode play(TreeNode lastAINode, TreeNode lastPlayerNode) {
		// Alpha-Beta pruning
//		return αβpruning(lastAINode, lastPlayerNode);

		// MCTS
		return mcts(lastAINode, lastPlayerNode);

		// Deductible
//		return deductible(lastAINode, lastPlayerNode);
	}

	/**
	 * Chooses a location for the next play by this AI.
	 *
	 * @param board The board state used in determining the play location
	 * @param last  The last play in the game, as made by the opponent
	 *
	 * @return A location to play at, chosen using MCTS
	 */
	public Location getPlayLocation(int[][] board, Location last) {
		// first move
		if (last == null || last.x == -1)
			return new Location(1, 1); // it's a strong move for the aiplayer

		long t = System.currentTimeMillis();

		TreeNode lastAINode = new TreeNode();
		lastAINode.playLocation = lastPlayed; // != last, last is player's last move

		TreeNode lastPlayerNode = new TreeNode();
		lastPlayerNode.playLocation = last;
		lastPlayerNode.board = board;
		
		// PLAY!!!
		TreeNode bestNode = play(lastAINode, lastPlayerNode); 

		int x = bestNode.playLocation.x;
		int y = bestNode.playLocation.y;
		System.out.println("best node: " + bestNode.score + " at: " + y + ":" + x);

		System.out.println(System.currentTimeMillis() - t);

		lastPlayed = bestNode.playLocation;
		return bestNode.playLocation;
	}
}

///**
// * Calculates the n-completion value for a given board.
// *
// * @param board   The board to calculate for
// * @param player  The player to calculate for
// * @param l       The current visited Location on the board
// * @param visited An ArrayList of Locations already visited
// * @param count   The number of spaces visted already
// *
// * @return The n-completion value for a path between l and the edge
// */
//private int calcN(int[][] board, int player, Location l, ArrayList<Location> visited, int count) {
//	if (count < minLen && ((player == 1 && l.y == 6) || (player == 2 && l.x == 6))) {
//		minLen = count;
//
//		return count;
//	} else if ((player == 1 && l.y == 6) || (player == 2 && l.x == 6)) {
//		return count;
//	}
//
//	if (player == 1 && count + (6 - l.y) >= minLen) {
//		return 999;
//	} else if (player == 2 && count + (6 - l.x) >= minLen) {
//		return 999;
//	}
//
//	ArrayList<Location> adj = l.getAdjacentLocations();
//	ArrayList<Bridge> bridges = l.getBridges();
//
//	ArrayList<Location> v = Utils.ALCopy(visited);
//
//	v.add(new Location(l.x, l.y));
//
//	int min = 999;
//
//	for (int i = 0; i < bridges.size(); i++) {
//		Bridge b = bridges.get(i);
//
//		boolean canUseBridge = board[b.mids.get(0).y][b.mids.get(0).x] == 0
//				&& board[b.mids.get(1).y][b.mids.get(1).x] == 0;
//		if (canUseBridge && !Utils.ALContains(v, b.l1) && (board[b.l1.y][b.l1.x] == player
//				|| board[b.l1.y][b.l1.x] == 0) /* && (board[b.l2.y][b.l2.x]==player || board[b.l2.y][b.l2.x]==0) */) {
//			int val = calcN(board, player, b.l1, v, count);
//
//			if (val < min) {
//				min = val;
//			}
//		}
//	}
//
//	for (int i = 0; i < adj.size(); i++) {
//		Location loc = adj.get(i);
//
//		if (!Utils.ALContains(v, loc) && (board[loc.y][loc.x] == player || board[loc.y][loc.x] == 0)) {
//			int val = 999;
//
//			if (board[loc.y][loc.x] == player) {
//				val = calcN(board, player, loc, v, count);
//			} else {
//				val = calcN(board, player, loc, v, count + 1);
//			}
//
//			if (val < min) {
//				min = val;
//			}
//		}
//	}
//
//	return min;
//}
//
///**
// * Calculates the n-completion value for a game board state. This is (more or
// * less) a nice wrapper around calcN.
// *
// * @param board The board to calculate based on
// *
// * @return The n-completion value for the board state provided
// */
//public double calcVal_(int[][] board) {
//	int opp = 1;
//	if (aiplayer == 1) {
//		opp = 2;
//	}
//
//	minLen = 49;
//	double maxno = 999;
//	double minnp = 999;
//
//	for (int i = 0; i < board.length; i++) {
//		if (board[i][0] != opp) {
//			int initCountP = 1;
//			int initCountO = 1;
//			if (board[i][0] == aiplayer) {
//				initCountP = 0;
//			}
//			if (board[i][0] == opp) {
//				initCountO = 0;
//			}
//
//			Location pLoc, oLoc;
//			if (aiplayer == 1) {
//				pLoc = new Location(i, 0);
//				oLoc = new Location(0, i);
//			} else {
//				pLoc = new Location(0, i);
//				oLoc = new Location(i, 0);
//			}
//
//			double no = (double) calcN(board, opp, oLoc, new ArrayList<Location>(), initCountO);
//			minLen = 49;
//
//			if (no < maxno) {
//				maxno = no;
//			}
//		}
//	}
//
//	return maxno;
//}
