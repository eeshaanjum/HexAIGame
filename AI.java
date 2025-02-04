/* 
 * BSD License
 */

/**
 * An interface for AI players, to be implmented by various subclasses.
 *
 */
public abstract class AI {
	public abstract int getPlayerCode();
	public abstract Location getPlayLocation(int[][] board, Location lastPlay);
}
