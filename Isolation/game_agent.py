import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    import math

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    width = game.width
    height = game.height
    (player_h, player_w) = game.get_player_location(player)

    if width % 2:
        w_center = int(width / 2)
        dist_x = abs(player_w - w_center)
    else:
        if player_w < width / 2:
            w_center = int(width / 2) - 1
            dist_x = abs(player_w - w_center)
        else:
            w_center = int(width / 2)
            dist_x = abs(player_w - w_center)

    if height % 2:
        h_center = int(height / 2)
        dist_y = abs(player_h - h_center)
    else:
        if player_h < height / 2:
            h_center = int(height / 2) - 1
            dist_y = abs(player_h - h_center)
        else:
            h_center = int(height / 2)
            dist_y = abs(player_h - h_center)

    dist = float(math.sqrt(dist_x * dist_x + dist_y * dist_y))
    bonus = float(2 / (dist + 1))
    return float(len(player_moves) - len(opponent_moves) + bonus)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    import math

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    width = game.width
    height = game.height
    (player_h, player_w) = game.get_player_location(player)

    if width % 2:
        w_center = int(width / 2)
        dist_x = abs(player_w - w_center)
    else:
        if player_w < width / 2:
            w_center = int(width / 2) - 1
            dist_x = abs(player_w - w_center)
        else:
            w_center = int(width / 2)
            dist_x = abs(player_w - w_center)

    if height % 2:
        h_center = int(height / 2)
        dist_y = abs(player_h - h_center)
    else:
        if player_h < height / 2:
            h_center = int(height / 2) - 1
            dist_y = abs(player_h - h_center)
        else:
            h_center = int(height / 2)
            dist_y = abs(player_h - h_center)

    dist = float(math.sqrt(dist_x * dist_x + dist_y * dist_y))
    bonus = 1 - float(2 / (dist + 1))
    return float(len(player_moves) - len(opponent_moves) + bonus)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    import math

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    width = game.width
    height = game.height
    (player_h, player_w) = game.get_player_location(player)

    if width % 2:
        w_center = int(width / 2)
        dist_x = abs(player_w - w_center)
    else:
        if player_w < width / 2:
            w_center = int(width / 2) - 1
            dist_x = abs(player_w - w_center)
        else:
            w_center = int(width / 2)
            dist_x = abs(player_w - w_center)

    if height % 2:
        h_center = int(height / 2)
        dist_y = abs(player_h - h_center)
    else:
        if player_h < height / 2:
            h_center = int(height / 2) - 1
            dist_y = abs(player_h - h_center)
        else:
            h_center = int(height / 2)
            dist_y = abs(player_h - h_center)

    dist = float(math.sqrt(dist_x * dist_x + dist_y * dist_y))
    bonus = 1 - float(4 / (dist + 1))
    return float(len(player_moves) - len(opponent_moves) + bonus)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score_2, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            return best_move

        # Return the best move from the last completed search iteration.
        return best_move

    def min_play(self, game, depth):

        if depth == 0:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        # Checks to see whether the game has finished
        if len(legal_moves) == 0:
            return float("inf")
        best_score = float('inf')
        best_move = legal_moves[0]
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            clone = game.forecast_move(move)
            score = self.max_play(clone, depth - 1)

            if score < best_score:
                best_score = score
                best_move = move

        return best_score

    def max_play(self, game, depth):

        if depth == 0:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return float("-inf")
        best_score = float('-inf')
        best_move = legal_moves[0]

        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            clone = game.forecast_move(move)
            score = self.min_play(clone, depth - 1)

            if score > best_score:
                best_score = score
                best_move = move

        return best_score

    def minimax(self, game, depth):
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (-1, -1)
        best_move = legal_moves[0]
        best_score = float('-inf')
        # min_play and max_play functions only return scores. This is the first max
        # loop, which decides what move to choose based on these scores.
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            clone = game.forecast_move(move)
            score = self.min_play(clone, depth - 1)
            if score > best_score:
                best_move = move
                best_score = score

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        depth = 1
        while(depth < 25):
            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_move = self.alphabeta(game, depth)
                depth += 1

            except SearchTimeout:
                return best_move

        # Return the best move from the last completed search iteration
        return best_move



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (-1, -1)
        best_move = legal_moves[0]
        max_score = alpha
        best_score = alpha
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            clone = game.forecast_move(move)
            score = self.min_play(clone, depth - 1, best_score, beta)
            if score > best_score:
                best_move = move
                best_score = score
            if best_score >= beta:
                return best_move

        return best_move

    def min_play(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        if depth == 0:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return float("inf")
        best_score = beta
        best_move = legal_moves[0]
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            clone = game.forecast_move(move)
            score = self.max_play(clone, depth - 1, alpha, best_score)
            # It checks that best_score is the lowest value after every iteration.
            if score < best_score:
                best_score = score
                best_move = move

            if alpha >= best_score:
                return best_score

        return best_score

    def max_play(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        if depth == 0:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return float("-inf")
        best_score = alpha
        best_move = legal_moves[0]
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            clone = game.forecast_move(move)
            score = self.min_play(clone, depth - 1, best_score, beta)

            if score > best_score:
                best_score = score
                best_move = move

            if best_score >= beta:
                return best_score

        return best_score
