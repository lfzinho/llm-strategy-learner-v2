# Game Rules
You are playing a social game with one other player where your objective is to maximize your own points. The game is divided into rounds. For every round, you have two options of action: action Alpha or action Beta. The other player has the same options. If both of you choose action Alpha, both of you receive 3 points each. If only you choose option Beta while the other player chooses option Alpha, you get 5 points and he gets 0 points. Otherwise, if only he chooses option Beta while you choose option Alpha, he gets 5 points and you get 0 points. If both of you choose option Beta, both of you receive 1 point each. In a table:

- You: Alpha, Him: Alpha -> You: +3 points, Him: +3 points
- You: Alpha, Him: Beta -> You: +0 points, Him: +5 points
- You: Beta, Him: Alpha -> You: +5 points, Him: +0 points
- You: Beta, Him: Beta -> You: +1 point, Him: +1 point

The game continues for a predetermined number of rounds given at the start. The game has no winner or loser, and the goal is to maximize your own points while considering the other player's potential choices. 

# Evaluation of Success
You will player a series of games with different players. Your total success will be evaluated based on the total points you accumulate across all games. The more points you earn, the better your performance in the game.

Keep in mind: you don't have to win against the other player; your goal is simply to maximize your own points. You should consider the other player's potential strategies and adapt your choices accordingly to achieve the highest possible score for yourself.

# Strategy Tips
1. **Analyze the optimal responses**: There are some obvious moves to play against very predictable players. If the other player only chooses Alpha, choosing Beta could yield higher points for you. Not conversely, if the other player tends to choose Beta, choosing Beta is the best option to win marginal points.
2. **Think about the other player strategy**: Not every player will play the same way every round. Some may switch strategies based on previous outcomes or on what you have done in the past. Pay attention to patterns in their choices and adjust your strategy accordingly. Think about how your choices might influence their future decisions and how will this affect your own point accumulation in the long run.
3. **Try to predict the other player's behavior**: To find out the other player's tendencies, you can try to predict their behavior based on their previous choices. Sometimes, the other player is playing a repeated action only becaise you are too. If you don't know their tendencies, you can try to test their reactions by varying your own choices. For example, if you notice that they tend to choose Alpha after you choose Beta, you might want to switch to Alpha to avoid giving them an advantage.
4. **Consider the long-term implications**: While it may be tempting to choose Beta for immediate gain, consider how your choices might affect the other player's future decisions. If you consistently choose Beta, the other player may adapt and start choosing Beta more often, which could lead to a lower overall score for both players in the long run. Striking a balance between immediate gains and long-term strategy is crucial for maximizing your total points across all rounds.

# Rounds
## Mode 1
Every game has exactly 20 rounds. Keep in mind that the best moment to explore and analyze the other player's strategy is in the first few rounds. After that, you should focus on maximizing your points based on the information you have gathered. In the last round, you can prioritize your own points without worrying about the other player's future choices.

## Mode 2
Every game has exactly 20 rounds. Keep in mind that the best moment to explore and analyze the other player's strategy is in the first few rounds. After that, you should focus on maximizing your points based on the information you have gathered.

## Mode 3
Every game has exactly 20 rounds.

## Mode 4
Every game has the number of rounds determined by a Normal(20, 1) distribution. This means that you won't know the exact number of rounds in advance, but it will be around 20 with some variation.

## Mode 5
Every game has a randomized number of rounds. You won't know the exact number of rounds in advance, and it could be significantly anywhere from 1 to infinity.

# Classical Strategies
1. **Always Alpha**: This strategy involves consistently choosing action Alpha in every round.
- How to play against this strategy: If the other player is always choosing Alpha, you can maximize your points by choosing Beta, as it will yield you 5 points while they get 0.

2. **Always Beta**: This strategy involves consistently choosing action Beta in every round.
- How to play against this strategy: If the other player is always choosing Beta, you can maximize your points by choosing Beta as well, as it will yield you 1 point while they also get 1 point. Choosing Alpha in this case would give you 0 points.

3. **Alpha Mimic**: This strategy starts with choosing Alpha and then mimics the other player's previous choice in subsequent rounds.
- How to play against this strategy: If the other player is using the Alpha Mimic strategy, your best approach is to choose Alpha always. This will ensure that you both receive 3 points each round, maximizing your score.

4. **Beta Mimic**: This strategy starts with choosing Beta and then mimics the other player's previous choice in subsequent rounds.
- How to play against this strategy: If the other player is using the Beta Mimic strategy, your best approach is to choose Alpha always. This will ensure that you receive 3 points each round. Keep in mind: even if the other player will have an score advantage in the game, this does not matter as your goal is to maximize your own points.

5. **Random Choice**: This strategy involves randomly choosing between Alpha and Beta in each round.
- How to play against this strategy: If the other player is using a Random Choice strategy, your best approach is to choose Beta always. Because your choices do not affect the other player's choices, you can choose Beta consistently to maximize your points, ending always in the best scenario regardless of the other player's choices.

6. **Beta Resentful**: This strategy involves choosing Alpha at first, but if the other player ever chooses Beta, they will switch to Beta in the next round and continue to choose Beta for the rest of the game.
- How to play against this strategy: If the other player is using the Beta Resentful strategy, your best approach is to keep choosing Alpha until they choose Beta. Once they switch to Beta, you can then choose Beta as well to ensure that you both receive 1 point each round.

7. **Zen Alpha Mimic**: This strategy involves choosing Alpha as default. If the other player chooses Beta once, they will keep choosing Alpha for the next round, after all "that could have been a mistake". If the other player chooses Beta again, than they will play Beta. If, finally, the other player comes back to Alpha, they will get back to playing Alpha as default.
- How to play against this strategy: If the other player is using the Zen Alpha Mimic strategy, your best approach is to alternate between Alpha and Beta. Choose Beta once to get more points, then switch back to Alpha to avoid triggering their Beta response. This way, you can maximize your points while keeping the other player in a state of Alpha plays.

# Conclusion
## Mode 1
This game is against a player using the strategy {strategy name}: {strategy description}. {counter strategy description}. Remember, your goal is to maximize your own points while considering the other player's potential choices.

## Mode 2
This game is against a player using the strategy {strategy name}: {strategy description}. Remember, your goal is to maximize your own points while considering the other player's potential choices.

## Mode 3
This game is against a player using one of the classical strategies. Remember, try to analyze their behavior and identify their strategy. After that, use the appropriate counter strategy to maximize your own points while considering the other player's potential choices.

## Mode 4
You are playing a game against another player. Remember, try to analyze their behavior and identify their strategy. After that, use the appropriate counter strategy to maximize your own points while considering the other player's potential choices.

# Middle of the Game Modifier
## Beginning of the Game
This is the first round of the game. You have no information about the other player's strategy yet. You can choose either Alpha or Beta. Keep in mind that your choice may influence the other player's future decisions, so consider the potential long-term implications of your choice.

## Answer to Choice
You chose {your choice}. The other player chose {other player's choice}. You received {your points} points, and the other player received {other player's points} points.

Now the next round will begin.

## Middle of the Game
This is the round {round number}. There are {remaining rounds} rounds left in the game. The game history so far is as follows:

- Round 1: You chose {your choice}, Other player chose {other player's choice}.
- Round 2: You chose {your choice}, Other player chose {other player's choice}.
- Round 3: You chose {your choice}, Other player chose {other player's choice}.

You can choose either Alpha or Beta. If you have identified the other player's strategy, you can use the appropriate counter strategy to maximize your own points. If you haven't identified their strategy yet, consider experimenting with different choices to gather more information about their behavior.

