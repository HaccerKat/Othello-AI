def mcts(root, model, debug=False, gameplay = False, num_simulations=200, exploration_constant=2):
    for i in range(num_simulations):
        node = root
        while not node.game_ends():
            # Not fully expanded, so expand
            if not node.next_boards or node.current_child < len(node.next_boards):
                # print("EXPAND")
                node = node.expand(model)
                break
            # Fully expanded, select child with highest UCT value
            else:
                # print("SELECT")
                node = node.select(exploration_constant)

        node.backpropagate(model)

    result = root.get_next_board(gameplay)
    if debug:
        root.print()

    return result