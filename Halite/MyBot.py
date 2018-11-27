import hlt

from hlt import constants

from hlt.positionals import Direction

import random

import logging

""" <<<Game Begin>>> """

game = hlt.Game()

# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("MyPythonBot")


logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

""" <<<Game Loop>>> """

# make ships go back to yard to drop of halite once full
ship_states = {}
while True:
    
    game.update_frame()
    me = game.me
    game_map = game.game_map

    
    command_queue = []
    #get direction order. a list
    direction_order = [Direction.North, Direction.South, Direction.East, Direction.West]
    #

    for ship in me.get_ships():
        if ship.id not in ship_states:
            ship_states[ship.id] = "collecting"
            
        #each ship moves only one time each turn , ship positions, get current ship position
        position_options = ship.position.get_surrounding_cardinals() + [ship.position]
        #movement mapped to exact coordinate
        position_dict = {}
        #acutal movement associated with how much halite
        halite_dict = {}
        #lets collect halites for postions
        #lets populate them
        for n , direction in enumerate(direction_order):
            position_dict[direction] = position_options[n]
        for direction in position_dict:
            position = position_dict[direction]
            halite_amount = game_map[position].halite_amount
            #maps to coordinates
            if position_dict[direction]not in postion_choices:
                if direction == Direction.Still: 
                    halite_dict[direction] = halite_amount*3

                else:
                    halite_dict[direction] = halite_amount

            else: 
                logging.info('attempting to move to same spot\n')

        if ship_states[ship.id] == "depositing":
            #naivigate ship to shipyard
            move = game_map.naive_naigate(ship, me.shipyard.position)
            position_choices.append(position_dict[move])
            command_queue.append(ship.move(move))
            # move from shipyard after droppping
            if move == Direction.Still:
                ship_states[ship.id] = "collecting"

        #  most eise collection
        elif ship_states[ship.id] =="collecting":
            directional_choice = max(halite_dict , key = halite_dict.get)
            position_choices.append(position_dict[directional_choice])
            command_queue.append(ship.move(game_map.naive_navigate(ship, position_dict[directional_choice]))

            # change to check alomost to max limit 
            if ship.halite_amount > constants.MAX_HALITE * .95:
                ship_states[ship.id] ="depositing"
                
    
    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    game.end_turn(command_queue)

