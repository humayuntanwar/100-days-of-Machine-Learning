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
shates_of_ships = {}
while True:
    
    game.update_frame()
    me = game.me
    game_map = game.game_map

    
    command_queue = []
    #get direction order. a list
    order_direction = [Direction.North, Direction.South, Direction.East, Direction.West]
    #
    position_choices = []

    for ship in me.get_ships():
        if ship.id not in shates_of_ships:
            shates_of_ships[ship.id] = "collecting"
        
        if ship_states[ship.id] == "collecting":
    
            #each ship moves only one time each turn , ship positions, get current ship position
            options_positions = ship.position.get_surrounding_cardinals() + [ship.position]
            #movement mapped to exact coordinate
            position_d = {}
            #acutal movement associated with how much halite
            halite_d = {}
            #lets collect halites for postions
            #lets populate them
            for n , direction in enumerate(order_direction):
                position_d[direction] = options_positions[n]
            for direction in position_d:
                position = position_d[direction]
                halite_amount = game_map[position].halite_amount
                #maps to coordinates
                if position_d[direction]not in postion_choices:
                    if direction == Direction.Still:
                        halite_amount *= 4
                    halite_dict[direction] = halite_amount

            directional_choice = max(halite_d,key = halite_d.get)
            position_choices.append(position_d[directional_choice])
            command_queue.append(ship.move(game_map.naive_navigate(ship, position_d[directional_choice]))

            # change to check alomost to max limit 
            if ship.halite_amount > constants.MAX_HALITE * .95:
                shates_of_ships[ship.id] ="depositing"

        elif shates_of_ships[ship.id] == "depositing":
            #naivigate ship to shipyard
            move = game_map.naive_naigate(ship, me.shipyard.position)
            upcoming_position = ship.position + Position(*move)
            if upcoming_position not in position_choices:

                position_choices.append(position_d[move])
                command_queue.append(ship.move(move))
                # move from shipyard after droppping
                if move == Direction.Still:
                    shates_of_ships[ship.id] = "collecting"
            else:
                position_choices.append(ship.position)
                command_queue.append(ship.move(game_map.naive_navigate(ship, ship.position+Position(*Direction.Still))))

                
    
    # ship costs 1000, dont make a ship on a ship or they both sink
    if len(me.get_ships()) < math.ceil(game.turn_number / 25):
        if me.halite_amount >= 1000 and not game_map[me.shipyard].is_occupied:
            command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
    game.end_turn(command_queue)

