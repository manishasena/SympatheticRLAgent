from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class SympathyMultiGrid(MultiGridEnv):
    mission = "collect green squares"
    metadata = {}

    def __init__(self, *args, n_clutter=None, clutter_density=None, randomize_goal=False, **kwargs):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        super().__init__(*args, **kwargs)

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = n_clutter

        self.randomize_goal = randomize_goal

        # self.reset()


    def _gen_grid(self, width, height):

        #List of food positions of other agent
        self.agents[0].foodpos = []
        self.agents[1].foodpos = []
        self.agents[0].door_opened = False
        #List of wall positions
        self.walls = []
        # Grid dimensions
        self.gamewidth = width
        self.gameheight = height

        # Wall positions
        wall_pos = [[6,1],[6,2],[4,2],[3,2],[2,2],[2,4],[2,5],[2,6],[4,6],[5,6],[3,4],[5,4],[6,4],[6,0]]
        human_food_positions = [[7,1],[4,3],[1,2],[1,5],[1,7],[3,7],[5,7],[2,1],[5,1],[2,3],[4,5],[7,4],[7,6],[6,6]]
        robot_food_positions = [[1,1],[2,7],[1,4],[4,4],[4,7],[5,5],[6,7],[4,1],[5,2],[5,3],[3,3],[1,3],[3,6],[7,3],[7,5],[1,6],[6,3],[7,7],[6,5]]
        door_position = [[7,2]]

        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        if getattr(self, 'randomize_goal', True):

            for k in range(len(door_position)):
                pos = np.array(door_position[k])
                self.try_place_obj(Door(color="green",state=2),pos)

            #for k in range(len(key_position)):
            #    pos = np.array(key_position[k])
            #    self.try_place_obj(Key(color="green",reward = 0),pos)

            for k in range(len(robot_food_positions)):
                pos = np.array(robot_food_positions[k]) #self.place_obj(FoodOther(color="orange", reward=0), max_tries=1000)
                self.try_place_obj(Food(color="red", reward=10), pos)
                self.agents[0].foodpos.append(pos)

            for k in range(len(human_food_positions)):
                pos = np.array(human_food_positions[k]) #self.place_obj(FoodOther(color="orange", reward=0), max_tries=1000)
                self.try_place_obj(FoodOther(color="yellow", reward=0), pos)
                self.agents[1].foodpos.append(pos)

        else:
            self.put_obj(Food(color="green", reward=1), width - 2, height - 2)

        #for i in wall_pos:
        #    self.walls.append(pos)
        if getattr(self, 'n_clutter', 0) > 0:
            for i in range(len(wall_pos)):
                #pos = self.place_obj(Wall(), max_tries=100)
                pos = np.array(wall_pos[i])
                self.try_place_obj(Wall(), pos)
                self.walls.append(pos)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)
